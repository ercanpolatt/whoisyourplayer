import cv2
import numpy as np
import glob
import os
import time
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── 1. Yapılandırma ve Model İndirme ─────────────────────────────────────
IMAGE_FOLDER = "futbolcular"
MODEL_PATH = "pose_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

# Model dosyası yoksa otomatik olarak indir
if not os.path.exists(MODEL_PATH):
    print("Model dosyası bulunamadı. Google sunucularından indiriliyor (pose_landmarker.task)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model başarıyla indirildi ve kaydedildi!")
    except Exception as e:
        print(f"Hata: Model dosyası indirilemedi: {e}")
        exit(1)

# Türkçe Karakterli Dosya İsimlerini Okunaklı Yapmak İçin Eşleştirme Tablosu
PLAYER_NAMES = {
    "abdulkerim.png": "ABDULKERIM BARDAKCI",
    "arda.png": "ARDA GULER",
    "baris.png": "BARIS ALPER YILMAZ",
    "ferdi.png": "FERDI KADIOGLU",
    "kenan.png": "KENAN YILDIZ",
    "kerem.png": "KEREM AKTURKOGLU",
    "merih.png": "MERIH DEMIRAL",
    "orkun.png": "ORKUN KOKCU",
    "ugurcan.png": "UGURCAN CAKIR"
}

# Poz karşılaştırmasında kullanılacak kritik vücut noktaları
POSE_INDICES = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# El şekli ve vücut duruşuna odaklanmak için omuz/dirsek/el bileği ağırlıkları
LANDMARK_WEIGHTS = {
    0: 1.0,   # Burun (kafa pozisyonu)
    11: 1.0,  # Sol Omuz
    12: 1.0,  # Sağ Omuz
    13: 1.5,  # Sol Dirsek
    14: 1.5,  # Sağ Dirsek
    15: 3.5,  # Sol El Bileği (el pozisyonu için çok kritik)
    16: 3.5,  # Sağ El Bileği (el pozisyonu için çok kritik)
    17: 2.0,  # Sol El Serçe Parmağı
    18: 2.0,  # Sağ El Serçe Parmağı
    19: 2.5,  # Sol El İşaret Parmağı
    20: 2.5,  # Sağ El İşaret Parmağı
}

# ── 2. Yardımcı Fonksiyonlar ───────────────────────────────────────────
def resize_and_pad(img, size=(380, 350), pad_color=(30, 25, 20)):
    """Görüntüyü en-boy oranını koruyarak yeniden boyutlandırır ve dolgu ekler."""
    h, w = img.shape[:2]
    sw, sh = size

    # En-boy oranını koru
    aspect = w / h
    if aspect > sw / sh:
        new_w = sw
        new_h = int(new_w / aspect)
    else:
        new_h = sh
        new_w = int(new_h * aspect)

    interp = cv2.INTER_AREA if h > sh or w > sw else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    # Dolguları ortalayarak yerleştir
    pad_h = sh - new_h
    pad_w = sw - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    return padded

def draw_text(img, text, x, y, font_scale=0.6, color=(255, 255, 255), thickness=1):
    """Gölgeli ve temiz metin çizer."""
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def draw_hud_card(canvas, pt1, pt2, title, border_color=(100, 100, 100), active=False):
    """Panel etrafına şık, yarı şeffaf cam görünümlü kart çizer."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    sub_img = canvas[y1:y2, x1:x2]
    rect = np.zeros(sub_img.shape, dtype=np.uint8)
    rect[:] = (15, 10, 5) if active else (5, 5, 5)
    canvas[y1:y2, x1:x2] = cv2.addWeighted(sub_img, 0.85, rect, 0.15, 0)
    
    color = border_color if active else (100, 100, 100)
    cv2.rectangle(canvas, pt1, pt2, color, 1 if not active else 2)
    
    if title:
        tw = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        cv2.rectangle(canvas, (x1, y1 - 18), (x1 + tw + 20, y1), color, -1)
        draw_text(canvas, title, x1 + 10, y1 - 5, 0.4, (255, 255, 255), 1)

def extract_normalized_landmarks(landmarks):
    """Vücut noktalarını omuz merkezine göre konumlandırır ve omuz genişliğine bölerek ölçekler (scale-invariant)."""
    # Sol Omuz (11), Sağ Omuz (12)
    ls = landmarks[11]
    rs = landmarks[12]
    
    mid_x = (ls.x + rs.x) / 2.0
    mid_y = (ls.y + rs.y) / 2.0
    
    # Omuz genişliği mesafesi
    scale = np.sqrt((ls.x - rs.x)**2 + (ls.y - rs.y)**2)
    if scale < 0.05:
        scale = 0.05
        
    normalized = {}
    for idx in POSE_INDICES:
        lm = landmarks[idx]
        nx = (lm.x - mid_x) / scale
        ny = (lm.y - mid_y) / scale
        normalized[idx] = (nx, ny, lm.visibility)
        
    return normalized

def compare_poses(user_pose, player_pose):
    """Kullanıcının canlı pozu ile futbolcu pozu arasındaki ağırlıklı uzaklığı bulup benzerlik yüzdesi çıkarır."""
    total_dist = 0.0
    total_weight = 0.0
    
    for idx in POSE_INDICES:
        if idx not in user_pose or idx not in player_pose:
            continue
            
        w = LANDMARK_WEIGHTS.get(idx, 1.0)
        ux, uy, _ = user_pose[idx]
        px, py, _ = player_pose[idx]
        
        dist = np.sqrt((ux - px)**2 + (uy - py)**2)
        total_dist += dist * w
        total_weight += w
        
    if total_weight == 0:
        return 0
        
    weighted_avg_dist = total_dist / total_weight
    
    # Mesafe 0.0 ise %100 benzerlik, 0.8 ve üzeri ise %0 benzerlik
    similarity = max(0, min(100, int((1.0 - (weighted_avg_dist / 0.8)) * 100)))
    return similarity

# ── 3. Fotoğraf Setini ve Poz Özelliklerini Yükle ───────────────────────
print("Fotoğraf seti yükleniyor ve poz özellikleri çıkartılıyor...")
images = {}
player_poses = {}

# MediaPipe Pose Landmarker Oluştur
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    running_mode=vision.RunningMode.IMAGE)

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    found_files = glob.glob(os.path.join(IMAGE_FOLDER, "*.png"))
    if not found_files:
        print(f"Hata: '{IMAGE_FOLDER}' klasöründe PNG bulunamadı. Lütfen klasör adını kontrol edin.")
        exit(1)

    for path in found_files:
        name = os.path.basename(path)
        try:
            with open(path, 'rb') as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is not None:
                # Ekranda göstereceğimiz kart görselini önceden hazırla
                img_preprocessed = resize_and_pad(img, (380, 350))
                images[name] = img_preprocessed
                
                # MediaPipe formatına çevirip pozu analiz et
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_preprocessed, cv2.COLOR_BGR2RGB))
                result = landmarker.detect(mp_image)
                
                if result.pose_landmarks:
                    player_poses[name] = extract_normalized_landmarks(result.pose_landmarks[0])
                    print(f"  -> {name} pozu basariyla indekslendi.")
                else:
                    print(f"  [!] Uyari: {name} gorselinde insan pozu tespit edilemedi!")
        except Exception as e:
            print(f"Hata: {name} dosyası yüklenirken sorun oluştu: {e}")

print(f"\nSistem Hazır! Başarıyla {len(images)} futbolcu yüklenip {len(player_poses)} poz indekslendi.")

# ── 4. Kamera ve HUD Döngüsü Başlatılıyor ───────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Durum Değişkenleri
gosterilen_player   = None
gosterilen_skor     = 0
son_eslesme_zamani  = 0.0
cooldown_zamani     = 2.0  # Eşleşmenin ekranda kalma süresi (sn)

# İskelet bağlantı hatları listesi
CONNECTIONS = [
    (11, 12),  # Omuzdan omuza
    (11, 13),  # Sol omuz - sol dirsek
    (12, 14),  # Sağ omuz - sağ dirsek
    (13, 15),  # Sol dirsek - sol bilek
    (14, 16),  # Sağ dirsek - sağ bilek
    (15, 17), (15, 19),  # Sol bilek - el noktaları
    (16, 18), (16, 20),  # Sağ bilek - el noktaları
]

window_name = "Kim Bu Futbolcu? | Who Is Your Player"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

print("\nKamera Açıldı!")
print("-> Kamera onunde bir futbolcunun ikonik gol sevincini taklit edin.")
print("-> Cikmak icin 'Q' tusuna basin.\n")

# MediaPipe Landmarker'ı CANLI modda veya IMAGE modunda kullanabiliriz.
# Tekil kareleri göndereceğimiz için IMAGE modu ile döngüde devam ediyoruz.
with vision.PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Hata: Kameradan görüntü alınamıyor!")
            break

        # Kullanıcının kendini aynada görüyor gibi hissetmesi için yatay çevirme
        frame = cv2.flip(frame, 1)

        # 1. Ana HUD Canvas'ı (1100x650) ve Gradyan Arka Planı oluştur
        canvas = np.zeros((650, 1100, 3), dtype=np.uint8)
        for x in range(1100):
            b = int(18 + (x / 1100) * 12)
            g = int(24 + (x / 1100) * 8)
            r = int(38 - (x / 1100) * 10)
            canvas[:, x] = (b, g, r)

        # 2. Canlı Kamera karesini analiz et
        webcam_resized = cv2.resize(frame, (600, 450))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(webcam_resized, cv2.COLOR_BGR2RGB))
        
        result = landmarker.detect(mp_image)
        
        user_pose = None
        user_landmarks = None
        pose_detected = False

        if result.pose_landmarks:
            user_landmarks = result.pose_landmarks[0]
            user_pose = extract_normalized_landmarks(user_landmarks)
            pose_detected = True

            # En iyi eşleşen futbolcuyu bul
            en_iyi_skor = -1
            en_iyi_isim = None
            
            for name, player_pose in player_poses.items():
                skor = compare_poses(user_pose, player_pose)
                if skor > en_iyi_skor:
                    en_iyi_skor = skor
                    en_iyi_isim = name
            
            # Eşleşme yüzdesi %30'un üzerindeyse yeni eşleşmeyi kilitle
            if en_iyi_isim and en_iyi_skor >= 30:
                gosterilen_player  = en_iyi_isim
                gosterilen_skor    = en_iyi_skor
                son_eslesme_zamani = time.time()

        # 3. Sol Panel (Webcam Görünümü) Üzerine İskelet Çizimi
        if pose_detected and user_landmarks:
            # İskelet bağlantı kemiklerini çiz (Neon turkuaz rengi)
            for start_idx, end_idx in CONNECTIONS:
                sl = user_landmarks[start_idx]
                el = user_landmarks[end_idx]
                
                # Sadece görünürlüğü yüksek noktaları birleştir
                if sl.visibility > 0.4 and el.visibility > 0.4:
                    pt1 = (int(sl.x * 600), int(sl.y * 450))
                    pt2 = (int(el.x * 600), int(el.y * 450))
                    cv2.line(webcam_resized, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Eklemleri çiz (Neon altın/sarı noktalar)
            for idx in POSE_INDICES:
                lm = user_landmarks[idx]
                if lm.visibility > 0.4:
                    pt = (int(lm.x * 600), int(lm.y * 450))
                    cv2.circle(webcam_resized, pt, 5, (0, 215, 255), -1, cv2.LINE_AA)
                    cv2.circle(webcam_resized, pt, 2, (255, 255, 255), -1, cv2.LINE_AA)
            
            # Algılanan pozun etrafına dinamik sınırlayıcı kutu çiz
            xs = [int(lm.x * 600) for lm in user_landmarks if lm.visibility > 0.4]
            ys = [int(lm.y * 450) for lm in user_landmarks if lm.visibility > 0.4]
            if xs and ys:
                min_x, max_x = max(0, min(xs) - 20), min(600, max(xs) + 20)
                min_y, max_y = max(0, min(ys) - 20), min(450, max(ys) + 20)
                cv2.rectangle(webcam_resized, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
                draw_text(webcam_resized, "ISKELET KILITLENDI", min_x, max(15, min_y - 5), 0.4, (0, 255, 0), 1)

        # Kamera karesini sol panel bölgesine koy
        canvas[120:570, 30:630] = webcam_resized
        draw_hud_card(canvas, (30, 120), (630, 570), "CANLI POZ ANALIZI (LIVE WEBCAM)", (0, 255, 0), active=pose_detected)

        # 4. Sağ Panel (Eşleşen Futbolcu) Çizimi
        gosterim_aktif = (gosterilen_player is not None) and (time.time() - son_eslesme_zamani < cooldown_zamani)
        
        if gosterim_aktif:
            # Eşleşen görseli sağ panele yerleştir
            canvas[130:480, 680:1060] = images[gosterilen_player]
            
            official_name = PLAYER_NAMES.get(gosterilen_player, gosterilen_player.replace(".png", "").upper())
            tw = cv2.getTextSize(official_name, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0]
            tx = 670 + (400 - tw) // 2
            draw_text(canvas, official_name, tx, 510, 0.65, (0, 215, 255), 2)
            
            # Benzerlik yüzdesi barı
            bar_x, bar_y = 720, 530
            bar_w, bar_h = 300, 16
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
            fill_w = int(bar_w * (gosterilen_skor / 100.0))
            if fill_w > 0:
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 215, 255), -1)
            
            skor_txt = f"ESLESME: %{gosterilen_skor}"
            stw = cv2.getTextSize(skor_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
            draw_text(canvas, skor_txt, bar_x + (bar_w - stw) // 2, bar_y + 12, 0.45, (255, 255, 255), 1)

            draw_hud_card(canvas, (670, 120), (1070, 570), "ESLESEN FUTBOLCU (MATCHED PLAYER)", (0, 215, 255), active=True)
        else:
            gosterilen_player = None
            canvas[130:480, 680:1060] = (25, 20, 15)
            
            # Radar/Tarama dairesi animasyonu
            pulse = int(10 + 6 * np.sin(time.time() * 6))
            center_x, center_y = 870, 305
            cv2.circle(canvas, (center_x, center_y), 65 + pulse, (80, 70, 60), 1)
            cv2.circle(canvas, (center_x, center_y), 45 - pulse // 2, (100, 90, 80), 1)
            cv2.circle(canvas, (center_x, center_y), 6, (0, 255, 0), -1)
            
            txt1 = "POZ VERIN..."
            txt2 = "Gol sevincini taklit etmeniz bekleniyor"
            tw1 = cv2.getTextSize(txt1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]
            tw2 = cv2.getTextSize(txt2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
            
            draw_text(canvas, txt1, center_x - tw1 // 2, 420, 0.55, (200, 200, 200), 2)
            draw_text(canvas, txt2, center_x - tw2 // 2, 445, 0.45, (130, 130, 130), 1)
            
            draw_hud_card(canvas, (670, 120), (1070, 570), "ESLESEN FUTBOLCU (MATCHED PLAYER)", (100, 100, 100), active=False)

        # 5. Başlık ve Alt Şeridi Çiz
        cv2.rectangle(canvas, (0, 0), (1100, 70), (15, 12, 10), -1)
        cv2.line(canvas, (0, 70), (1100, 70), (0, 255, 0), 1)
        draw_text(canvas, "FUTBOLCUN KIM? | WHO IS YOUR PLAYER?", 30, 45, 0.8, (0, 255, 0), 2)
        
        status_str = "SISTEM: AKTIF" if pose_detected else "SISTEM: BEKLEMEDE"
        status_color = (0, 255, 0) if pose_detected else (0, 165, 255)
        cv2.circle(canvas, (1055, 38), 6, status_color, -1)
        draw_text(canvas, status_str, 900 if pose_detected else 890, 43, 0.45, (200, 200, 200), 1)

        cv2.rectangle(canvas, (0, 600), (1100, 650), (15, 12, 10), -1)
        cv2.line(canvas, (0, 600), (1100, 600), (100, 100, 100), 1)
        draw_text(canvas, "[Q] CIKIS (EXIT) | Kamera onunde bir gol sevinci pozu taklit ederek eslesin.", 30, 630, 0.45, (170, 170, 170), 1)

        cv2.imshow(window_name, canvas)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("\nUygulama başarıyla kapatıldı.")