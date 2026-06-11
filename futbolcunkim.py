import cv2
import numpy as np
import glob
import os
import time

# ── 1. Yapılandırma ve Tanımlamalar ─────────────────────────────────────
IMAGE_FOLDER = "futbolcular"  # Klasör adı güncellendi (fotograflar -> futbolcular)

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

# ── 2. Yardımcı Fonksiyonlar ───────────────────────────────────────────
def resize_and_pad(img, size=(380, 350), pad_color=(30, 25, 20)):
    """Görüntüyü en-boy oranını koruyarak yeniden boyutlandırır ve siyah dolgu ekler."""
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

    # Dolguları hesapla (ortalayarak yerleştir)
    pad_h = sh - new_h
    pad_w = sw - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    return padded

def draw_text(img, text, x, y, font_scale=0.6, color=(255, 255, 255), thickness=1):
    """Gölgeli ve temiz metin çizer (okunabilirliği artırmak için)."""
    # Siyah gölge
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    # Ana metin
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def draw_hud_card(canvas, pt1, pt2, title, border_color=(100, 100, 100), active=False):
    """Panel etrafına şık, yarı şeffaf cam görünümlü kart çizer."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Arka planı hafifçe karart (cam efekti)
    sub_img = canvas[y1:y2, x1:x2]
    rect = np.zeros(sub_img.shape, dtype=np.uint8)
    rect[:] = (15, 10, 5) if active else (5, 5, 5)
    canvas[y1:y2, x1:x2] = cv2.addWeighted(sub_img, 0.85, rect, 0.15, 0)
    
    # Dış çerçeve
    color = border_color if active else (100, 100, 100)
    cv2.rectangle(canvas, pt1, pt2, color, 1 if not active else 2)
    
    # Kart başlık etiketi
    if title:
        tw = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        cv2.rectangle(canvas, (x1, y1 - 18), (x1 + tw + 20, y1), color, -1)
        draw_text(canvas, title, x1 + 10, y1 - 5, 0.4, (255, 255, 255), 1)

# ── 3. Fotoğraf Setini ve ORB Özelliklerini Yükle ───────────────────────
print("Fotoğraf seti yükleniyor...")
images = {}
photo_features = {}

# ORB Özellik çıkarıcı
orb = cv2.ORB_create(nfeatures=1000)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

found_files = glob.glob(os.path.join(IMAGE_FOLDER, "*.png"))
if not found_files:
    print(f"Hata: '{IMAGE_FOLDER}' klasöründe PNG bulunamadı. Lütfen klasör adını kontrol edin.")
    exit(1)

for path in found_files:
    name = os.path.basename(path)
    # Türkçe karakterli veya boşluklu yollarda Windows uyumluluğu için ikili dosya okuma
    try:
        with open(path, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is not None:
            # Görseli optimize et ve yükle
            img_preprocessed = resize_and_pad(img, (380, 350))
            images[name] = img_preprocessed
            
            gray = cv2.cvtColor(img_preprocessed, cv2.COLOR_BGR2GRAY)
            kp, des = orb.detectAndCompute(gray, None)
            photo_features[name] = (kp, des)
    except Exception as e:
        print(f"Hata: {name} dosyası yüklenirken sorun oluştu: {e}")

print(f"Başarıyla {len(images)} futbolcu yüklendi ve indekslendi.")

# ── 4. Benzerlik Eşleştirme Fonksiyonu ──────────────────────────────────
def en_benzer_foto(query_img):
    """Kırpılan hareket alanını futbolcu fotoğraflarıyla eşleştirip en benzer olanı döner."""
    if len(photo_features) == 0:
        return None, 0
        
    gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    kp_q, des_q = orb.detectAndCompute(gray, None)

    # Yeterince detay yoksa eşleştirmeyi atla
    if des_q is None or len(des_q) < 15:
        return None, 0

    en_iyi_skor = -1
    en_iyi_isim = None

    for name, (kp_p, des_p) in photo_features.items():
        if des_p is None:
            continue
            
        # ORB eşleştirmesi yap
        matches = matcher.match(des_q, des_p)
        
        # Hamming mesafesi < 65 olan kaliteli eşleşmeleri filtrele
        good_matches = [m for m in matches if m.distance < 65]
        skor = len(good_matches)

        if skor > en_iyi_skor:
            en_iyi_skor = skor
            en_iyi_isim = name

    # Güven yüzdesi (en az 45 kaliteli eşleşmeyi %100 olarak kabul et)
    percentage = min(100, int((en_iyi_skor / 45.0) * 100)) if en_iyi_skor > 5 else 0
    return en_iyi_isim, percentage

# ── 5. Kamera ve HUD Döngüsü Başlatılıyor ───────────────────────────────
cap = cv2.VideoCapture(0)

# Çözünürlük ayarlarını kontrol et ve oku
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ref_frame    = None
esik         = 20        # hareket hassasiyeti (biraz daha hassas yapıldı)
min_alan     = 4000      # küçük gürültüleri filtrele (px²)

# Durum Değişkenleri
gosterilen_player   = None
gosterilen_skor     = 0
son_eslesme_zamani  = 0.0
cooldown_zamani     = 2.5  # Eşleşmenin ekranda kalma süresi (sn)

# Animasyon Değişkenleri
scan_y = 0
scan_direction = 1
window_name = "Kim Bu Futbolcu? | Who Is Your Player"

cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
print("\nSistem Başlatıldı!")
print("→ Kamera önünde hareket edin veya futbolcu pozlarını taklit edin.")
print("→ Çıkmak için pencere seçiliyken 'Q' tuşuna basın.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Hata: Kameradan görüntü alınamıyor!")
        break

    # Aynalama efekti (kullanıcının kendini rahat görmesi için)
    frame = cv2.flip(frame, 1)

    # 1. Ana HUD Canvas'ı Oluştur (1100x650)
    canvas = np.zeros((650, 1100, 3), dtype=np.uint8)
    
    # Modern koyu mavi/gri arka plan gradyanı oluştur
    for x in range(1100):
        b = int(18 + (x / 1100) * 12)
        g = int(24 + (x / 1100) * 8)
        r = int(38 - (x / 1100) * 10)
        canvas[:, x] = (b, g, r)

    # 2. Kamera karesini griye dönüştür ve bulanıklaştır
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)

    if ref_frame is None:
        ref_frame = blur
        continue

    # Hareket Algılama (Arka Plan Farkı)
    diff    = cv2.absdiff(ref_frame, blur)
    _, mask = cv2.threshold(diff, esik, 255, cv2.THRESH_BINARY)
    mask    = cv2.dilate(mask, None, iterations=2)

    konturlar, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hareket_var = False
    en_buyuk_krop = None
    en_buyuk_alan = 0
    kutu_koordinatlari = None

    for k in konturlar:
        alan = cv2.contourArea(k)
        if alan < min_alan:
            continue

        hareket_var = True
        x, y, w, h = cv2.boundingRect(k)
        
        # En belirgin (en büyük) hareket alanını seç
        if alan > en_buyuk_alan:
            en_buyuk_alan = alan
            en_buyuk_krop = frame[y:y+h, x:x+w]
            kutu_koordinatlari = (x, y, w, h)

    # Eğer hareket varsa ve crop geçerliyse eşleştirme yap
    if hareket_var and en_buyuk_krop is not None and en_buyuk_krop.size > 0:
        isim, skor = en_benzer_foto(en_buyuk_krop)
        
        # Eşleşme kabul edilebilir bir orandaysa (%20 ve üzeri)
        if isim and skor >= 20:
            gosterilen_player  = isim
            gosterilen_skor    = skor
            son_eslesme_zamani = time.time()

    # Referans arka planı yavaşça güncelle
    ref_frame = cv2.addWeighted(ref_frame, 0.95, blur, 0.05, 0)

    # 3. Sol Panel (Webcam Görünümü) Çizimi
    # Kamera beslemesini panel boyutuna sığacak şekilde yeniden boyutlandır (600x450)
    webcam_resized = cv2.resize(frame, (600, 450))
    
    # Hareket algılandıysa yeşil tarama çizgisi ve kutu çiz
    if hareket_var:
        # Tarama çizgisi hareketi
        scan_y += scan_direction * 8
        if scan_y >= 450:
            scan_y = 450
            scan_direction = -1
        elif scan_y <= 0:
            scan_y = 0
            scan_direction = 1
            
        # İnce neon yeşil tarama çizgisi
        cv2.line(webcam_resized, (0, scan_y), (600, scan_y), (0, 255, 0), 2)
        # Tarama çizgisinin altına hafif bir parlama ekle
        overlay = webcam_resized.copy()
        if scan_direction == 1:
            cv2.rectangle(overlay, (0, max(0, scan_y-15)), (600, scan_y), (0, 150, 0), -1)
        else:
            cv2.rectangle(overlay, (0, scan_y), (600, min(450, scan_y+15)), (0, 150, 0), -1)
        cv2.addWeighted(overlay, 0.3, webcam_resized, 0.7, 0, webcam_resized)

        # Hareket kutusunu çiz (webcam_resized koordinatlarına göre ölçekle)
        if kutu_koordinatlari:
            kx, ky, kw, kh = kutu_koordinatlari
            # 640x480 -> 600x450 ölçeklemesi
            ox = int(kx * (600 / 640))
            oy = int(ky * (450 / 480))
            ow = int(kw * (600 / 640))
            oh = int(kh * (450 / 480))
            cv2.rectangle(webcam_resized, (ox, oy), (ox+ow, oy+oh), (0, 255, 255), 2)
            draw_text(webcam_resized, "HAREKET ALGILANDI", ox, max(15, oy - 5), 0.4, (0, 255, 255), 1)

    # Sol paneli tuvale yerleştir
    canvas[120:570, 30:630] = webcam_resized
    # Sol panel çerçevesini çiz
    draw_hud_card(canvas, (30, 120), (630, 570), "CANLI KAMERA (LIVE WEBCAM)", (0, 255, 0), active=hareket_var)

    # 4. Sağ Panel (Futbolcu Kartı) Çizimi
    # Eğer cooldown süresi bitmemişse eşleşen futbolcuyu göster
    gosterim_aktif = (gosterilen_player is not None) and (time.time() - son_eslesme_zamani < cooldown_zamani)
    
    if gosterim_aktif:
        # Futbolcu fotoğrafını sağ paneldeki yerine koy (680-1060, Y: 130-480)
        canvas[130:480, 680:1060] = images[gosterilen_player]
        
        # Futbolcu adını eşleştir ve yaz
        official_name = PLAYER_NAMES.get(gosterilen_player, gosterilen_player.replace(".png", "").upper())
        
        # İsmi ortala
        tw = cv2.getTextSize(official_name, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0]
        tx = 670 + (400 - tw) // 2
        draw_text(canvas, official_name, tx, 510, 0.65, (0, 215, 255), 2)
        
        # Eşleşme yüzdesi çubuğu
        bar_x, bar_y = 720, 530
        bar_w, bar_h = 300, 16
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
        fill_w = int(bar_w * (gosterilen_skor / 100.0))
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 215, 255), -1)
        # Skor metni
        skor_txt = f"BENZERLIK: %{gosterilen_skor}"
        stw = cv2.getTextSize(skor_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
        draw_text(canvas, skor_txt, bar_x + (bar_w - stw) // 2, bar_y + 12, 0.45, (255, 255, 255), 1)

        # Kart etrafına aktif altın sarısı çerçeve çiz
        draw_hud_card(canvas, (670, 120), (1070, 570), "ESLESEN FUTBOLCU (MATCHED PLAYER)", (0, 215, 255), active=True)
    else:
        # Eşleşme yokken veya süre dolmuşken bekleme ekranı animasyonu
        gosterilen_player = None # sıfırla
        
        # Boş panel alanı temizliği
        canvas[130:480, 680:1060] = (25, 20, 15)
        
        # Dönüş/Halka animasyonu (Hedef imleci görünümü)
        pulse = int(10 + 6 * np.sin(time.time() * 6))
        center_x, center_y = 870, 305
        cv2.circle(canvas, (center_x, center_y), 65 + pulse, (80, 70, 60), 1)
        cv2.circle(canvas, (center_x, center_y), 45 - pulse // 2, (100, 90, 80), 1)
        cv2.circle(canvas, (center_x, center_y), 6, (0, 255, 0), -1)
        
        # Bekleme Metinleri
        txt1 = "HAREKET BEKLENIYOR..."
        txt2 = "Kamera karsisinda hareket edin"
        tw1 = cv2.getTextSize(txt1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]
        tw2 = cv2.getTextSize(txt2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
        
        draw_text(canvas, txt1, center_x - tw1 // 2, 420, 0.55, (200, 200, 200), 2)
        draw_text(canvas, txt2, center_x - tw2 // 2, 445, 0.45, (130, 130, 130), 1)
        
        # Normal çerçeve çiz
        draw_hud_card(canvas, (670, 120), (1070, 570), "ESLESEN FUTBOLCU (MATCHED PLAYER)", (100, 100, 100), active=False)

    # 5. Başlık ve Alt Bilgi Panellerini Çiz
    # Üst Başlık Şeridi
    cv2.rectangle(canvas, (0, 0), (1100, 70), (15, 12, 10), -1)
    cv2.line(canvas, (0, 70), (1100, 70), (0, 255, 0), 1)
    draw_text(canvas, "FUTBOLCUN KIM? | WHO IS YOUR PLAYER?", 30, 45, 0.8, (0, 255, 0), 2)
    
    # Canlı Saat / FPS veya Durum Bilgisi sağ köşede
    status_str = "SISTEM: AKTIF"
    cv2.circle(canvas, (1055, 38), 6, (0, 255, 0), -1)
    draw_text(canvas, status_str, 940, 43, 0.45, (200, 200, 200), 1)

    # Alt Bilgi Şeridi
    cv2.rectangle(canvas, (0, 600), (1100, 650), (15, 12, 10), -1)
    cv2.line(canvas, (0, 600), (1100, 600), (100, 100, 100), 1)
    draw_text(canvas, "[Q] CIKIS (EXIT) | Kamera onunde hareket edip veya poz vererek futbolcularla eslesin.", 30, 630, 0.45, (170, 170, 170), 1)

    # Tuvali Ekranda Göster
    cv2.imshow(window_name, canvas)
    
    # Q tuşuna basılırsa döngüden çık
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Temizlik işlemleri
cap.release()
cv2.destroyAllWindows()
print("\nUygulama başarıyla kapatıldı.")