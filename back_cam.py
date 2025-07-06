import cv2
import time

CAMERA_INDEX = 0 # BURAYI KONTROL ET! Doğru indeks mi?

# --- İstenen Ayarları ZORLA ---
# v4l2-ctl çıktısına göre GEREKİRSE değiştir.
DESIRED_WIDTH = 640
DESIRED_HEIGHT = 480
DESIRED_FPS = 30.0
# -----------------------------

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"HATA: Kamera indeksi {CAMERA_INDEX} açılamadı.")
    exit()

# Ayarları uygulamayı DENE (Kamera kabul etmeyebilir)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, DESIRED_FPS)

# Ayarların gerçekten ne olduğunu KONTROL ET
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Istenen: {DESIRED_WIDTH}x{DESIRED_HEIGHT} @ {DESIRED_FPS}fps")
print(f"Gerceklesen: {actual_width}x{actual_height} @ {actual_fps}fps") # Bu satır önemli!

# FPS Hesaplama için değişkenler
prev_frame_time = 0
new_frame_time = 0

print("Geri Görüş Kamerası başlatıldı. Çıkmak için 'q' tuşuna basın.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Hata: Kameradan görüntü alınamadı.")
        time.sleep(0.5) # Tekrar denemeden önce kısa bekleme
        # Kamera bağlantısı tamamen koptuysa yeniden bağlanmayı deneyebiliriz
        # cap.release()
        # cap = cv2.VideoCapture(CAMERA_INDEX)
        # ... (ama şimdilik basit tutalım)
        continue # Döngünün başına dön

    # Görüntüyü yatayda çevir
    mirrored_frame = cv2.flip(frame, 1)

    # FPS Hesaplama ve Gösterme
    new_frame_time = time.time()
    if (new_frame_time - prev_frame_time) > 0:
         fps = 1 / (new_frame_time - prev_frame_time)
    else:
         fps = 0
    prev_frame_time = new_frame_time
    fps_text = f"FPS: {fps:.1f}"

    cv2.putText(mirrored_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('Geri Gorus Kamerasi', mirrored_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Çıkış yapılıyor...")
        break

cap.release()
cv2.destroyAllWindows()
print("Geri Görüş Kamerası kapatıldı.")