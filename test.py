1 import cv2
    2 from ultralytics import YOLO
    3 from picamera2 import Picamera2
    4 import time
    5 
    6 # Modelimizin yeni yolu (TFLite modeli)
    7 MODEL_PATH = 'best.tflite'
    8 
    9 # Modeli yüklüyoruz
   10 try:
   11     model = YOLO(MODEL_PATH)
   12 except Exception as e:
   13     print(f"Model yüklenirken bir hata oluştu: {e}")
   14     exit()
   15 
   16 # PiCamera2'yi başlat ve yapılandır
   17 picam2 = Picamera2()
   18 # Kamerayı daha düşük çözünürlükte yapılandırmak FPS'yi artırır
   19 config = picam2.create_preview_configuration(main={"size": (640, 480)})
   20 picam2.configure(config)
   21 picam2.start()
   22 
   23 print("Kamera başarıyla başlatıldı. Çıkmak için pencere seçiliyken 'q' tuşuna basın."
      )
   24 time.sleep(1) # Kameranın ısınması için kısa bir bekleme
   25 
   26 # FPS hesaplaması için değişkenler
   27 prev_frame_time = 0
   28 new_frame_time = 0
   29 
   30 while True:
   31     # Kameradan kareyi yakala (NumPy dizisi olarak)
   32     frame = picam2.capture_array()
   33 
   34     # Görüntüyü BGR'dan RGB'ye dönüştür (OpenCV BGR, diğerleri RGB kullanır)
   35     # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # ultralytics bunu kendi 
      hallediyor
   36 
   37     # Modeli kare üzerinde çalıştır
   38     results = model(frame, stream=True)
   39 
   40     # Sonuçları işle
   41     for r in results:
   42         # Tespit kutularını ve etiketleri çiz
   43         frame = r.plot()
   44 
   45     # FPS'yi hesapla ve ekrana yazdır
   46     new_frame_time = time.time()
   47     fps = 1 / (new_frame_time - prev_frame_time)
   48     prev_frame_time = new_frame_time
   49     cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,
      255, 0), 2)
   50 
   51     # İşlenmiş kareyi göster
   52     cv2.imshow("Raspberry Pi - Trafik Isareti Algilama", frame)
   53 
   54     # 'q' tuşuna basılırsa döngüden çık
   55     if cv2.waitKey(1) & 0xFF == ord('q'):
   56         break
   57 
   58 # Her şeyi serbest bırak
   59 cv2.destroyAllWindows()
   60 picam2.stop()
   61 print("Uygulama kapatıldı.")
