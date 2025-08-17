#!/usr/bin/env python3
"""
ADAS Şerit Takibi Uyarı Sistemi - Raspberry Pi 5
OpenCV ile gerçek zamanlı şerit algılama ve uyarı sistemi
"""

import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import threading
from dataclasses import dataclass
from typing import Tuple, List, Optional
import logging
from enum import Enum
import math

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/lane_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LanePosition(Enum):
    """Araç pozisyonu"""
    CENTER = "center"
    LEFT_WARNING = "left_warning"
    RIGHT_WARNING = "right_warning"
    LEFT_CRITICAL = "left_critical"
    RIGHT_CRITICAL = "right_critical"

@dataclass
class LaneConfig:
    """Şerit algılama konfigürasyonu"""
    # Kamera parametreleri
    camera_width: int = 640
    camera_height: int = 480
    fps: int = 30
    
    # ROI (Region of Interest) parametreleri
    roi_top_ratio: float = 0.6      # Ekranın üst %60'ı
    roi_bottom_ratio: float = 1.0   # Ekranın alt %100'ü
    
    # Şerit algılama parametreleri
    canny_low: int = 50
    canny_high: int = 150
    hough_threshold: int = 20
    hough_min_line_length: int = 20
    hough_max_line_gap: int = 300
    
    # Uyarı parametreleri
    warning_threshold: float = 0.15  # Şerit genişliğinin %15'i
    critical_threshold: float = 0.05 # Şerit genişliğinin %5'i

@dataclass
class AlertPins:
    """Uyarı sistemi GPIO pinleri"""
    left_led: int = 16
    right_led: int = 26
    warning_led: int = 13
    buzzer: int = 6
    vibrator: int = 12

class LaneDetectionSystem:
    """Şerit Takibi Uyarı Sistemi"""
    
    def __init__(self, use_pi_camera: bool = True):
        self.config = LaneConfig()
        self.alert_pins = AlertPins()
        self.use_pi_camera = use_pi_camera
        
        # Sistem durumu
        self.is_running = False
        self.current_position = LanePosition.CENTER
        self.lane_center = 0
        self.vehicle_center = 0
        self.lane_width = 0
        
        # Kamera
        self.camera = None
        self.frame = None
        self.processed_frame = None
        
        # Performans metrikleri
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        # GPIO başlatma
        self.setup_gpio()
        
        # Kamera başlatma
        self.setup_camera()

    def setup_gpio(self):
        """GPIO pinlerini yapılandır"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Uyarı pinlerini output olarak ayarla
            GPIO.setup(self.alert_pins.left_led, GPIO.OUT)
            GPIO.setup(self.alert_pins.right_led, GPIO.OUT)
            GPIO.setup(self.alert_pins.warning_led, GPIO.OUT)
            GPIO.setup(self.alert_pins.buzzer, GPIO.OUT)
            GPIO.setup(self.alert_pins.vibrator, GPIO.OUT)
            
            # Tüm uyarıları kapat
            self.clear_all_alerts()
            
            logger.info("GPIO pinleri başarıyla yapılandırıldı")
            
            # Başlangıç testi
            self.startup_test()
            
        except Exception as e:
            logger.error(f"GPIO yapılandırma hatası: {e}")
            raise

    def setup_camera(self):
        """Kamera sistemini başlat"""
        try:
            if self.use_pi_camera:
                # Raspberry Pi Camera
                from picamera2 import Picamera2
                self.camera = Picamera2()
                
                # Kamera konfigürasyonu
                config = self.camera.create_preview_configuration(
                    main={"format": "RGB888", "size": (self.config.camera_width, self.config.camera_height)},
                    controls={"FrameRate": self.config.fps}
                )
                self.camera.configure(config)
                self.camera.start()
                logger.info("Pi Camera başarıyla başlatıldı")
                
            else:
                # USB Kamera
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
                self.camera.set(cv2.CAP_PROP_FPS, self.config.fps)
                
                if not self.camera.isOpened():
                    raise Exception("USB kamera açılamadı")
                
                logger.info("USB Kamera başarıyla başlatıldı")
                
            # Kamera ısınması için bekle
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Kamera başlatma hatası: {e}")
            raise

    def startup_test(self):
        """Sistem başlangıç testi"""
        logger.info("Şerit takibi sistem testi başlatılıyor...")
        
        # LED test dizisi
        test_leds = [
            (self.alert_pins.left_led, "Sol LED"),
            (self.alert_pins.right_led, "Sağ LED"),
            (self.alert_pins.warning_led, "Uyarı LED")
        ]
        
        for led_pin, name in test_leds:
            GPIO.output(led_pin, True)
            time.sleep(0.3)
            GPIO.output(led_pin, False)
            time.sleep(0.1)
        
        # Buzzer test
        GPIO.output(self.alert_pins.buzzer, True)
        time.sleep(0.2)
        GPIO.output(self.alert_pins.buzzer, False)
        
        logger.info("Sistem testi tamamlandı")

    def clear_all_alerts(self):
        """Tüm uyarıları temizle"""
        GPIO.output(self.alert_pins.left_led, False)
        GPIO.output(self.alert_pins.right_led, False)
        GPIO.output(self.alert_pins.warning_led, False)
        GPIO.output(self.alert_pins.buzzer, False)
        GPIO.output(self.alert_pins.vibrator, False)

    def capture_frame(self) -> Optional[np.ndarray]:
        """Kameradan frame yakala"""
        try:
            if self.use_pi_camera:
                # Pi Camera
                frame = self.camera.capture_array()
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                # USB Camera
                ret, frame = self.camera.read()
                return frame if ret else None
                
        except Exception as e:
            logger.error(f"Frame yakalama hatası: {e}")
            return None

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Frame ön işleme"""
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Gaussian blur uygula
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # ROI (Region of Interest) uygula
        height, width = blurred.shape
        roi_top = int(height * self.config.roi_top_ratio)
        roi_bottom = int(height * self.config.roi_bottom_ratio)
        
        # ROI maskesi oluştur
        mask = np.zeros_like(blurred)
        roi_vertices = np.array([
            [(0, height), (0, roi_top), (width, roi_top), (width, height)]
        ], dtype=np.int32)
        
        cv2.fillPoly(mask, roi_vertices, 255)
        masked = cv2.bitwise_and(blurred, mask)
        
        return masked

    def detect_edges(self, preprocessed: np.ndarray) -> np.ndarray:
        """Kenar algılama"""
        return cv2.Canny(preprocessed, self.config.canny_low, self.config.canny_high)

    def detect_lines(self, edges: np.ndarray) -> List[np.ndarray]:
        """Hough transform ile çizgi algılama"""
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.config.hough_threshold,
            minLineLength=self.config.hough_min_line_length,
            maxLineGap=self.config.hough_max_line_gap
        )
        
        return lines if lines is not None else []

    def filter_lines(self, lines: List[np.ndarray], frame_shape: Tuple[int, int]) -> Tuple[List, List]:
        """Çizgileri sol ve sağ şerit olarak filtrele"""
        height, width = frame_shape
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Çizgi eğimini hesapla
            if x2 - x1 == 0:  # Dikey çizgiyi atla
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            
            # Eğim filtreleme (çok yatay çizgileri atla)
            if abs(slope) < 0.3:
                continue
            
            # Çizgi merkez noktası
            mid_x = (x1 + x2) / 2
            
            # Sol ve sağ şerit ayırımı
            if slope < 0 and mid_x < width * 0.5:  # Sol şerit (negatif eğim, sol yarı)
                left_lines.append(line[0])
            elif slope > 0 and mid_x > width * 0.5:  # Sağ şerit (pozitif eğim, sağ yarı)
                right_lines.append(line[0])
        
        return left_lines, right_lines

    def fit_lane_line(self, lines: List) -> Optional[Tuple[float, float]]:
        """Şerit çizgisi için en iyi doğruyu fit et"""
        if not lines:
            return None
        
        # Tüm noktaları topla
        x_coords = []
        y_coords = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        
        if len(x_coords) < 2:
            return None
        
        # Linear regression ile doğru fit et
        coefficients = np.polyfit(y_coords, x_coords, 1)
        return coefficients  # [slope, intercept]

    def calculate_lane_position(self, left_line: Optional[Tuple], right_line: Optional[Tuple], frame_shape: Tuple) -> LanePosition:
        """Araç pozisyonunu hesapla"""
        height, width = frame_shape
        
        # Frame alt orta noktası (araç pozisyonu)
        self.vehicle_center = width // 2
        
        # Şerit merkezini hesapla
        y_eval = height - 50  # Alt kısımdan değerlendir
        
        left_x = None
        right_x = None
        
        if left_line is not None:
            slope, intercept = left_line
            left_x = slope * y_eval + intercept
        
        if right_line is not None:
            slope, intercept = right_line
            right_x = slope * y_eval + intercept
        
        # Şerit merkezi ve genişliği
        if left_x is not None and right_x is not None:
            self.lane_center = (left_x + right_x) / 2
            self.lane_width = abs(right_x - left_x)
        elif left_x is not None:
            # Sadece sol şerit görünüyor, sağ şeridi tahmin et
            self.lane_width = 300  # Varsayılan şerit genişliği
            self.lane_center = left_x + self.lane_width / 2
        elif right_x is not None:
            # Sadece sağ şerit görünüyor, sol şeridi tahmin et
            self.lane_width = 300
            self.lane_center = right_x - self.lane_width / 2
        else:
            # Hiç şerit görünmüyor
            return LanePosition.CENTER
        
        # Araç pozisyonunu analiz et
        deviation = self.vehicle_center - self.lane_center
        deviation_ratio = abs(deviation) / (self.lane_width / 2) if self.lane_width > 0 else 0
        
        if deviation_ratio > self.config.critical_threshold:
            if deviation > 0:
                return LanePosition.RIGHT_CRITICAL
            else:
                return LanePosition.LEFT_CRITICAL
        elif deviation_ratio > self.config.warning_threshold:
            if deviation > 0:
                return LanePosition.RIGHT_WARNING
            else:
                return LanePosition.LEFT_WARNING
        else:
            return LanePosition.CENTER

    def handle_alerts(self, position: LanePosition):
        """Uyarı sistemini kontrol et"""
        current_time = time.time()
        
        # Tüm uyarıları temizle
        self.clear_all_alerts()
        
        if position == LanePosition.LEFT_WARNING:
            # Sol uyarı - yavaş yanıp sönme
            GPIO.output(self.alert_pins.left_led, int(current_time * 2) % 2)
            GPIO.output(self.alert_pins.warning_led, int(current_time * 2) % 2)
            
        elif position == LanePosition.RIGHT_WARNING:
            # Sağ uyarı - yavaş yanıp sönme
            GPIO.output(self.alert_pins.right_led, int(current_time * 2) % 2)
            GPIO.output(self.alert_pins.warning_led, int(current_time * 2) % 2)
            
        elif position == LanePosition.LEFT_CRITICAL:
            # Sol kritik - hızlı yanıp sönme + buzzer
            GPIO.output(self.alert_pins.left_led, int(current_time * 6) % 2)
            GPIO.output(self.alert_pins.warning_led, int(current_time * 6) % 2)
            GPIO.output(self.alert_pins.buzzer, int(current_time * 4) % 2)
            GPIO.output(self.alert_pins.vibrator, int(current_time * 4) % 2)
            
        elif position == LanePosition.RIGHT_CRITICAL:
            # Sağ kritik - hızlı yanıp sönme + buzzer
            GPIO.output(self.alert_pins.right_led, int(current_time * 6) % 2)
            GPIO.output(self.alert_pins.warning_led, int(current_time * 6) % 2)
            GPIO.output(self.alert_pins.buzzer, int(current_time * 4) % 2)
            GPIO.output(self.alert_pins.vibrator, int(current_time * 4) % 2)

    def draw_debug_info(self, frame: np.ndarray, left_lines: List, right_lines: List, 
                       left_line: Optional[Tuple], right_line: Optional[Tuple]) -> np.ndarray:
        """Debug bilgilerini frame üzerine çiz"""
        height, width = frame.shape[:2]
        
        # Algılanan çizgileri çiz
        for line in left_lines:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        for line in right_lines:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Fit edilen şerit çizgilerini çiz
        y_top = int(height * 0.6)
        y_bottom = height - 1
        
        if left_line is not None:
            slope, intercept = left_line
            x_top = int(slope * y_top + intercept)
            x_bottom = int(slope * y_bottom + intercept)
            cv2.line(frame, (x_top, y_top), (x_bottom, y_bottom), (255, 0, 0), 3)
        
        if right_line is not None:
            slope, intercept = right_line
            x_top = int(slope * y_top + intercept)
            x_bottom = int(slope * y_bottom + intercept)
            cv2.line(frame, (x_top, y_top), (x_bottom, y_bottom), (255, 0, 0), 3)
        
        # Araç merkezi ve şerit merkezi
        cv2.line(frame, (self.vehicle_center, y_bottom - 30), 
                (self.vehicle_center, y_bottom), (0, 0, 255), 3)  # Araç merkezi (kırmızı)
        
        if hasattr(self, 'lane_center'):
            cv2.line(frame, (int(self.lane_center), y_bottom - 30), 
                    (int(self.lane_center), y_bottom), (0, 255, 255), 3)  # Şerit merkezi (sarı)
        
        # Durum bilgisi
        status_text = f"Pozisyon: {self.current_position.value}"
        fps_text = f"FPS: {self.current_fps:.1f}"
        
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

    def update_fps(self):
        """FPS hesapla"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Ana frame işleme döngüsü"""
        # Ön işleme
        preprocessed = self.preprocess_frame(frame)
        
        # Kenar algılama
        edges = self.detect_edges(preprocessed)
        
        # Çizgi algılama
        lines = self.detect_lines(edges)
        
        # Çizgileri filtrele
        left_lines, right_lines = self.filter_lines(lines, frame.shape[:2])
        
        # Şerit çizgilerini fit et
        left_line = self.fit_lane_line(left_lines)
        right_line = self.fit_lane_line(right_lines)
        
        # Araç pozisyonunu hesapla
        self.current_position = self.calculate_lane_position(left_line, right_line, frame.shape[:2])
        
        # Uyarıları kontrol et
        self.handle_alerts(self.current_position)
        
        # Debug info çiz
        debug_frame = self.draw_debug_info(frame, left_lines, right_lines, left_line, right_line)
        
        return debug_frame

    def main_loop(self):
        """Ana işleme döngüsü"""
        logger.info("Şerit takibi ana döngüsü başlatılıyor...")
        
        while self.is_running:
            try:
                # Frame yakala
                frame = self.capture_frame()
                if frame is None:
                    continue
                
                # Frame işle
                self.processed_frame = self.process_frame(frame)
                
                # FPS güncelle
                self.update_fps()
                
                # Durum logla (sadece uyarı durumlarında)
                if self.current_position != LanePosition.CENTER:
                    logger.warning(f"Şerit uyarısı: {self.current_position.value}")
                
                time.sleep(0.01)  # CPU yükünü azalt
                
            except Exception as e:
                logger.error(f"Main loop hatası: {e}")
                time.sleep(0.1)

    def start_detection(self):
        """Şerit takibi sistemini başlat"""
        if self.is_running:
            logger.warning("Sistem zaten çalışıyor")
            return
        
        self.is_running = True
        logger.info("Şerit takibi sistemi başlatılıyor...")
        
        try:
            # Ana thread'i başlat
            main_thread = threading.Thread(target=self.main_loop, daemon=True)
            main_thread.start()
            
            logger.info("Şerit takibi sistemi başarıyla başlatıldı")
            
            # Sistem çalışır durumda bekle
            while self.is_running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Kullanıcı tarafından durduruldu")
        except Exception as e:
            logger.error(f"Sistem hatası: {e}")
        finally:
            self.stop_detection()

    def stop_detection(self):
        """Sistemi güvenli şekilde kapat"""
        logger.info("Sistem kapatılıyor...")
        self.is_running = False
        
        # Uyarıları temizle
        self.clear_all_alerts()
        
        # Kamerayı kapat
        try:
            if self.use_pi_camera:
                self.camera.stop()
            else:
                self.camera.release()
            logger.info("Kamera kapatıldı")
        except Exception as e:
            logger.error(f"Kamera kapatma hatası: {e}")
        
        # GPIO temizle
        try:
            GPIO.cleanup()
            logger.info("GPIO temizlendi")
        except Exception as e:
            logger.error(f"GPIO temizleme hatası: {e}")
        
        logger.info("Sistem güvenli şekilde kapatıldı")

    def run_camera_test(self):
        """Kamera testi"""
        logger.info("Kamera testi başlatılıyor...")
        
        test_duration = 10  # 10 saniye
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < test_duration:
            frame = self.capture_frame()
            if frame is not None:
                frame_count += 1
                
                # Frame kaydet (test için)
                if frame_count == 1:
                    cv2.imwrite('/tmp/test_frame.jpg', frame)
                    logger.info("Test frame kaydedildi: /tmp/test_frame.jpg")
            
            time.sleep(0.1)
        
        fps = frame_count / test_duration
        logger.info(f"Kamera testi tamamlandı. Ortalama FPS: {fps:.2f}")

def main():
    """Ana program"""
    print("=" * 60)
    print("ADAS Şerit Takibi Uyarı Sistemi v1.0")
    print("Raspberry Pi 5 + OpenCV + Camera")
    print("=" * 60)
    
    try:
        # Kamera tipini seç
        camera_choice = input("Kamera tipi (1: Pi Camera, 2: USB Camera): ").strip()
        use_pi_camera = (camera_choice == "1")
        
        detector = LaneDetectionSystem(use_pi_camera=use_pi_camera)
        
        while True:
            print("\n--- MENÜ ---")
            print("1. Şerit Takibini Başlat")
            print("2. Kamera Testi")
            print("3. Çıkış")
            
            choice = input("\nSeçiminiz (1-3): ").strip()
            
            if choice == '1':
                print("Şerit takibi başlatılıyor... (Durdurmak için Ctrl+C)")
                detector.start_detection()
                
            elif choice == '2':
                detector.run_camera_test()
                
            elif choice == '3':
                print("Sistem kapatılıyor...")
                break
                
            else:
                print("Geçersiz seçim!")
    
    except Exception as e:
        logger.error(f"Ana program hatası: {e}")
    
    finally:
        try:
            GPIO.cleanup()
            print("GPIO temizlendi. İyi günler!")
        except:
            pass

if __name__ == "__main__":
    main()