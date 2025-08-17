import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import cv2
import json
import time
import threading
from datetime import datetime
import math
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import queue

# GPS verilerini simüle etmek için (gerçek uygulamada GPS modülünden gelir)
try:
    import serial
    GPS_AVAILABLE = True
except ImportError:
    GPS_AVAILABLE = False
    print("GPS modülü bulunamadı, simülasyon modu aktif")

@dataclass
class GPSCoordinate:
    """GPS koordinat sınıfı"""
    latitude: float
    longitude: float
    timestamp: datetime
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class MapPoint:
    """Harita üzerindeki nokta sınıfı"""
    x: float
    y: float
    
class GPSModule:
    """GPS modülü sınıfı - gerçek GPS modülü ile iletişim"""
    
    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 9600):
        self.port = port
        self.baudrate = baudrate
        self.is_connected = False
        self.serial_connection = None
        self.simulation_mode = not GPS_AVAILABLE
        
        # Simülasyon için test koordinatları
        self.sim_coordinates = [
            (40.7128, -74.0060),  # New York
            (40.7129, -74.0061),
            (40.7130, -74.0062),
            (40.7131, -74.0063),
            (40.7132, -74.0064),
        ]
        self.sim_index = 0
        
        if not self.simulation_mode:
            self.connect()
    
    def connect(self):
        """GPS modülüne bağlan"""
        try:
            if not self.simulation_mode:
                self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=1)
                self.is_connected = True
                print(f"GPS modülü bağlandı: {self.port}")
            else:
                self.is_connected = True
                print("Simülasyon modunda GPS aktif")
        except Exception as e:
            print(f"GPS bağlantı hatası: {e}")
            self.simulation_mode = True
            self.is_connected = True
    
    def read_gps_data(self) -> Optional[GPSCoordinate]:
        """GPS verilerini oku"""
        if not self.is_connected:
            return None
            
        if self.simulation_mode:
            # Simülasyon verisi
            lat, lon = self.sim_coordinates[self.sim_index]
            self.sim_index = (self.sim_index + 1) % len(self.sim_coordinates)
            return GPSCoordinate(lat, lon, datetime.now())
        else:
            # Gerçek GPS verisi okuma
            try:
                line = self.serial_connection.readline().decode('utf-8').strip()
                if line.startswith('$GPGGA'):
                    parts = line.split(',')
                    if len(parts) >= 6 and parts[2] and parts[4]:
                        lat = float(parts[2][:2]) + float(parts[2][2:]) / 60
                        lon = float(parts[4][:3]) + float(parts[4][3:]) / 60
                        
                        if parts[3] == 'S':
                            lat = -lat
                        if parts[5] == 'W':
                            lon = -lon
                            
                        return GPSCoordinate(lat, lon, datetime.now())
            except Exception as e:
                print(f"GPS okuma hatası: {e}")
                return None
        return None
    
    def disconnect(self):
        """GPS bağlantısını kapat"""
        if self.serial_connection:
            self.serial_connection.close()
        self.is_connected = False

class RaceTrack:
    """Yarış pisti sınıfı"""
    
    def __init__(self, track_name: str = "Test Pisti"):
        self.track_name = track_name
        self.reference_points = []  # GPS referans noktaları
        self.track_points = []      # Harita koordinatları
        self.track_bounds = None
        self.scale_factor = 1.0
        self.center_offset = (0, 0)
        
        # Varsayılan test pisti oluştur
        self.create_default_track()
    
    def create_default_track(self):
        """Varsayılan test pisti oluştur"""
        # GPS referans noktaları (gerçek koordinatlar)
        self.reference_points = [
            GPSCoordinate(40.7128, -74.0060, datetime.now()),
            GPSCoordinate(40.7130, -74.0060, datetime.now()),
            GPSCoordinate(40.7130, -74.0058, datetime.now()),
            GPSCoordinate(40.7128, -74.0058, datetime.now()),
        ]
        
        # Harita koordinatları (piksel/metre)
        self.track_points = [
            MapPoint(100, 100),
            MapPoint(500, 100),
            MapPoint(500, 300),
            MapPoint(100, 300),
        ]
        
        self.calculate_bounds()
        self.calculate_scale_factor()
    
    def calculate_bounds(self):
        """Harita sınırlarını hesapla"""
        if not self.track_points:
            return
            
        min_x = min(p.x for p in self.track_points)
        max_x = max(p.x for p in self.track_points)
        min_y = min(p.y for p in self.track_points)
        max_y = max(p.y for p in self.track_points)
        
        self.track_bounds = {
            'min_x': min_x, 'max_x': max_x,
            'min_y': min_y, 'max_y': max_y
        }
    
    def calculate_scale_factor(self):
        """Ölçek faktörünü hesapla"""
        if len(self.reference_points) < 2 or len(self.track_points) < 2:
            return
            
        # İlk iki nokta arasındaki mesafeyi kullan
        gps_dist = self.calculate_gps_distance(
            self.reference_points[0], self.reference_points[1]
        )
        map_dist = self.calculate_map_distance(
            self.track_points[0], self.track_points[1]
        )
        
        if map_dist > 0:
            self.scale_factor = gps_dist / map_dist
    
    def calculate_gps_distance(self, point1: GPSCoordinate, point2: GPSCoordinate) -> float:
        """İki GPS noktası arasındaki mesafeyi hesapla (metre)"""
        R = 6371000  # Dünya yarıçapı (metre)
        
        lat1, lon1 = math.radians(point1.latitude), math.radians(point1.longitude)
        lat2, lon2 = math.radians(point2.latitude), math.radians(point2.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def calculate_map_distance(self, point1: MapPoint, point2: MapPoint) -> float:
        """İki harita noktası arasındaki mesafeyi hesapla"""
        return math.sqrt((point2.x - point1.x)**2 + (point2.y - point1.y)**2)
    
    def gps_to_map_coordinates(self, gps_coord: GPSCoordinate) -> MapPoint:
        """GPS koordinatlarını harita koordinatlarına dönüştür"""
        if not self.reference_points or not self.track_points:
            return MapPoint(0, 0)
        
        # En yakın referans noktasını bul
        ref_gps = self.reference_points[0]
        ref_map = self.track_points[0]
        
        # GPS farkını hesapla
        lat_diff = gps_coord.latitude - ref_gps.latitude
        lon_diff = gps_coord.longitude - ref_gps.longitude
        
        # Harita koordinatlarına dönüştür
        # Bu basit bir dönüşüm - gerçek uygulamada daha karmaşık projeksiyon gerekebilir
        x = ref_map.x + (lon_diff * 111320 / self.scale_factor)  # 1 derece ≈ 111320 metre
        y = ref_map.y + (lat_diff * 111320 / self.scale_factor)
        
        return MapPoint(x, y)

class NavigationSystem:
    """Ana navigasyon sistemi"""
    
    def __init__(self):
        self.gps_module = GPSModule()
        self.race_track = RaceTrack()
        self.current_position = None
        self.position_history = []
        self.is_running = False
        self.update_interval = 0.5  # Saniye
        
        # Görselleştirme
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.vehicle_marker = None
        self.track_line = None
        self.history_line = None
        
        # Thread güvenliği için
        self.position_queue = queue.Queue()
        self.lock = threading.Lock()
        
        self.setup_visualization()
    
    def setup_visualization(self):
        """Görselleştirme ayarları"""
        self.ax.set_title('Yarış Navigasyonu Sistemi', fontsize=16, fontweight='bold')
        self.ax.set_xlabel('X Koordinatı (metre)', fontsize=12)
        self.ax.set_ylabel('Y Koordinatı (metre)', fontsize=12)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        # Pist çizgisini çiz
        self.draw_race_track()
        
        # Araç işaretçisini başlat
        self.vehicle_marker = plt.Circle((0, 0), 5, color='red', zorder=5)
        self.ax.add_patch(self.vehicle_marker)
        
        # Tarihçe çizgisi
        self.history_line, = self.ax.plot([], [], 'b-', alpha=0.6, linewidth=2, label='Geçmiş Rota')
        
        self.ax.legend()
    
    def draw_race_track(self):
        """Yarış pistini çiz"""
        if not self.race_track.track_points:
            return
        
        # Pist noktalarını al
        track_x = [p.x for p in self.race_track.track_points]
        track_y = [p.y for p in self.race_track.track_points]
        
        # Pisti kapalı bir döngü yap
        track_x.append(track_x[0])
        track_y.append(track_y[0])
        
        # Pist çizgisini çiz
        self.track_line, = self.ax.plot(track_x, track_y, 'g-', linewidth=3, 
                                       label='Yarış Pisti', zorder=3)
        
        # Pist sınırlarını ayarla
        margin = 50
        if self.race_track.track_bounds:
            bounds = self.race_track.track_bounds
            self.ax.set_xlim(bounds['min_x'] - margin, bounds['max_x'] + margin)
            self.ax.set_ylim(bounds['min_y'] - margin, bounds['max_y'] + margin)
    
    def start_navigation(self):
        """Navigasyonu başlat"""
        if not self.gps_module.is_connected:
            print("GPS modülü bağlı değil!")
            return
        
        self.is_running = True
        
        # GPS okuma thread'i başlat
        gps_thread = threading.Thread(target=self.gps_reading_loop)
        gps_thread.daemon = True
        gps_thread.start()
        
        # Animasyonu başlat
        self.animation = FuncAnimation(self.fig, self.update_visualization, 
                                     interval=int(self.update_interval * 1000),
                                     blit=False, cache_frame_data=False)
        
        plt.show()
    
    def gps_reading_loop(self):
        """GPS okuma döngüsü"""
        while self.is_running:
            try:
                gps_data = self.gps_module.read_gps_data()
                if gps_data:
                    # GPS koordinatlarını harita koordinatlarına dönüştür
                    map_position = self.race_track.gps_to_map_coordinates(gps_data)
                    
                    # Thread güvenli güncelleme
                    with self.lock:
                        self.position_queue.put((gps_data, map_position))
                
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"GPS okuma hatası: {e}")
                time.sleep(1)
    
    def update_visualization(self, frame):
        """Görselleştirmeyi güncelle"""
        # Yeni konumları al
        while not self.position_queue.empty():
            try:
                gps_pos, map_pos = self.position_queue.get_nowait()
                
                with self.lock:
                    self.current_position = map_pos
                    self.position_history.append(map_pos)
                    
                    # Tarihçeyi sınırla (son 100 nokta)
                    if len(self.position_history) > 100:
                        self.position_history.pop(0)
                
                # Konum bilgisini yazdır
                print(f"Konum: GPS({gps_pos.latitude:.6f}, {gps_pos.longitude:.6f}) -> "
                      f"Harita({map_pos.x:.1f}, {map_pos.y:.1f})")
                
            except queue.Empty:
                break
        
        # Araç konumunu güncelle
        if self.current_position:
            self.vehicle_marker.center = (self.current_position.x, self.current_position.y)
        
        # Tarihçe çizgisini güncelle
        if len(self.position_history) > 1:
            history_x = [p.x for p in self.position_history]
            history_y = [p.y for p in self.position_history]
            self.history_line.set_data(history_x, history_y)
        
        # Başlığı güncelle
        current_time = datetime.now().strftime("%H:%M:%S")
        self.ax.set_title(f'Yarış Navigasyonu Sistemi - {current_time}', 
                         fontsize=16, fontweight='bold')
        
        return [self.vehicle_marker, self.history_line]
    
    def calculate_track_deviation(self) -> float:
        """Pistten sapma mesafesini hesapla"""
        if not self.current_position or not self.race_track.track_points:
            return 0.0
        
        # En yakın pist noktasını bul
        min_distance = float('inf')
        
        for i in range(len(self.race_track.track_points)):
            track_point = self.race_track.track_points[i]
            distance = math.sqrt(
                (self.current_position.x - track_point.x)**2 + 
                (self.current_position.y - track_point.y)**2
            )
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def get_current_sector(self) -> int:
        """Araçın hangi sektörde olduğunu hesapla"""
        if not self.current_position or not self.race_track.track_points:
            return 0
        
        # En yakın pist noktasının indeksini bul
        min_distance = float('inf')
        closest_index = 0
        
        for i, track_point in enumerate(self.race_track.track_points):
            distance = math.sqrt(
                (self.current_position.x - track_point.x)**2 + 
                (self.current_position.y - track_point.y)**2
            )
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        # Sektör numarası (pist noktaları arası)
        return closest_index + 1
    
    def save_track_data(self, filename: str):
        """Pist verilerini kaydet"""
        track_data = {
            'track_name': self.race_track.track_name,
            'reference_points': [
                {
                    'latitude': p.latitude,
                    'longitude': p.longitude,
                    'timestamp': p.timestamp.isoformat()
                }
                for p in self.race_track.reference_points
            ],
            'track_points': [
                {'x': p.x, 'y': p.y} for p in self.race_track.track_points
            ],
            'scale_factor': self.race_track.scale_factor
        }
        
        with open(filename, 'w') as f:
            json.dump(track_data, f, indent=2)
        
        print(f"Pist verisi kaydedildi: {filename}")
    
    def load_track_data(self, filename: str):
        """Pist verilerini yükle"""
        try:
            with open(filename, 'r') as f:
                track_data = json.load(f)
            
            self.race_track.track_name = track_data['track_name']
            self.race_track.reference_points = [
                GPSCoordinate(
                    p['latitude'], p['longitude'], 
                    datetime.fromisoformat(p['timestamp'])
                )
                for p in track_data['reference_points']
            ]
            self.race_track.track_points = [
                MapPoint(p['x'], p['y']) for p in track_data['track_points']
            ]
            self.race_track.scale_factor = track_data['scale_factor']
            
            # Sınırları yeniden hesapla
            self.race_track.calculate_bounds()
            
            print(f"Pist verisi yüklendi: {filename}")
            
        except Exception as e:
            print(f"Pist verisi yükleme hatası: {e}")
    
    def stop_navigation(self):
        """Navigasyonu durdur"""
        self.is_running = False
        self.gps_module.disconnect()
        plt.close(self.fig)

# Ana program
def main():
    """Ana program fonksiyonu"""
    print("Yarış Navigasyonu Sistemi Başlatılıyor...")
    print("=" * 50)
    
    # Navigasyon sistemini başlat
    nav_system = NavigationSystem()
    
    try:
        # Test verisi kaydetme (isteğe bağlı)
        nav_system.save_track_data("test_track.json")
        
        # Navigasyonu başlat
        print("Navigasyon başlatılıyor...")
        print("Pencereyi kapatmak için X tuşuna basın.")
        
        nav_system.start_navigation()
        
    except KeyboardInterrupt:
        print("\nProgram kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"Hata: {e}")
    finally:
        nav_system.stop_navigation()
        print("Navigasyon sistemi kapatıldı.")

if __name__ == "__main__":
    main()