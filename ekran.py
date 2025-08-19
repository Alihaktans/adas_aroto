import sys
import cv2
import time
import RPi.GPIO as GPIO
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout,
    QHBoxLayout, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage, QPainter, QFont # Import QPainter and QFont for placeholder

# Global camera index and desired settings for the camera feed
CAMERA_INDEX = 0
DESIRED_WIDTH = 640
DESIRED_HEIGHT = 480
DESIRED_FPS = 30.0

class MapWidget(QLabel):
    """
    A QLabel subclass specifically designed to display a map image.
    It scales the image while maintaining its aspect ratio.
    If the image path is invalid, it displays a placeholder.
    """
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.pixmap_original = QPixmap(image_path)
        
        # Check if the image loaded successfully. If not, create a dummy pixmap.
        if self.pixmap_original.isNull():
            print(f"Uyarı: '{image_path}' bulunamadı veya yüklenemedi. Yer tutucu kullanılacak.")
            # Create a simple placeholder pixmap if the image is not found
            # Adjust placeholder size to a reasonable default or based on common map ratios
            placeholder_width = 250
            placeholder_height = 400
            self.pixmap_original = QPixmap(placeholder_width, placeholder_height)
            self.pixmap_original.fill(Qt.lightGray) # Fill with a light gray color
            
            # Draw some text on the placeholder for clarity
            painter = QPainter(self.pixmap_original)
            painter.setFont(QFont('Arial', 20, QFont.Bold))
            painter.setPen(Qt.darkGray)
            painter.drawText(self.pixmap_original.rect(), Qt.AlignCenter, "Harita Yok\n(harita.png)")
            painter.end()

        self.setAlignment(Qt.AlignCenter)
        self.setContentsMargins(0, 0, 0, 0) # No margins for the image itself
        # Apply border-radius directly to the QLabel, assuming it's within a frame
        self.setStyleSheet("margin:0px; padding:0px; border-radius:10px;")

    def resizeEvent(self, event):
        """
        Overrides the resize event to scale the pixmap when the widget resizes.
        It keeps the aspect ratio by expanding to fill the space.
        """
        if not self.pixmap_original.isNull():
            scaled = self.pixmap_original.scaled(
                self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            self.setPixmap(scaled)
        super().resizeEvent(event)

class CameraThread(QThread):
    """
    A QThread subclass to handle OpenCV camera capture in a separate thread.
    This prevents the main GUI from freezing while video frames are processed.
    It emits QPixmap objects for new frames and also the current FPS.
    """
    # Signal emitted when a new frame is ready as a QPixmap
    change_pixmap_signal = pyqtSignal(QPixmap)
    # Signal emitted to update the FPS display text
    update_fps_signal = pyqtSignal(str)

    def __init__(self, camera_index=0, width=640, height=480, fps=30.0):
        super().__init__()
        self.camera_index = camera_index
        self.desired_width = width
        self.desired_height = height
        self.desired_fps = fps
        self._run_flag = True # Flag to control the thread's main loop

    def run(self):
        """
        The main loop for the camera thread.
        It continuously captures frames, processes them, and emits them.
        """
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            print(f"HATA: Kamera indeksi {self.camera_index} açılamadı. Lütfen doğru indeksi kontrol edin.")
            self._run_flag = False # Stop the thread if the camera cannot be opened
            return

        # Attempt to set camera properties.
        # Note: The camera hardware might not support all desired settings.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.desired_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.desired_height)
        cap.set(cv2.CAP_PROP_FPS, self.desired_fps)

        # Get the actual properties set by the camera
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Kamera Ayarları (İstenen): {self.desired_width}x{self.desired_height} @ {self.desired_fps}fps")
        print(f"Kamera Ayarları (Gerçekleşen): {actual_width}x{actual_height} @ {actual_fps}fps")

        prev_frame_time = 0 # Variable to calculate FPS

        while self._run_flag:
            ret, frame = cap.read() # Read a frame from the camera
            if ret:
                # Mirror the frame horizontally, typical for rearview cameras
                mirrored_frame = cv2.flip(frame, 1)

                # Convert OpenCV's BGR format to RGB for PyQt5
                rgb_image = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                
                # Create QImage from the numpy array
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                # Scale the QImage to fit the desired display size (optional, can be done by QLabel)
                p = convert_to_qt_format.scaled(
                    self.desired_width, self.desired_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.change_pixmap_signal.emit(QPixmap.fromImage(p)) # Emit the QPixmap

                # Calculate and emit FPS
                new_frame_time = time.time()
                if (new_frame_time - prev_frame_time) > 0:
                    fps = 1 / (new_frame_time - prev_frame_time)
                else:
                    fps = 0 # Avoid division by zero on the first frame
                prev_frame_time = new_frame_time
                self.update_fps_signal.emit(f"FPS: {fps:.1f}") # Emit the FPS string
            else:
                print("Hata: Kameradan görüntü alınamadı. Yeniden deneniyor...")
                time.sleep(0.5) # Wait a bit before retrying to prevent busy-looping

        cap.release() # Release the camera resources when the loop exits
        print("Kamera akışı durduruldu.")

    def stop(self):
        """
        Sets the internal flag to stop the thread's main loop.
        It also waits for the thread to finish its execution.
        """
        self._run_flag = False
        self.wait() # Block until the thread has finished execution

class MainDisplay(QWidget):
    """
    Main application window for the Efficiency Challenge interface.
    It integrates camera view, traffic signs, blind spots, and map display
    using a responsive layout.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Efficiency Challenge Arayüzü")
        # Set a minimum size to ensure content is visible, and allow for resizing
        self.setMinimumSize(1280, 400)
        self.resize(1280, 400) # Initial size for the window

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10) # Outer margins for the whole window
        main_layout.setSpacing(10) # Spacing between major sections
        self.setLayout(main_layout)

        # --- Left Panel: Traffic Signs + Blind Spots ---
        self.left_panel = QVBoxLayout()
        self.left_panel.setSpacing(10) # Spacing between traffic signs and blind spots

        # Traffic signs section
        self.traffic_signs_frame = QFrame()
        self.traffic_signs_frame.setFrameShape(QFrame.StyledPanel)
        self.traffic_signs_frame.setStyleSheet("""
            background-color: #FFC857;
            border-radius: 10px;
            padding: 5px; # Internal padding for content
        """)
        traffic_signs_layout_inner = QVBoxLayout(self.traffic_signs_frame)
        traffic_signs_layout_inner.setContentsMargins(10, 10, 10, 10)
        
        self.traffic_signs_title = QLabel("Trafik İşaretleri", alignment=Qt.AlignCenter)
        self.traffic_signs_title.setStyleSheet("color: white; font-size: 18px; font-weight: bold; margin-bottom: 5px;")
        traffic_signs_layout_inner.addWidget(self.traffic_signs_title)
        
        # Placeholder for actual traffic signs content (e.g., detected sign names, images)
        self.traffic_signs_content_label = QLabel("Tespit Yok", alignment=Qt.AlignCenter)
        self.traffic_signs_content_label.setStyleSheet("color: white; font-size: 14px; padding-bottom: 5px;")
        traffic_signs_layout_inner.addWidget(self.traffic_signs_content_label)
        
        self.left_panel.addWidget(self.traffic_signs_frame)
        self.left_panel.setStretchFactor(self.traffic_signs_frame, 1) # Allows it to take available vertical space

        # Blind spots section (horizontal layout for left/right)
        self.blindspot_layout = QHBoxLayout()
        self.blindspot_layout.setSpacing(10) # Spacing between left and right blind spot frames

        # Left Blind Spot
        self.blindspot_left_frame = QFrame()
        self.blindspot_left_frame.setFrameShape(QFrame.StyledPanel)
        self.blindspot_left_frame.setStyleSheet("""
            background-color: #E9724C;
            border-radius: 10px;
            padding: 5px;
        """)
        blindspot_left_layout_inner = QVBoxLayout(self.blindspot_left_frame)
        blindspot_left_layout_inner.setContentsMargins(10, 10, 10, 10)
        
        self.blindspot_left_title = QLabel("Kör Nokta (Sol)", alignment=Qt.AlignCenter)
        self.blindspot_left_title.setStyleSheet("color: white; font-size: 18px; font-weight: bold; margin-bottom: 5px;")
        blindspot_left_layout_inner.addWidget(self.blindspot_left_title)
        
        self.blindspot_left_status = QLabel("Güvenli", alignment=Qt.AlignCenter) # Status indicator (e.g., "Güvenli", "Tehlike")
        self.blindspot_left_status.setStyleSheet("color: white; font-size: 14px; padding-bottom: 5px;")
        blindspot_left_layout_inner.addWidget(self.blindspot_left_status)
        self.blindspot_layout.addWidget(self.blindspot_left_frame)

        # Right Blind Spot
        self.blindspot_right_frame = QFrame()
        self.blindspot_right_frame.setFrameShape(QFrame.StyledPanel)
        self.blindspot_right_frame.setStyleSheet("""
            background-color: #E9724C;
            border-radius: 10px;
            padding: 5px;
        """)
        blindspot_right_layout_inner = QVBoxLayout(self.blindspot_right_frame)
        blindspot_right_layout_inner.setContentsMargins(10, 10, 10, 10)
        
        self.blindspot_right_title = QLabel("Kör Nokta (Sağ)", alignment=Qt.AlignCenter)
        self.blindspot_right_title.setStyleSheet("color: white; font-size: 18px; font-weight: bold; margin-bottom: 5px;")
        blindspot_right_layout_inner.addWidget(self.blindspot_right_title)
        
        self.blindspot_right_status = QLabel("Güvenli", alignment=Qt.AlignCenter) # Status indicator
        self.blindspot_right_status.setStyleSheet("color: white; font-size: 14px; padding-bottom: 5px;")
        blindspot_right_layout_inner.addWidget(self.blindspot_right_status)
        self.blindspot_layout.addWidget(self.blindspot_right_frame)
        self.left_panel.addLayout(self.blindspot_layout)

        self.left_panel_widget = QWidget()
        self.left_panel_widget.setLayout(left_panel_layout)

        # --- Middle Panel: Camera View ---
        self.camera_view_frame = QFrame()
        self.camera_view_frame.setFrameShape(QFrame.StyledPanel)
        self.camera_view_frame.setStyleSheet("""
            background-color: #43BCCD;
            border-radius: 10px;
            padding: 5px;
        """)
        self.camera_view_layout = QVBoxLayout(self.camera_view_frame)
        self.camera_view_layout.setContentsMargins(5, 5, 5, 5) # Inner margins for the camera content

        # This QLabel will display the live camera feed
        self.camera_feed_label = QLabel("Kamera Görüntüsü Yükleniyor...", alignment=Qt.AlignCenter)
        self.camera_feed_label.setStyleSheet("color: white; font-size: 18px; font-weight: bold; background-color: black;") # Added black background for camera area
        self.camera_feed_label.setMinimumSize(DESIRED_WIDTH // 2, DESIRED_HEIGHT // 2) # Set a sensible minimum size
        self.camera_feed_label.setScaledContents(True) # Ensure the pixmap scales with the label
        self.camera_view_layout.addWidget(self.camera_feed_label)

        # FPS overlay label, positioned at the top-right
        self.fps_label = QLabel("FPS: N/A", alignment=Qt.AlignRight | Qt.AlignTop)
        self.fps_label.setStyleSheet("""
            color: yellow;
            font-size: 14px;
            background-color: rgba(0,0,0,0.5); /* Semi-transparent background for readability */
            padding: 2px 5px;
            border-radius: 5px;
            margin: 5px; /* Margin from the edges of its parent layout */
        """)
        # Add the FPS label to the layout, aligning it to the top right
        # Note: This positions it within the camera_view_layout. For true overlay,
        # a QStackedLayout or similar might be used, but this is simpler and works.
        self.camera_view_layout.addWidget(self.fps_label, alignment=Qt.AlignRight | Qt.AlignTop)
        # To make FPS label float on top of camera_feed_label, typically a QStackedLayout is used.
        # For this simple layout, it will occupy its own space, but stretch factors can minimize this.
        self.camera_view_layout.setStretchFactor(self.camera_feed_label, 1) # Allows camera feed to expand

        # --- Right Panel: Map ---
        self.right_panel = QVBoxLayout()
        self.right_panel.setSpacing(10) # Spacing between map and any other future elements

        self.map_view_frame = QFrame()
        self.map_view_frame.setFrameShape(QFrame.StyledPanel)
        self.map_view_frame.setStyleSheet("""
            background-color: #90BE6D;
            border-radius: 10px;
            padding: 0px; # MapWidget handles its own margins
        """)
        self.map_view_layout = QVBoxLayout(self.map_view_frame)
        self.map_view_layout.setContentsMargins(0, 0, 0, 0) # No internal margins for the map layout
        self.map_view_layout.setSpacing(0) # No spacing between map widgets if multiple

        self.map_label = MapWidget("harita.png")
        self.map_view_layout.addWidget(self.map_label)
        self.right_panel.addWidget(self.map_view_frame)

        self.right_panel_widget = QWidget()
        self.right_panel_widget.setLayout(right_panel_layout)

        # --- Main Layout Assembly ---
        main_layout.addLayout(self.left_panel_widget, stretch=1) # Left panel takes 1 unit of horizontal space
        main_layout.addWidget(self.camera_view_frame, stretch=2) # Camera view takes 2 units (more space)
        main_layout.addLayout(self.right_panel_widget, stretch=1) # Right panel takes 1 unit of horizontal space

        # Initialize and start the camera thread
        self.camera_thread = CameraThread(CAMERA_INDEX, DESIRED_WIDTH, DESIRED_HEIGHT, DESIRED_FPS)
        # Connect the thread's signal to the slot that updates the camera feed QLabel
        self.camera_thread.change_pixmap_signal.connect(self.update_camera_feed)
        # Connect the thread's signal to the slot that updates the FPS display
        self.camera_thread.update_fps_signal.connect(self.update_fps_display)
        self.camera_thread.start() # Start the camera capture thread

        #GPIO thread'i
        self.gpio_thread = GPIOThread(pin=REVERSE_GEAR_PIN)
        self.gpio_thread.reverse_gear_signal.connect(self.handle_reverse_gear)

    @pyqtSlot(QPixmap)
    def update_camera_feed(self, pixmap):
        """
        Slot to receive a QPixmap from the camera thread and display it
        on the camera_feed_label.
        """
        self.camera_feed_label.setPixmap(pixmap)

    @pyqtSlot(str)
    def update_fps_display(self, fps_text):
        """
        Slot to update the FPS display QLabel.
        """
        self.fps_label.setText(fps_text)

    #GPIO SİNYALLER FALAN FİŞMAN
    @pyqtSlot(bool)
    def handle_reverse_gear(self, is_reverse_active):
        self.left_panel_widget.setVisible(not is_reverse_active)
        self.right_panel_widget.setVisible(not is_reverse_active)

    def closeEvent(self, event):
        """
        Overrides the close event to ensure the camera thread is stopped gracefully
        when the application window is closed.
        """
        print("Uygulama kapatılıyor. Kamera akışı durduruluyor...")
        self.camera_thread.stop() # Send stop signal to the camera thread
        event.accept() # Accept the close event

if __name__ == "__main__":
    # The application entry point
    app = QApplication(sys.argv)
    try:
        window = MainDisplay()
        window.show()
        sys.exit(app.exec_()) # Start the PyQt5 event loop
    finally:
        GPIO.cleanup()
        print("GPIO temizlendi.")