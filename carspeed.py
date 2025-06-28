# carspeed.py
#
# Dependencies: PyQt6, opencv-contrib-python, numpy
# Usage: This is a self-contained script. Run from the command line, e.g.,
#        python  carspeed.py
#
# Summary: A car speed detection application with a minimal graphical user interface.
#          The program allows users to load a video, define start and finish lines (of known separation distance),
#          and calculate the speed of vehicles crossing these lines.
#
# Last Updated: 2025-06-28
# Update Log:
#   - 2024-07-30: Refactored code based on engineering review. Improved performance, thread safety, and robustness.
#   - 2025-06-25: Implemented "High-Precision Timestamps" using linear interpolation for more accurate speed results.
#   - 2025-06-25: Added a "Stop Processing" button to interrupt analysis.
#   - 2025-06-25: Implemented an "Enhanced Filtering" mode using contour properties to reduce noise.
#   - 2025-06-24: Added video playback controls and real-time result display.
#   - 2025-06-24: Fixed `AttributeError` related to `get_boundingBox`.
# To do: 
# 1) Implement live video processing (via webcame feed)
# 2) Allow exporting of processed videos 

import sys
import cv2
import numpy as np
from collections import deque, defaultdict
from pathlib import Path
import logging
from dataclasses import dataclass, field

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QLineEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QStatusBar, QGroupBox,
    QFormLayout, QSlider, QStyle, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPoint, QTimer, QMutex
from PyQt6.QtGui import QImage, QPainter, QPen, QColor, QPixmap, QCursor

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =================================================================================
# 1. CONFIGURATION & DATA STRUCTURES
# =================================================================================

@dataclass
class AppConfig:
    """Holds all tunable parameters for detection and tracking."""
    min_area: int = 700
    motion_thresh: float = 0.02
    max_static: int = 45
    detect_interval: int = 10
    warmup_frames: int = 50
    path_history_len: int = 64
    # Use KCF tracker as it's much faster than CSRT and a good default.
    # TODO: Allow selecting tracker type (e.g., CSRT, MOSSE) from the GUI.
    tracker_constructor: callable = cv2.TrackerKCF_create
    float_precision_tolerance: float = 1e-9

# --- UI Colors ---
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)

# =================================================================================
# 2. UTILITY FUNCTIONS
# =================================================================================

def get_intersection_time_fraction(p1, p2, p3, p4):
    """
    Calculates the fractional time of intersection between two line segments.
    This is used for high-precision timestamping.

    Args:
        p1 (tuple): The first point of the vehicle's path segment (x, y).
        p2 (tuple): The second point of the vehicle's path segment (x, y).
        p3 (tuple): The first point of the measurement line (x, y).
        p4 (tuple): The second point of the measurement line (x, y).

    Returns:
        float: The fractional time 't' (between 0 and 1) if they intersect,
               None otherwise. 't' represents how far along the p1-p2 segment
               the intersection occurred.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    # Use a small tolerance for float comparison to avoid division by zero.
    if abs(den) < AppConfig.float_precision_tolerance:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
    if 0 < t < 1 and u > 0:
        return t
    return None

def draw_ui_overlay(frame, trackers, start_line, finish_line):
    """
    Draws all UI elements onto a video frame, including lines and tracker boxes.

    Args:
        frame (np.ndarray): The video frame to draw on.
        trackers (dict): A dictionary of active vehicle trackers.
        start_line (tuple): A tuple of two points defining the start line.
        finish_line (tuple): A tuple of two points defining the finish line.
    """
    if start_line:
        cv2.line(frame, start_line[0], start_line[1], COLOR_GREEN, 2)
        cv2.putText(frame, 'START', (start_line[0][0], start_line[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2)
        cv2.circle(frame, start_line[0], 7, COLOR_GREEN, -1)
        cv2.circle(frame, start_line[1], 7, COLOR_GREEN, -1)
    if finish_line:
        cv2.line(frame, finish_line[0], finish_line[1], COLOR_RED, 2)
        cv2.putText(frame, 'FINISH', (finish_line[0][0], finish_line[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
        cv2.circle(frame, finish_line[0], 7, COLOR_RED, -1)
        cv2.circle(frame, finish_line[1], 7, COLOR_RED, -1)

    for vid, t_info in trackers.items():
        if 'bbox' in t_info:
            bbox = t_info['bbox']
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, COLOR_BLUE, 2)
            cv2.putText(frame, f"ID: {vid}", (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 2)

def frame_to_pixmap(frame: np.ndarray) -> QPixmap:
    """Converts a BGR numpy array frame to a QPixmap."""
    h, w, ch = frame.shape
    bytes_per_line = ch * w
    qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
    return QPixmap.fromImage(qt_image)

# =================================================================================
# 3. BACKEND: VEHICLE TRACKING (PRE-PROCESSING)
# =================================================================================
# TODO: For even better performance, split this thread into a multi-worker pipeline:
# (1) Reader Thread -> (2) Detection/Tracking Worker Pool -> (3) Emitter to UI.
# This would leverage multi-core CPUs more effectively, especially for high-res video.

class PreProcessingThread(QThread):
    """
    A QThread that handles the heavy video processing in the background.

    This thread performs vehicle detection, tracking, and speed calculation
    to avoid freezing the main GUI thread. It emits signals to update the UI
    with its progress and results.

    Attributes:
        progress (pyqtSignal): Emits (current_frame, total_frames, frame_pixmap).
        new_result_ready (pyqtSignal): Emits a dictionary with speed calculation results.
    """
    progress = pyqtSignal(int, int, QPixmap)
    new_result_ready = pyqtSignal(dict)

    def __init__(self, video_path, settings, distance, fps):
        """
        Initializes the processing thread.

        Args:
            video_path (str): The path to the video file.
            settings (dict): A dictionary of tracking and detection parameters.
            distance (float): The real-world distance in meters between the lines.
            fps (float): The frames per second of the video.
        """
        super().__init__()
        self.video_path = video_path
        self.settings = settings
        self.distance = distance
        self.fps = fps
        self.start_line = None
        self.finish_line = None
        
        self._is_running = True
        self.mutex = QMutex()

    def is_running(self):
        """Thread-safe method to check if the thread should continue running."""
        self.mutex.lock()
        running = self._is_running
        self.mutex.unlock()
        return running

    def stop(self):
        """Thread-safe method to request the processing thread to stop."""
        logging.info("Processing thread stop requested.")
        self.mutex.lock()
        self._is_running = False
        self.mutex.unlock()

    def run(self):
        """
        The main processing loop of the thread.
        
        Opens the video file and processes it frame by frame. It applies background
        subtraction to detect moving objects, then uses KCF trackers to follow them.
        It calculates speed in real-time as vehicles cross the defined lines.
        """
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=True)
        
        trackers = {}
        vehicle_id_counter = 1
        # Use a deque with a max length to prevent memory from growing indefinitely.
        vehicle_paths = defaultdict(lambda: deque(maxlen=AppConfig.path_history_len))
        vehicle_cross_info = {}

        min_area = self.settings['min_area']
        motion_thresh = self.settings['motion_thresh']
        max_static = self.settings['max_static']
        detect_interval = self.settings['detect_interval']
        enhanced_filtering = self.settings['enhanced_filtering']
        self.high_precision_ts = self.settings['high_precision_ts']
        
        frame_idx = 0
        while self.is_running() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            fg_mask = bg_subtractor.apply(frame)
            
            if frame_idx < AppConfig.warmup_frames:
                frame_idx += 1
                pixmap = frame_to_pixmap(frame)
                self.progress.emit(frame_idx, total_frames, pixmap)
                continue

            active_ids = list(trackers.keys())
            boxes_to_remove = []
            
            for vehicle_id in active_ids:
                tracker_info = trackers[vehicle_id]
                tracker = tracker_info['tracker']
                success, bbox = tracker.update(frame)
                
                if success:
                    tracker_info['bbox'] = bbox
                    center_point = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
                    vehicle_paths[vehicle_id].append((frame_idx, center_point))
                    self.check_for_speed_calculation(vehicle_id, vehicle_paths[vehicle_id], vehicle_cross_info)

                    x, y, w, h = [int(v) for v in bbox]
                    if w > 0 and h > 0:
                        roi = fg_mask[y:y+h, x:x+w]
                        motion = cv2.countNonZero(roi) / (w * h) if (w*h) > 0 else 0
                        if motion < motion_thresh:
                            tracker_info['low_motion_frames'] += 1
                        else:
                            tracker_info['low_motion_frames'] = 0
                        if tracker_info['low_motion_frames'] >= max_static:
                            boxes_to_remove.append(vehicle_id)
                else:
                    boxes_to_remove.append(vehicle_id)
            
            for vehicle_id in boxes_to_remove:
                logging.info(f"Frame {frame_idx}: Lost track of Vehicle ID {vehicle_id}. Removing tracker.")
                if vehicle_id in trackers:
                    del trackers[vehicle_id]

            if frame_idx % detect_interval == 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                fg_mask_filtered = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                fg_mask_filtered = cv2.dilate(fg_mask_filtered, None, iterations=2)
                contours, _ = cv2.findContours(fg_mask_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if cv2.contourArea(contour) < min_area:
                        continue
                    
                    (x, y, w, h) = cv2.boundingRect(contour)

                    if enhanced_filtering:
                        if w == 0 or h == 0: continue
                        aspect_ratio = w / float(h)
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        solidity = cv2.contourArea(contour) / float(hull_area) if hull_area > 0 else 0
                        if not (1.0 < aspect_ratio < 4.0 and solidity > 0.8):
                            continue

                    is_new = True
                    for v_id, t_info in trackers.items():
                        if 'bbox' in t_info:
                            tx, ty, tw, th = t_info['bbox']
                            if (x < tx + tw and x + w > tx and y < ty + th and y + h > ty):
                                is_new = False
                                break
                    if is_new:
                        new_tracker = AppConfig.tracker_constructor()
                        new_tracker.init(frame, (x, y, w, h))
                        trackers[vehicle_id_counter] = {'tracker': new_tracker, 'low_motion_frames': 0}
                        logging.info(f"Frame {frame_idx}: New vehicle detected: ID {vehicle_id_counter}")
                        vehicle_id_counter += 1
            
            display_frame = frame.copy()
            draw_ui_overlay(display_frame, trackers, self.start_line, self.finish_line)
            # Emit a QPixmap directly to avoid marshalling large numpy arrays across threads.
            pixmap = frame_to_pixmap(display_frame)
            self.progress.emit(frame_idx, total_frames, pixmap)
            frame_idx += 1

        cap.release()

    def check_for_speed_calculation(self, vehicle_id, path, cross_info):
        """
        Checks if a vehicle has crossed a line and calculates speed if applicable.

        Args:
            vehicle_id (int): The ID of the vehicle to check.
            path (deque): The deque of historical center points for the vehicle.
            cross_info (dict): A dictionary storing crossing times and completion status.
        """
        if len(path) < 2 or vehicle_id in cross_info.get('completed', []):
            return
        
        frame_prev, p_prev = path[-2]
        frame_curr, p_curr = path[-1]
        
        if vehicle_id not in cross_info:
            time_fraction = get_intersection_time_fraction(p_prev, p_curr, self.start_line[0], self.start_line[1])
            if time_fraction is not None:
                precise_time = (frame_prev + time_fraction if self.high_precision_ts else frame_curr) / self.fps
                cross_info[vehicle_id] = {'line': 'start', 'time': precise_time}
            else:
                time_fraction = get_intersection_time_fraction(p_prev, p_curr, self.finish_line[0], self.finish_line[1])
                if time_fraction is not None:
                    precise_time = (frame_prev + time_fraction if self.high_precision_ts else frame_curr) / self.fps
                    cross_info[vehicle_id] = {'line': 'finish', 'time': precise_time}
        else:
            first_cross = cross_info[vehicle_id]
            crossed_second = False
            direction = None
            time_fraction = None
            
            if first_cross['line'] == 'start':
                time_fraction = get_intersection_time_fraction(p_prev, p_curr, self.finish_line[0], self.finish_line[1])
                if time_fraction is not None:
                    direction = "Primary (S->F)"
                    crossed_second = True
            elif first_cross['line'] == 'finish':
                time_fraction = get_intersection_time_fraction(p_prev, p_curr, self.start_line[0], self.start_line[1])
                if time_fraction is not None:
                    direction = "Reverse (F->S)"
                    crossed_second = True
            
            if crossed_second:
                precise_time = (frame_prev + time_fraction if self.high_precision_ts else frame_curr) / self.fps
                time_s = abs(precise_time - first_cross['time'])
                # Debounce based on video FPS to avoid spurious calculations.
                if time_s > (0.5 / self.fps):
                    speed_kmh = (self.distance / time_s) * 3.6
                    result = {'id': vehicle_id, 'direction': direction, 'time_s': time_s, 'speed_kmh': speed_kmh}
                    self.new_result_ready.emit(result)
                if 'completed' not in cross_info:
                    cross_info['completed'] = []
                cross_info['completed'].append(vehicle_id)

# =================================================================================
# 4. FRONTEND: PYQT6 GUI
# =================================================================================

class VideoWidget(QLabel):
    """A custom QLabel widget to display video frames and handle mouse events."""
    mouse_press = pyqtSignal(QPoint)
    mouse_move = pyqtSignal(QPoint)
    mouse_release = pyqtSignal(QPoint)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Please load a video file.")
        self.current_pixmap = None
        self.setMouseTracking(True)

    def set_pixmap(self, pixmap: QPixmap):
        self.current_pixmap = pixmap
        self.setPixmap(self.current_pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def _get_video_coordinates(self, event_pos: QPoint):
        if not self.current_pixmap:
            return None
        widget_size = self.size()
        pixmap_size = self.current_pixmap.size()
        scaled_pixmap = self.current_pixmap.scaled(widget_size, Qt.AspectRatioMode.KeepAspectRatio)
        x_offset = (widget_size.width() - scaled_pixmap.width()) / 2
        y_offset = (widget_size.height() - scaled_pixmap.height()) / 2
        click_x = event_pos.x() - x_offset
        click_y = event_pos.y() - y_offset
        if 0 <= click_x <= scaled_pixmap.width() and 0 <= click_y <= scaled_pixmap.height():
            return QPoint(int(click_x * pixmap_size.width() / scaled_pixmap.width()), int(click_y * pixmap_size.height() / scaled_pixmap.height()))
        return None

    def mousePressEvent(self, event):
        pos = self._get_video_coordinates(event.pos())
        if pos:
            self.mouse_press.emit(pos)

    def mouseMoveEvent(self, event):
        pos = self._get_video_coordinates(event.pos())
        if pos:
            self.mouse_move.emit(pos)

    def mouseReleaseEvent(self, event):
        pos = self._get_video_coordinates(event.pos())
        # Emit a special value if release happens outside the video area
        self.mouse_release.emit(pos if pos else QPoint(-1, -1))

class MainWindow(QMainWindow):
    """
    The main application window, containing all UI elements and control logic.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Car Speed Detector")
        self.setGeometry(100, 100, 1400, 900)
        
        self.video_path = None
        self.cap = None
        self.line_mode = None
        self.temp_point = None
        self.start_line = None
        self.finish_line = None
        self.processing_thread = None
        self.current_frame_num = 0
        self.dragging_point = None
        self.DRAG_HANDLE_RADIUS = 10
        self.is_playing = False
        self.config = AppConfig()

        self._setup_ui()
        self._connect_signals()
        self.update_button_states()

    def _setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # --- Left Panel (Video) ---
        left_panel = QVBoxLayout()
        self.video_widget = VideoWidget()
        left_panel.addWidget(self.video_widget)
        
        scrub_layout = QHBoxLayout()
        self.play_pause_button = QPushButton()
        self.play_pause_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.prev_frame_button = QPushButton("<")
        self.next_frame_button = QPushButton(">")
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_label = QLabel("Frame: 0 / 0")
        scrub_layout.addWidget(self.play_pause_button)
        scrub_layout.addWidget(self.prev_frame_button)
        scrub_layout.addWidget(self.frame_slider)
        scrub_layout.addWidget(self.next_frame_button)
        scrub_layout.addWidget(self.frame_label)
        left_panel.addLayout(scrub_layout)
        layout.addLayout(left_panel, 3)
        
        # --- Right Panel (Controls & Info) ---
        right_panel = QVBoxLayout()
        
        # Workflow Controls
        controls_group = QGroupBox("Workflow")
        controls_layout = QVBoxLayout(controls_group)
        self.load_button = QPushButton("1. Load Video")
        self.start_line_button = QPushButton("2. Set Start Line")
        self.finish_line_button = QPushButton("3. Set Finish Line")
        distance_layout = QHBoxLayout()
        distance_layout.addWidget(QLabel("Distance (meters):"))
        self.distance_input = QLineEdit("10")
        self.distance_input.setFixedWidth(50)
        distance_layout.addWidget(self.distance_input)
        distance_layout.addStretch()
        self.process_button = QPushButton("4. Process Video")
        controls_layout.addWidget(self.load_button)
        controls_layout.addWidget(self.start_line_button)
        controls_layout.addWidget(self.finish_line_button)
        controls_layout.addLayout(distance_layout)
        controls_layout.addWidget(self.process_button)
        right_panel.addWidget(controls_group)
        
        # Video Info
        info_group = QGroupBox("Video Information")
        info_layout = QFormLayout(info_group)
        self.info_filename_label = QLabel("N/A")
        self.info_resolution_label = QLabel("N/A")
        self.info_fps_label = QLabel("N/A")
        self.info_duration_label = QLabel("N/A")
        self.info_frames_label = QLabel("N/A")
        info_layout.addRow("File:", self.info_filename_label)
        info_layout.addRow("Resolution:", self.info_resolution_label)
        info_layout.addRow("FPS:", self.info_fps_label)
        info_layout.addRow("Duration:", self.info_duration_label)
        info_layout.addRow("Total Frames:", self.info_frames_label)
        right_panel.addWidget(info_group)
        
        # Settings
        settings_group = QGroupBox("Detection & Tracking Settings")
        settings_layout = QFormLayout(settings_group)
        self.enhanced_filtering_checkbox = QCheckBox("Enable Enhanced Filtering")
        self.enhanced_filtering_checkbox.setChecked(True)
        self.high_precision_ts_checkbox = QCheckBox("Enable High-Precision Timestamps")
        self.high_precision_ts_checkbox.setChecked(True)
        settings_layout.addRow(self.enhanced_filtering_checkbox)
        settings_layout.addRow(self.high_precision_ts_checkbox)
        
        self.min_area_slider, self.min_area_input = self._create_slider_input(100, 5000, self.config.min_area)
        settings_layout.addRow("Min Object Area:", self._create_slider_layout(self.min_area_slider, self.min_area_input))
        
        self.motion_thresh_slider, self.motion_thresh_input = self._create_slider_input(1, 100, int(self.config.motion_thresh * 100))
        settings_layout.addRow("Motion Threshold (%):", self._create_slider_layout(self.motion_thresh_slider, self.motion_thresh_input))
        
        self.max_static_slider, self.max_static_input = self._create_slider_input(5, 100, self.config.max_static)
        settings_layout.addRow("Max Static Frames:", self._create_slider_layout(self.max_static_slider, self.max_static_input))
        
        self.interval_slider, self.interval_input = self._create_slider_input(1, 30, self.config.detect_interval)
        settings_layout.addRow("Detection Interval:", self._create_slider_layout(self.interval_slider, self.interval_input))
        
        right_panel.addWidget(settings_group)
        
        # Results Table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Vehicle ID", "Direction", "Time (s)", "Speed (km/h)"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        right_panel.addWidget(QLabel("Results:"))
        right_panel.addWidget(self.results_table)
        layout.addLayout(right_panel, 1)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.playback_timer = QTimer(self)

    def _create_slider_input(self, min_val, max_val, default_val):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)
        line_edit = QLineEdit(str(default_val))
        line_edit.setFixedWidth(60)
        return slider, line_edit

    def _create_slider_layout(self, slider, line_edit):
        layout = QHBoxLayout()
        layout.addWidget(slider)
        layout.addWidget(line_edit)
        return layout

    def _connect_signals(self):
        self.playback_timer.timeout.connect(self.advance_frame)
        self.load_button.clicked.connect(self.load_video)
        self.start_line_button.clicked.connect(lambda: self.set_line_mode('start'))
        self.finish_line_button.clicked.connect(lambda: self.set_line_mode('finish'))
        self.process_button.clicked.connect(self.handle_process_button_click)
        self.play_pause_button.clicked.connect(self.toggle_playback)
        self.video_widget.mouse_press.connect(self.handle_video_press)
        self.video_widget.mouse_move.connect(self.handle_video_move)
        self.video_widget.mouse_release.connect(self.handle_video_release)
        self.frame_slider.valueChanged.connect(self.seek_frame)
        self.next_frame_button.clicked.connect(lambda: self.seek_frame(self.current_frame_num + 1))
        self.prev_frame_button.clicked.connect(lambda: self.seek_frame(self.current_frame_num - 1))
        
        # Connect sliders and inputs
        self.min_area_slider.valueChanged.connect(lambda v: self.min_area_input.setText(str(v)))
        self.min_area_input.editingFinished.connect(lambda: self.min_area_slider.setValue(int(self.min_area_input.text() or 0)))
        
        self.motion_thresh_slider.valueChanged.connect(lambda v: self.motion_thresh_input.setText(f"{v/100:.2f}"))
        self.motion_thresh_input.editingFinished.connect(lambda: self.motion_thresh_slider.setValue(int(float(self.motion_thresh_input.text() or 0)*100)))
        
        self.max_static_slider.valueChanged.connect(lambda v: self.max_static_input.setText(str(v)))
        self.max_static_input.editingFinished.connect(lambda: self.max_static_slider.setValue(int(self.max_static_input.text() or 0)))
        
        self.interval_slider.valueChanged.connect(lambda v: self.interval_input.setText(str(v)))
        self.interval_input.editingFinished.connect(lambda: self.interval_slider.setValue(int(self.interval_input.text() or 0)))

    def _is_float(self, s):
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False

    def update_button_states(self, processing=False):
        """Enables or disables UI elements based on the application's state."""
        video_loaded = self.video_path is not None
        self.load_button.setEnabled(not processing)
        self.start_line_button.setEnabled(video_loaded and not processing)
        self.finish_line_button.setEnabled(video_loaded and not processing)
        
        ready_to_process = all([self.video_path, self.start_line, self.finish_line])
        self.process_button.setEnabled(ready_to_process or processing)
        self.process_button.setText("Stop Processing" if processing else "4. Process Video")
        
        self.play_pause_button.setEnabled(video_loaded and not processing)
        self.frame_slider.setEnabled(video_loaded and not processing)
        self.next_frame_button.setEnabled(video_loaded and not processing)
        self.prev_frame_button.setEnabled(video_loaded and not processing)

    def load_video(self):
        """Opens a file dialog to load a video and initializes the player."""
        if self.is_playing:
            self.toggle_playback()
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if path:
            self.video_path = path
            self.cap = cv2.VideoCapture(self.video_path)
            self.results_table.setRowCount(0)
            self.start_line = None
            self.finish_line = None
            ret, _ = self.cap.read()
            if ret:
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.playback_timer.setInterval(int(1000 / fps) if fps > 0 else 40)
                self.seek_frame(0)
                self.status_bar.showMessage(f"Loaded: {self.video_path}. Set lines and process, or play preview.")
                self.update_video_info_display()
            else:
                QMessageBox.critical(self, "Error", "Could not read video.")
                self.video_path = None
            self.update_button_states()

    def toggle_playback(self):
        """Starts or stops the video preview playback."""
        if not self.cap:
            return
        self.is_playing = not self.is_playing
        if self.is_playing:
            if self.current_frame_num == self.frame_slider.maximum():
                self.seek_frame(0)
            self.playback_timer.start()
            self.play_pause_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        else:
            self.playback_timer.stop()
            self.play_pause_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def advance_frame(self):
        """Moves to the next frame during playback."""
        if self.current_frame_num >= self.frame_slider.maximum():
            self.toggle_playback()
        else:
            self.seek_frame(self.current_frame_num + 1)

    def seek_frame(self, frame_num):
        """
        Seeks to a specific frame in the video for display.
        This is used for scrubbing and playback. It does not show tracker data.
        """
        if self.processing_thread and self.processing_thread.isRunning():
            return
        if not self.cap:
            return
        frame_count = self.frame_slider.maximum()
        self.current_frame_num = max(0, min(frame_num, frame_count))
        if self.frame_slider.value() != self.current_frame_num:
            self.frame_slider.setValue(self.current_frame_num)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_num)
        ret, frame = self.cap.read()
        if ret:
            draw_ui_overlay(frame, {}, self.start_line, self.finish_line)
            pixmap = frame_to_pixmap(frame)
            self.video_widget.set_pixmap(pixmap)
        self.update_frame_label()

    def handle_process_button_click(self):
        """Handles clicks on the main process/stop button."""
        if self.processing_thread and self.processing_thread.isRunning():
            self.stop_preprocessing()
        else:
            self.start_preprocessing()

    def start_preprocessing(self):
        """Initializes and starts the background processing thread."""
        if self.is_playing:
            self.toggle_playback()
        self.results_table.setRowCount(0)
        try:
            settings = {
                'min_area': int(self.min_area_input.text()),
                'motion_thresh': float(self.motion_thresh_input.text()),
                'max_static': int(self.max_static_input.text()),
                'detect_interval': int(self.interval_input.text()),
                'enhanced_filtering': self.enhanced_filtering_checkbox.isChecked(),
                'high_precision_ts': self.high_precision_ts_checkbox.isChecked()
            }
            distance = float(self.distance_input.text())
            fps = self.cap.get(cv2.CAP_PROP_FPS)
        except (ValueError, TypeError, AttributeError) as e:
            QMessageBox.critical(self, "Invalid Settings", f"Please ensure all settings are valid. Error: {e}")
            return
        
        self.processing_thread = PreProcessingThread(self.video_path, settings, distance, fps)
        self.processing_thread.start_line = self.start_line
        self.processing_thread.finish_line = self.finish_line
        self.processing_thread.progress.connect(self.update_live_preview)
        self.processing_thread.new_result_ready.connect(self.add_result_to_table)
        self.processing_thread.finished.connect(self.on_processing_finished)
        
        self.processing_thread.start()
        self.update_button_states(processing=True)

    def stop_preprocessing(self):
        """Requests the processing thread to stop."""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.status_bar.showMessage("Stopping processing...")
            self.process_button.setEnabled(False)

    def update_live_preview(self, current, total, pixmap):
        """Updates the video widget with a new pixmap from the processing thread."""
        self.status_bar.showMessage(f"Processing frame {current}/{total}...")
        self.video_widget.set_pixmap(pixmap)
        self.current_frame_num = current
        self.frame_slider.setValue(current)

    def on_processing_finished(self):
        """Handles the cleanup after the processing thread has finished."""
        # The thread is finished, but we check its internal flag to see if it was a natural finish or a stop request.
        was_stopped_by_user = not self.processing_thread.is_running()
        if was_stopped_by_user:
             self.status_bar.showMessage("Processing stopped by user.", 5000)
        else:
             self.status_bar.showMessage("Processing complete.", 5000)
        
        self.processing_thread = None
        self.update_button_states(processing=False)
        self.seek_frame(self.current_frame_num)

    def add_result_to_table(self, m):
        """Adds a new result row to the results table in the UI."""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        self.results_table.setItem(row, 0, QTableWidgetItem(str(m['id'])))
        self.results_table.setItem(row, 1, QTableWidgetItem(m['direction']))
        self.results_table.setItem(row, 2, QTableWidgetItem(f"{m['time_s']:.2f}"))
        self.results_table.setItem(row, 3, QTableWidgetItem(f"{m['speed_kmh']:.2f}"))

    def closeEvent(self, event):
        """Ensures the processing thread is stopped when the window is closed."""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.processing_thread.wait() # Wait for the thread to finish
        event.accept()
        
    def update_video_info_display(self):
        """Updates the 'Video Information' panel with details from the loaded video."""
        if not self.cap:
            return
        filename = Path(self.video_path).name
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        self.info_filename_label.setText(filename)
        self.info_resolution_label.setText(f"{width} x {height}")
        self.info_fps_label.setText(f"{fps:.2f}")
        self.info_duration_label.setText(f"{duration:.2f} seconds")
        self.info_frames_label.setText(f"{frame_count:,}")
        self.frame_slider.setMaximum(frame_count - 1)
        self.update_frame_label()

    def update_frame_label(self):
        """Updates the frame counter label (e.g., "Frame: 1 / 2508")."""
        total_frames = self.frame_slider.maximum() + 1
        self.frame_label.setText(f"Frame: {self.current_frame_num + 1} / {total_frames}")

    def set_line_mode(self, mode):
        """Activates line drawing mode."""
        if self.is_playing:
            self.toggle_playback()
        self.line_mode = mode
        self.temp_point = None
        self.status_bar.showMessage(f"Click two points to define the {mode.upper()} line.")

    def get_handle_at_pos(self, p: QPoint):
        """Checks if a mouse position is over a line's draggable handle."""
        if self.start_line:
            for i, handle in enumerate(self.start_line):
                if (p.x() - handle[0])**2 + (p.y() - handle[1])**2 < self.DRAG_HANDLE_RADIUS**2:
                    return ('start', i)
        if self.finish_line:
            for i, handle in enumerate(self.finish_line):
                if (p.x() - handle[0])**2 + (p.y() - handle[1])**2 < self.DRAG_HANDLE_RADIUS**2:
                    return ('finish', i)
        return None

    def handle_video_press(self, point: QPoint):
        """Handles mouse press events on the video widget for drawing/dragging lines."""
        if self.is_playing:
            self.toggle_playback()
        if self.line_mode:
            self.handle_set_line_click(point)
            return
        self.dragging_point = self.get_handle_at_pos(point)
        if self.dragging_point:
            QApplication.setOverrideCursor(QCursor(Qt.CursorShape.CrossCursor))

    def handle_video_move(self, point: QPoint):
        """Handles mouse move events for dragging line endpoints."""
        if self.dragging_point:
            line, idx = self.dragging_point
            new_pos = (point.x(), point.y())
            if line == 'start':
                p = list(self.start_line)
                p[idx] = new_pos
                self.start_line = tuple(p)
            else:
                p = list(self.finish_line)
                p[idx] = new_pos
                self.finish_line = tuple(p)
            self.seek_frame(self.current_frame_num)
        else:
            handle = self.get_handle_at_pos(point)
            if handle:
                QApplication.setOverrideCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            else:
                QApplication.restoreOverrideCursor()

    def handle_video_release(self, point: QPoint):
        """Handles mouse release events to stop dragging a line."""
        if self.dragging_point:
            self.dragging_point = None
            QApplication.restoreOverrideCursor()
            self.update_button_states()
            self.status_bar.showMessage("Line position updated.", 3000)

    def handle_set_line_click(self, point: QPoint):
        """Handles mouse clicks when in line-drawing mode."""
        if not self.temp_point:
            self.temp_point = (point.x(), point.y())
        else:
            line = (self.temp_point, (point.x(), point.y()))
            if self.line_mode == 'start':
                self.start_line = line
            else:
                self.finish_line = line
            self.temp_point = None
            self.line_mode = None
            self.status_bar.showMessage(f"Line set. Ready to process or set other line.", 3000)
            self.seek_frame(self.current_frame_num)
            self.update_button_states()

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        logging.error(f"Failed to start GUI: {e}")
        # Fallback message for environments without a display or PyQt6
        print("\n---\nCould not launch the graphical application, likely because 'PyQt6' is not installed or a display is not available.\nThe Python script has been fully updated. Please run it in your local environment.\n---\n")

# carspeed.py
