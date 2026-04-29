"""
Simple Driver Drowsiness Detection - Lightweight Version
Focuses on core detection with minimal system load
"""
import cv2
import time
import winsound
import numpy as np
from datetime import datetime
from collections import deque
import threading
import csv
import os
import smtplib
from email.message import EmailMessage

# MediaPipe for face detection
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("=" * 60)
print("  DRIVER DROWSINESS DETECTION SYSTEM")
print("  Lightweight Version - Eyes Closed Detection")
print("=" * 60)


class SimpleDrowsinessDetector:
    """Lightweight drowsiness detector using only eye closure"""
    
    def __init__(self):
        # Eye landmark indices (MediaPipe Face Mesh)
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # Detection settings - VERY SENSITIVE
        self.EAR_THRESHOLD = 0.20  # Below this = eyes closed
        self.ALERT_FRAMES = 15  # Frames before alert (~0.5 sec at 30fps)
        self.CRITICAL_FRAMES = 45  # Frames before critical (~1.5 sec)
        
        # State tracking
        self.closed_frames = 0
        self.total_frames = 0
        self.alert_active = False
        self.last_alert_time = 0
        self.alert_cooldown = 3  # seconds between alerts
        self.last_high_email_time = 0
        self.high_email_cooldown = 60  # seconds between HIGH email alerts
        self.critical_episode_active = False
        self.critical_start_time = None
        self.last_critical_email_time = 0
        self.critical_email_count = 0
        
        # Session data
        self.session_start = datetime.now()
        self.alerts = []
        self.ear_history = deque(maxlen=100)
        self.last_ear = 0.3
        self.detection_interval = max(1, int(os.getenv("DETECTION_INTERVAL_FRAMES", "2").strip()))
        self.process_scale = max(0.3, min(1.0, float(os.getenv("PROCESS_SCALE", "0.5").strip())))
        self.max_fps = max(5, int(os.getenv("MAX_FPS", "20").strip()))
        
        # CSV export
        self.csv_file = f"drowsiness_log_{self.session_start.strftime('%Y%m%d_%H%M%S')}.csv"
        self._init_csv()
        
        # Email settings
        self._load_env_file()
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com").strip()
        self.smtp_port = int(os.getenv("SMTP_PORT", "587").strip())
        self.sender_email = os.getenv("SENDER_EMAIL", "").strip()
        self.sender_password = os.getenv("SENDER_PASSWORD", "").strip()
        contacts = os.getenv("EMERGENCY_CONTACTS", "").strip()
        self.receiver_emails = [email.strip() for email in contacts.split(",") if email.strip()]
        self.critical_email_interval = int(
            os.getenv("CRITICAL_EMAIL_INTERVAL_SECONDS", "20").strip()
        )
        self.smtp_enabled = bool(
            self.sender_email and self.sender_password and self.receiver_emails
        )
        
        # Initialize MediaPipe
        self._init_face_mesh()
        
        print(f"\n[INIT] EAR Threshold: {self.EAR_THRESHOLD}")
        print(f"[INIT] Alert after {self.ALERT_FRAMES} frames closed")
        print(f"[INIT] Critical after {self.CRITICAL_FRAMES} frames closed")
        print(f"[INIT] CSV Log: {self.csv_file}")
        print(f"[INIT] Detection interval: every {self.detection_interval} frame(s)")
        print(f"[INIT] Processing scale: {self.process_scale:.2f}")
        print(f"[INIT] Max FPS cap: {self.max_fps}")
        if self.smtp_enabled:
            print(f"[INIT] Email alerts enabled for: {', '.join(self.receiver_emails)}")
            print(
                f"[INIT] CRITICAL spam email interval: {self.critical_email_interval} seconds"
            )
        else:
            print("[INIT] Email alerts disabled (missing SMTP env settings)")
    
    def _init_csv(self):
        """Initialize CSV file for logging"""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 'EAR', 'Eyes_Status', 'Drowsiness_%', 
                'Alert_Level', 'Closed_Frames', 'Alert_Triggered'
            ])
    
    def _init_face_mesh(self):
        """Initialize MediaPipe Face Mesh"""
        try:
            # Find model file
            model_paths = [
                "models/face_landmarker.task",
                "../models/face_landmarker.task",
                "face_landmarker.task"
            ]
            
            model_path = None
            for p in model_paths:
                if os.path.exists(p):
                    model_path = p
                    break
            
            if model_path:
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=1,
                    min_face_detection_confidence=0.3,
                    min_face_presence_confidence=0.3,
                    min_tracking_confidence=0.3
                )
                self.face_mesh = vision.FaceLandmarker.create_from_options(options)
                self.use_task_api = True
                print("[INIT] Using MediaPipe Task API")
            else:
                # Fallback to legacy API
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3
                )
                self.use_task_api = False
                print("[INIT] Using MediaPipe Legacy API")
                
        except Exception as e:
            print(f"[WARN] Task API failed ({e}), using legacy")
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
            self.use_task_api = False
    
    def _load_env_file(self, env_path=".env"):
        """Load key-value pairs from .env into environment if not already set."""
        if not os.path.exists(env_path):
            return
        
        with open(env_path, "r", encoding="utf-8-sig") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    
    def _calculate_ear(self, landmarks, eye_indices, img_w, img_h):
        """Calculate Eye Aspect Ratio"""
        try:
            pts = []
            for idx in eye_indices:
                if self.use_task_api:
                    lm = landmarks[idx]
                    pts.append((lm.x * img_w, lm.y * img_h))
                else:
                    lm = landmarks.landmark[idx]
                    pts.append((lm.x * img_w, lm.y * img_h))
            
            # Vertical distances
            v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
            v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
            
            # Horizontal distance
            h = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
            
            if h == 0:
                return 0.3
            
            ear = (v1 + v2) / (2.0 * h)
            return ear
            
        except Exception:
            return 0.3
    
    def _play_alert(self, level):
        """Play alert sound"""
        try:
            if level == "warning":
                winsound.Beep(800, 300)
            elif level == "high":
                winsound.Beep(1000, 500)
            elif level == "critical":
                for _ in range(3):
                    winsound.Beep(1500, 200)
                    time.sleep(0.1)
        except:
            pass
    
    def _log_to_csv(self, ear, eyes_status, drowsiness, level, alert_triggered):
        """Log data to CSV"""
        try:
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    f"{ear:.4f}",
                    eyes_status,
                    f"{drowsiness:.1f}",
                    level,
                    self.closed_frames,
                    alert_triggered
                ])
        except:
            pass
    
    def _send_high_alert_email(self, drowsiness, ear):
        """Send emergency email when HIGH alert is detected."""
        if not self.smtp_enabled:
            return False
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = EmailMessage()
        message["Subject"] = "HIGH Drowsiness Alert - Immediate Attention Needed"
        message["From"] = self.sender_email
        message["To"] = ", ".join(self.receiver_emails)
        message.set_content(
            "Driver Drowsiness Detection Alert\n\n"
            f"Alert Level: HIGH\n"
            f"Time: {timestamp}\n"
            f"Drowsiness: {drowsiness:.1f}%\n"
            f"Eyes Closed Frames: {self.closed_frames}\n"
            f"EAR: {ear:.4f}\n\n"
            "Please contact the driver immediately.\n"
        )
        
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(message)
        except (smtplib.SMTPException, OSError) as error:
            print(f"[EMAIL][ERROR] HIGH alert email failed: {error}")
            return False
        
        print(f"[EMAIL] HIGH alert email sent to: {', '.join(self.receiver_emails)}")
        return True

    def _send_critical_spam_email(self, level, drowsiness, ear, elapsed_seconds):
        """Send repeated email updates during an active critical episode."""
        if not self.smtp_enabled or self.critical_start_time is None:
            return False
        
        critical_start = datetime.fromtimestamp(self.critical_start_time).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = EmailMessage()
        message["Subject"] = f"CRITICAL Episode Ongoing - Update #{self.critical_email_count}"
        message["From"] = self.sender_email
        message["To"] = ", ".join(self.receiver_emails)
        message.set_content(
            "Driver Drowsiness Detection - CRITICAL Episode Ongoing\n\n"
            f"Current Level: {level}\n"
            f"Critical Started At: {critical_start}\n"
            f"Update Sent At: {now}\n"
            f"Duration Since Critical Start: {elapsed_seconds:.1f} seconds\n"
            f"Drowsiness: {drowsiness:.1f}%\n"
            f"Closed Frames: {self.closed_frames}\n"
            f"EAR: {ear:.4f}\n\n"
            "Driver is NOT SAFE yet. Please contact immediately.\n"
        )
        
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(message)
        except (smtplib.SMTPException, OSError) as error:
            print(f"[EMAIL][ERROR] CRITICAL update email failed: {error}")
            return False
        
        print(
            f"[EMAIL] CRITICAL update #{self.critical_email_count} sent "
            f"(duration {elapsed_seconds:.1f}s)"
        )
        return True

    def _send_critical_spam_email_snapshot(
        self, level, drowsiness, ear, elapsed_seconds, critical_start_time, email_count, closed_frames
    ):
        """Send repeated email updates using immutable snapshot values."""
        if not self.smtp_enabled:
            return False

        critical_start = datetime.fromtimestamp(critical_start_time).strftime("%Y-%m-%d %H:%M:%S")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = EmailMessage()
        message["Subject"] = f"CRITICAL Episode Ongoing - Update #{email_count}"
        message["From"] = self.sender_email
        message["To"] = ", ".join(self.receiver_emails)
        message.set_content(
            "Driver Drowsiness Detection - CRITICAL Episode Ongoing\n\n"
            f"Current Level: {level}\n"
            f"Critical Started At: {critical_start}\n"
            f"Update Sent At: {now}\n"
            f"Duration Since Critical Start: {elapsed_seconds:.1f} seconds\n"
            f"Drowsiness: {drowsiness:.1f}%\n"
            f"Closed Frames: {closed_frames}\n"
            f"EAR: {ear:.4f}\n\n"
            "Driver is NOT SAFE yet. Please contact immediately.\n"
        )

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(message)
        except (smtplib.SMTPException, OSError) as error:
            print(f"[EMAIL][ERROR] CRITICAL update email failed: {error}")
            return False

        print(f"[EMAIL] CRITICAL update #{email_count} sent (duration {elapsed_seconds:.1f}s)")
        return True

    def _send_safe_recovery_email(self, critical_duration_seconds):
        """Send one recovery email when driver returns to NORMAL after critical episode."""
        if not self.smtp_enabled or self.critical_start_time is None:
            return False
        
        critical_start = datetime.fromtimestamp(self.critical_start_time).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        safe_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = EmailMessage()
        message["Subject"] = "SAFE Recovery - Critical Drowsiness Episode Ended"
        message["From"] = self.sender_email
        message["To"] = ", ".join(self.receiver_emails)
        message.set_content(
            "Driver Drowsiness Detection - SAFE Recovery\n\n"
            f"Critical Started At: {critical_start}\n"
            f"Safe At: {safe_time}\n"
            f"Total Duration From Critical Start To Safe: {critical_duration_seconds:.1f} seconds\n"
            f"Critical Update Emails Sent: {self.critical_email_count}\n\n"
            "Driver status has returned to NORMAL.\n"
        )
        
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(message)
        except (smtplib.SMTPException, OSError) as error:
            print(f"[EMAIL][ERROR] SAFE recovery email failed: {error}")
            return False
        
        print(
            "[EMAIL] SAFE recovery email sent "
            f"(critical duration {critical_duration_seconds:.1f}s)"
        )
        return True

    def _send_safe_recovery_email_snapshot(self, critical_duration_seconds, critical_start_time, email_count):
        """Send one recovery email with snapshot values captured at transition."""
        if not self.smtp_enabled:
            return False

        critical_start = datetime.fromtimestamp(critical_start_time).strftime("%Y-%m-%d %H:%M:%S")
        safe_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = EmailMessage()
        message["Subject"] = "SAFE Recovery - Critical Drowsiness Episode Ended"
        message["From"] = self.sender_email
        message["To"] = ", ".join(self.receiver_emails)
        message.set_content(
            "Driver Drowsiness Detection - SAFE Recovery\n\n"
            f"Critical Started At: {critical_start}\n"
            f"Safe At: {safe_time}\n"
            f"Total Duration From Critical Start To Safe: {critical_duration_seconds:.1f} seconds\n"
            f"Critical Update Emails Sent: {email_count}\n\n"
            "Driver status has returned to NORMAL.\n"
        )

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(message)
        except (smtplib.SMTPException, OSError) as error:
            print(f"[EMAIL][ERROR] SAFE recovery email failed: {error}")
            return False

        print(
            "[EMAIL] SAFE recovery email sent "
            f"(critical duration {critical_duration_seconds:.1f}s)"
        )
        return True

    def _run_async(self, fn, *args):
        threading.Thread(target=fn, args=args, daemon=True).start()

    def _handle_critical_email_flow(self, level, drowsiness, ear, current_time):
        """Start CRITICAL episode, spam updates until NORMAL, then send one SAFE email."""
        if level == "CRITICAL" and not self.critical_episode_active:
            self.critical_episode_active = True
            self.critical_start_time = current_time
            self.last_critical_email_time = 0
            self.critical_email_count = 0
        
        if self.critical_episode_active and level != "NORMAL":
            if (current_time - self.last_critical_email_time) >= self.critical_email_interval:
                self.critical_email_count += 1
                elapsed = current_time - self.critical_start_time
                self.last_critical_email_time = current_time
                self._run_async(
                    self._send_critical_spam_email_snapshot,
                    level,
                    drowsiness,
                    ear,
                    elapsed,
                    self.critical_start_time,
                    self.critical_email_count,
                    self.closed_frames
                )
        
        if self.critical_episode_active and level == "NORMAL":
            critical_duration = current_time - self.critical_start_time
            critical_start_time = self.critical_start_time
            critical_email_count = self.critical_email_count
            self._run_async(
                self._send_safe_recovery_email_snapshot,
                critical_duration,
                critical_start_time,
                critical_email_count
            )
            self.critical_episode_active = False
            self.critical_start_time = None
            self.last_critical_email_time = 0
            self.critical_email_count = 0
    
    def process_frame(self, frame):
        """Process a single frame and detect drowsiness"""
        self.total_frames += 1
        h, w = frame.shape[:2]
        
        # Default state
        ear = 0.3
        eyes_status = "UNKNOWN"
        drowsiness = 0.0
        level = "NORMAL"
        alert_triggered = False
        ear = self.last_ear
        
        try:
            should_detect = (self.total_frames % self.detection_interval == 0) or self.total_frames == 1
            if should_detect:
                if self.process_scale < 1.0:
                    frame_for_detection = cv2.resize(
                        frame,
                        None,
                        fx=self.process_scale,
                        fy=self.process_scale,
                        interpolation=cv2.INTER_LINEAR
                    )
                else:
                    frame_for_detection = frame
                
                dh, dw = frame_for_detection.shape[:2]
                rgb = cv2.cvtColor(frame_for_detection, cv2.COLOR_BGR2RGB)
                
                if self.use_task_api:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    results = self.face_mesh.detect(mp_image)
                    
                    if results.face_landmarks and len(results.face_landmarks) > 0:
                        landmarks = results.face_landmarks[0]
                        left_ear = self._calculate_ear(landmarks, self.LEFT_EYE, dw, dh)
                        right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE, dw, dh)
                        ear = (left_ear + right_ear) / 2.0
                else:
                    results = self.face_mesh.process(rgb)
                    
                    if results.multi_face_landmarks:
                        landmarks = results.multi_face_landmarks[0]
                        left_ear = self._calculate_ear(landmarks, self.LEFT_EYE, dw, dh)
                        right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE, dw, dh)
                        ear = (left_ear + right_ear) / 2.0
             
            # Store EAR
            self.last_ear = ear
            self.ear_history.append(ear)
            
            # Determine eye status
            if ear < self.EAR_THRESHOLD:
                eyes_status = "CLOSED"
                self.closed_frames += 1
            else:
                eyes_status = "OPEN"
                self.closed_frames = max(0, self.closed_frames - 2)  # Decay slowly
            
            # Calculate drowsiness percentage (0-100)
            # Based on how long eyes have been closed
            if self.closed_frames >= self.CRITICAL_FRAMES:
                drowsiness = min(100, 80 + (self.closed_frames - self.CRITICAL_FRAMES))
                level = "CRITICAL"
            elif self.closed_frames >= self.ALERT_FRAMES:
                drowsiness = 50 + (self.closed_frames - self.ALERT_FRAMES) * 2
                level = "WARNING" if drowsiness < 70 else "HIGH"
            else:
                drowsiness = (self.closed_frames / self.ALERT_FRAMES) * 50
                level = "NORMAL"
            
            # Trigger alerts
            current_time = time.time()
            self._handle_critical_email_flow(level, drowsiness, ear, current_time)
            if level != "NORMAL" and (current_time - self.last_alert_time) > self.alert_cooldown:
                alert_triggered = True
                self.last_alert_time = current_time
                self.alerts.append({
                    'time': datetime.now(),
                    'level': level,
                    'drowsiness': drowsiness,
                    'ear': ear
                })
                
                # Play sound in background thread would be better, but for simplicity:
                if level == "CRITICAL":
                    self._run_async(self._play_alert, "critical")
                elif level == "HIGH":
                    self._run_async(self._play_alert, "high")
                    if (
                        not self.critical_episode_active
                        and (current_time - self.last_high_email_time) > self.high_email_cooldown
                    ):
                        self.last_high_email_time = current_time
                        self._run_async(self._send_high_alert_email, drowsiness, ear)
                else:
                    self._run_async(self._play_alert, "warning")
                
                print(f"\n🚨 ALERT: {level} - Drowsiness: {drowsiness:.1f}% - Eyes closed for {self.closed_frames} frames")
            
            # Log to CSV periodically
            if self.total_frames % 10 == 0 or alert_triggered:
                self._log_to_csv(ear, eyes_status, drowsiness, level, alert_triggered)
            
        except Exception as e:
            pass
        
        # Draw on frame
        self._draw_overlay(frame, ear, eyes_status, drowsiness, level)
        
        return frame, {
            'ear': ear,
            'eyes_status': eyes_status,
            'drowsiness': drowsiness,
            'level': level,
            'closed_frames': self.closed_frames,
            'alert_triggered': alert_triggered
        }
    
    def _draw_overlay(self, frame, ear, eyes_status, drowsiness, level):
        """Draw simple overlay on frame"""
        h, w = frame.shape[:2]
        
        # Color based on level
        colors = {
            "NORMAL": (0, 255, 0),    # Green
            "WARNING": (0, 255, 255),  # Yellow
            "HIGH": (0, 165, 255),     # Orange
            "CRITICAL": (0, 0, 255)    # Red
        }
        color = colors.get(level, (255, 255, 255))
        
        # Draw background box
        cv2.rectangle(frame, (10, 10), (350, 140), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 140), color, 2)
        
        # Draw text
        cv2.putText(frame, f"DROWSINESS: {drowsiness:.0f}%", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Status: {level}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Eyes: {eyes_status}", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Closed Frames: {self.closed_frames}", (20, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw progress bar for drowsiness
        bar_w = 320
        bar_h = 20
        bar_x = 15
        bar_y = h - 40
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * (drowsiness / 100))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
        
        # Large warning text when critical
        if level == "CRITICAL":
            cv2.putText(frame, "!!! WAKE UP !!!", (w//2 - 150, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        elif level == "HIGH":
            cv2.putText(frame, "! DROWSY !", (w//2 - 100, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 2)
    
    def get_session_stats(self):
        """Get session statistics"""
        duration = (datetime.now() - self.session_start).total_seconds()
        return {
            'duration_minutes': duration / 60,
            'total_frames': self.total_frames,
            'total_alerts': len(self.alerts),
            'avg_ear': np.mean(list(self.ear_history)) if self.ear_history else 0,
            'csv_file': self.csv_file
        }


def main():
    """Main detection loop"""
    print("\n[INFO] Initializing camera...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    detector = SimpleDrowsinessDetector()
    
    print("\n" + "=" * 60)
    print("  DETECTION STARTED")
    print("  Press 'Q' to quit")
    print("  Press 'S' to save session stats")
    print("=" * 60 + "\n")
    
    fps_time = time.time()
    frame_count = 0
    fps = 0
    
    try:
        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame, state = detector.process_frame(frame)
            
            # Calculate FPS
            frame_count += 1
            if time.time() - fps_time >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_time = time.time()
            
            # Show FPS
            cv2.putText(frame, f"FPS: {fps}", (frame.shape[1] - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display
            cv2.imshow("Drowsiness Detection - Press Q to Quit", frame)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
                stats = detector.get_session_stats()
                print(f"\n📊 SESSION STATS:")
                print(f"   Duration: {stats['duration_minutes']:.1f} minutes")
                print(f"   Total Frames: {stats['total_frames']}")
                print(f"   Total Alerts: {stats['total_alerts']}")
                print(f"   Average EAR: {stats['avg_ear']:.3f}")
                print(f"   CSV File: {stats['csv_file']}")

            frame_time = time.time() - loop_start
            target_frame_time = 1.0 / detector.max_fps
            if frame_time < target_frame_time:
                time.sleep(target_frame_time - frame_time)
    
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")
    
    finally:
        stats = detector.get_session_stats()
        print(f"\n" + "=" * 60)
        print("  SESSION ENDED")
        print(f"  Duration: {stats['duration_minutes']:.1f} minutes")
        print(f"  Total Alerts: {stats['total_alerts']}")
        print(f"  CSV saved to: {stats['csv_file']}")
        print("=" * 60)
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
