import os
import logging
import threading
import time
import numpy as np
import cv2
import datetime
from pathlib import Path

# Configs
RTSP_URL = "rtsp://192.168.1.100:554/profile1" # 720p
MOTION_THRESHOLD = 25      # Pixel intensity difference threshold
MOTION_AREA_THRESHOLD = 500  # Minimum number of changed pixels to trigger motion
CAPTURE_COOLDOWN = 2.0     # Seconds between captures to avoid flooding
WARMUP_FRAMES = 10
STALL_TIMEOUT = 10
STARTUP_TIMEOUT = 10

# Preserve original ffmpeg transport and flags via OpenCV environment variables
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;genpts+discardcorrupt|flags;low_delay"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class OpenCvRtspStream:
    """Decode RTSP via OpenCV VideoCapture with a background thread to avoid lag."""

    def __init__(self, src):
        self.src = src
        self.stopped = False
        self.ret = False
        self.frame = None
        self.frame_id = 0
        self.lock = threading.Lock()
        self._reconnect_lock = threading.Lock()

        self.cap = None
        self.thread = None
        self.started_at = 0.0
        self._start_pipeline()

    def start(self):
        return self

    def _start_pipeline(self):
        self.started_at = time.time()
        with self.lock:
            self.ret = False
            self.frame = None
            self.frame_id = 0

        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            logger.error("OpenCV could not open video stream: %s", self.src)

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                with self.lock:
                    self.ret = False
                    self.frame = None
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            
            if not ret:
                logger.error("OpenCvRtspStream: Failed to read frame.")
                with self.lock:
                    self.ret = False
                    self.frame = None
                time.sleep(0.1)
                continue

            with self.lock:
                first = self.frame_id == 0
                self.ret = True
                self.frame = frame
                self.frame_id += 1
            
            if first:
                w = frame.shape[1]
                h = frame.shape[0]
                logger.info("First frame received (%sx%s) (%.1fs after start).", w, h, time.time() - self.started_at)

    def read(self):
        with self.lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy(), self.frame_id
            return False, None, self.frame_id

    def _stop_pipeline(self):
        self.stopped = True
        if self.thread is not None:
            self.thread.join(timeout=3.0)
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.thread = None

    def reconnect(self):
        with self._reconnect_lock:
            logger.info("OpenCvRtspStream: Reconnecting to %s...", self.src)
            self._stop_pipeline()
            time.sleep(0.5)
            self.stopped = False
            self._start_pipeline()
            logger.info("OpenCvRtspStream: Reconnect complete.")
            return True

    def stop(self):
        with self._reconnect_lock:
            self._stop_pipeline()

def main():
    logger.info("Connecting to %s via OpenCV (RTSP/UDP)...", RTSP_URL)
    vs = OpenCvRtspStream(RTSP_URL).start()

    frame_count = 0
    last_frame_time = vs.started_at
    last_frame_id = 0
    
    # Capture settings
    CAPTURES_DIR = Path("capt_motion")
    CAPTURES_DIR.mkdir(exist_ok=True)
    
    last_capture_time = 0.0
    
    # Motion detection state
    baseline_frame = None
    while True:
        ret, frame, frame_id = vs.read()
        now = time.time()
        if frame_id <= WARMUP_FRAMES:
            continue
        if ret and frame is not None and frame_id != last_frame_id:
            last_frame_id = frame_id
            last_frame_time = now
            frame_count += 1
            
            # 1. Motion Detection
            # Convert to grayscale and blur to reduce noise
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if baseline_frame is None:
                baseline_frame = gray
                logger.info("Baseline frame established.")
                continue

            # Compute absolute difference between current frame and baseline
            frame_delta = cv2.absdiff(baseline_frame, gray)
            thresh = cv2.threshold(frame_delta, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
            
            # Dilate the thresholded image to fill in holes
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Count non-zero pixels to determine if motion occurred
            motion_pixels = cv2.countNonZero(thresh)
            
            if motion_pixels > MOTION_AREA_THRESHOLD:
                if now - last_capture_time >= CAPTURE_COOLDOWN:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = CAPTURES_DIR / f"motion_{timestamp}.jpg"
                    cv2.imwrite(str(filename), frame)
                    logger.info("Movement detected (%d pixels)! Image saved to %s", motion_pixels, filename)
                    last_capture_time = now
                else:
                    logger.debug("Movement detected, but cooldown active.")
            
            # Update baseline using a running average to adapt to lighting
            baseline_frame = cv2.addWeighted(baseline_frame, 0.95, gray, 0.05, 0)

            # Visual feedback
            cv2.imshow('Motion Detection', frame)
        elif frame_id == 0 and now - vs.started_at < STARTUP_TIMEOUT:
            pass
        elif now - last_frame_time > STALL_TIMEOUT:
            logger.warning(
                "Stream timeout after %s displayed frames (frame_id=%s). Reconnecting...",
                frame_count,
                frame_id,
            )
            if vs.reconnect():
                last_frame_id = 0
                last_frame_time = vs.started_at
            else:
                last_frame_time = now

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    logger.info("Stopping stream and destroying windows.")
    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
