import os
import logging
import threading
import time
import numpy as np
import cv2
import datetime
from pathlib import Path
from ultralytics import YOLO

# Configs
#RTSP_URL = "rtsp://192.168.68.72:554/live/ch00_0"
RTSP_URL = "rtsp://192.168.1.100:554/profile1" #720p
#RTSP_URL = "rtsp://192.168.1.100:554/profile2" #576p
CAT_DETECTION_THRESHOLD = 0.1
STARTUP_TIMEOUT = 20.0
STALL_TIMEOUT = 5.0

# Preserve original ffmpeg transport and flags via OpenCV environment variables
# Original cmd: -rtsp_transport udp, -fflags genpts+discardcorrupt, -flags low_delay
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;genpts+discardcorrupt|flags;low_delay"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("camera_test.log"),
        logging.StreamHandler()
    ]
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
    # Load YOLOv8n model (Nano is fastest for real-time RTSP)
    logger.info("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt') 

    logger.info("Connecting to %s via OpenCV (RTSP/UDP)...", RTSP_URL)

    vs = OpenCvRtspStream(RTSP_URL).start()

    frame_count = 0
    last_frame_time = vs.started_at
    last_frame_id = 0
    # Capture settings
    CAPTURES_DIR = Path("captures")
    CAPTURES_DIR.mkdir(exist_ok=True)
    cat_detected_start = None
    image_saved = False

    while True:
        ret, frame, frame_id = vs.read()
        now = time.time()

        if ret and frame is not None and frame_id != last_frame_id:
            last_frame_id = frame_id
            last_frame_time = now
            frame_count += 1
            if frame_count % 100 == 0:
                logger.debug("Successfully processed %s frames", frame_count)
            
            # Run YOLO inference filtering only for class 15 (cat)
            results = model.predict(frame, classes=[15], verbose=False)[0]
            
            # Handle continuous detection logic
            if len(results.boxes) > 0:
                if cat_detected_start is None:
                    cat_detected_start = time.time()
                
                # If cat detected for >= 1 second and image not yet saved for this event
                if not image_saved and (time.time() - cat_detected_start >= CAT_DETECTION_THRESHOLD):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = CAPTURES_DIR / f"cat_{timestamp}.jpg"
                    cv2.imwrite(str(filename), frame)
                    logger.info("Cat detected for 1s! Image saved to %s", filename)
                    image_saved = True
            else:
                # Reset tracking when no cat is present
                cat_detected_start = None
                image_saved = False
            
            # Plot bounding boxes on the frame
            annotated_frame = results.plot()
            cv2.imshow('RTSP Stream', annotated_frame)
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
