RTSP_URL = "rtsp://192.168.68.63:554/live/ch00_0"
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("camera_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

import subprocess
import threading
import time

import numpy as np

import cv2

STARTUP_TIMEOUT = 20.0  # RTSP + first keyframe can take several seconds
STALL_TIMEOUT = 5.0


def probe_stream_size(src, timeout=30.0):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-rtsp_transport", "tcp",
        "-probesize", "5000000",
        "-analyzeduration", "5000000",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        src,
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed ({result.returncode}): {result.stderr.strip()}"
        )
    size = result.stdout.strip()
    width_str, height_str = size.split("x")
    return int(width_str), int(height_str)


class FfmpegRtspStream:
    """Decode RTSP via system ffmpeg (same stack as ffplay), not OpenCV VideoCapture."""

    def __init__(self, src):
        self.src = src
        self.stopped = False
        self.ret = False
        self.frame = None
        self.frame_id = 0
        self.lock = threading.Lock()
        self._reconnect_lock = threading.Lock()

        self.width, self.height = probe_stream_size(src)
        self.frame_bytes = self.width * self.height * 3
        logger.info("Stream %sx%s @ %s", self.width, self.height, src)

        self.proc = None
        self.thread = None
        self._stderr_thread = None
        self.started_at = 0.0
        self._start_pipeline()

    def start(self):
        return self

    def _ffmpeg_cmd(self):
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-rtsp_transport", "udp",
            "-fflags", "genpts+discardcorrupt",
            "-flags", "low_delay",
            "-i", self.src,
            "-an", "-sn",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "pipe:1",
        ]

    def _drain_stderr(self):
        assert self.proc is not None and self.proc.stderr is not None
        for line in self.proc.stderr:
            if self.stopped:
                break
            text = line.decode(errors="replace").rstrip()
            if text:
                logger.debug("ffmpeg: %s", text)

    def _start_pipeline(self):
        self.started_at = time.time()
        with self.lock:
            self.ret = False
            self.frame = None
            self.frame_id = 0

        self.proc = subprocess.Popen(
            self._ffmpeg_cmd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self.frame_bytes * 2,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_thread.start()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        assert self.proc is not None and self.proc.stdout is not None
        stdout = self.proc.stdout
        while not self.stopped:
            if self.proc.poll() is not None:
                logger.error(
                    "FfmpegRtspStream: ffmpeg exited with code %s.",
                    self.proc.returncode,
                )
                with self.lock:
                    self.ret = False
                    self.frame = None
                time.sleep(0.1)
                continue

            raw = stdout.read(self.frame_bytes)
            if len(raw) != self.frame_bytes:
                if self.stopped:
                    break
                logger.error(
                    "FfmpegRtspStream: short read (%s/%s bytes). Stream misaligned. Stopping pipeline.",
                    len(raw),
                    self.frame_bytes,
                )
                with self.lock:
                    self.ret = False
                    self.frame = None
                self.stopped = True
                self.proc.kill()
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.height, self.width, 3)
            )
            with self.lock:
                first = self.frame_id == 0
                self.ret = True
                self.frame = frame
                self.frame_id += 1
            if first:
                logger.info(
                    "First frame received (%.1fs after ffmpeg start).",
                    time.time() - self.started_at,
                )

    def read(self):
        with self.lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy(), self.frame_id
            return False, None, self.frame_id

    def _stop_pipeline(self):
        self.stopped = True
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                pass
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self.proc = None
        self.thread = None
        self._stderr_thread = None

    def reconnect(self):
        with self._reconnect_lock:
            logger.info("FfmpegRtspStream: Reconnecting to %s...", self.src)
            self._stop_pipeline()

            time.sleep(0.5)
            self.stopped = False
            self._start_pipeline()
            logger.info("FfmpegRtspStream: Reconnect complete.")
            return True

    def stop(self):
        with self._reconnect_lock:
            self._stop_pipeline()


def main():
    logger.info("Connecting to %s via ffmpeg pipe (RTSP/TCP)...", RTSP_URL)

    try:
        vs = FfmpegRtspStream(RTSP_URL).start()
    except RuntimeError as exc:
        logger.error("Could not open video stream: %s", exc)
        return

    frame_count = 0
    last_frame_time = vs.started_at
    last_frame_id = 0

    while True:
        ret, frame, frame_id = vs.read()
        now = time.time()

        if ret and frame is not None and frame_id != last_frame_id:
            last_frame_id = frame_id
            last_frame_time = now
            frame_count += 1
            if frame_count % 100 == 0:
                logger.debug("Successfully processed %s frames", frame_count)
            cv2.imshow('RTSP Stream', frame)
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
