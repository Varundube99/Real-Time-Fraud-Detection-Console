import time
import threading


class MonitoringService:
    """
    Independent Producer Background Service.
    Drives transaction ingestion in an isolated thread.
    Pages act strictly as read-only Consumers.
    """
    def __init__(self, backend):
        self.backend = backend
        self.lock = threading.Lock()
        self._thread = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    @property
    def is_paused(self) -> bool:
        return self._pause_event.is_set()

    def start(self):
        with self.lock:
            if self._thread is not None and self._thread.is_alive():
                self._pause_event.clear()
                self._stop_event.clear()
                return
            self._stop_event.clear()
            self._pause_event.clear()
            if hasattr(self.backend, "monitor") and hasattr(self.backend.monitor, "state"):
                self.backend.monitor.state.start()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def pause(self):
        with self.lock:
            self._pause_event.set()
            if hasattr(self.backend, "monitor") and hasattr(self.backend.monitor, "state"):
                self.backend.monitor.state.pause()

    def resume(self):
        with self.lock:
            self._pause_event.clear()
            if hasattr(self.backend, "monitor") and hasattr(self.backend.monitor, "state"):
                self.backend.monitor.state.start()

    def stop(self):
        with self.lock:
            self._stop_event.set()
            self._pause_event.clear()
            if hasattr(self.backend, "monitor") and hasattr(self.backend.monitor, "state"):
                self.backend.monitor.state.stop()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

    def reset(self):
        self.stop()
        with self.lock:
            self.backend.reset()

    def _run_loop(self):
        while not self._stop_event.is_set():
            if not self._pause_event.is_set():
                with self.lock:
                    self.backend.process_next()
                
                speed = 1.0
                if hasattr(self.backend, "monitor") and hasattr(self.backend.monitor, "state"):
                    speed = getattr(self.backend.monitor.state, "speed", 1.0)
                
                sleep_time = 0.5 / max(0.1, float(speed))
            else:
                sleep_time = 0.2

            elapsed = 0.0
            while elapsed < sleep_time and not self._stop_event.is_set():
                time.sleep(0.05)
                elapsed += 0.05
