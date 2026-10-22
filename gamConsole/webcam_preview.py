"""Optional live webcam tile for the menu (requires opencv-python)."""

from typing import Any, Optional, Tuple

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


class WebcamPreview:
    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self._cap: Optional["cv2.VideoCapture"] = None
        self._available = cv2 is not None

    @property
    def available(self) -> bool:
        return self._available

    def start(self) -> Tuple[bool, str]:
        if cv2 is None:
            return False, "Installer opencv-python: pip install -r requirements.txt"
        self.stop()
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            cap.release()
            return False, f"Kunne ikke åbne kamera {self.camera_index}."
        self._cap = cap
        return True, ""

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read_rgb_array(
        self, width: Optional[int] = None, height: Optional[int] = None
    ) -> Optional[Any]:
        """Return H×W×3 RGB numpy array, or None."""
        if self._cap is None or cv2 is None:
            return None
        ok, frame = self._cap.read()
        if not ok:
            return None
        if width is not None and height is not None:
            frame = cv2.resize(frame, (width, height))
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
