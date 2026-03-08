import logging
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QProgressDialog

from pupil_labs import neon_player

logger = logging.getLogger(__name__)


def export_video(
    video_path: Path,
    scene_times: np.ndarray,
    window_indices: list[int],
) -> bool:
    """Render the Neon Player scene (with all plugin overlays) for every
    frame in *window_indices* and write the result to an MP4 file.

    Returns True on success, False on failure or cancellation.
    """
    app = neon_player.instance()
    vid_w = app.recording.scene.width
    vid_h = app.recording.scene.height

    window_start_ns = scene_times[window_indices[0]]
    n_frames = len(window_indices)
    duration_ns = scene_times[window_indices[-1]] - window_start_ns
    fps = (
        (n_frames - 1) / (duration_ns / 1e9)
        if duration_ns > 0 and n_frames > 1
        else 30.0
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (vid_w, vid_h))

    if not writer.isOpened():
        logger.error("Failed to open video writer for %s", video_path)
        return False

    progress = QProgressDialog("Exporting video…", "Cancel", 0, n_frames)
    progress.setWindowModality(Qt.WindowModality.ApplicationModal)
    progress.setMinimumDuration(0)

    try:
        for seq, frame_idx in enumerate(window_indices):
            if progress.wasCanceled():
                logger.info("Video export cancelled")
                writer.release()
                if video_path.exists():
                    video_path.unlink()
                return False

            ts = int(scene_times[frame_idx])

            qimg = QImage(vid_w, vid_h, QImage.Format.Format_ARGB32)
            qimg.fill(0)
            painter = QPainter(qimg)
            painter.setRenderHints(
                QPainter.RenderHint.Antialiasing
                | QPainter.RenderHint.SmoothPixmapTransform
            )
            app.render_to(painter, ts)
            painter.end()

            qimg_rgb = qimg.convertToFormat(QImage.Format.Format_RGB888)
            ptr = qimg_rgb.bits()
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape(vid_h, vid_w, 3)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            writer.write(bgr)

            progress.setValue(seq + 1)
    finally:
        writer.release()
        progress.close()

    logger.info(
        "Video exported to %s (%d frames, %.2f fps)", video_path, n_frames, fps
    )
    return True
