# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "opencv-python",
#     "numpy",
# ]
# ///
import logging
import typing as T
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout
from qt_property_widgets.utilities import FilePath, property_params

from pupil_labs import neon_player
from pupil_labs.neon_player import ProgressUpdate
from pupil_labs.neon_recording import NeonRecording


logger = logging.getLogger(__name__)


class AOISelectionLabel(QLabel):
    """A QLabel that lets the user click four corners to define a
    quadrilateral AOI.  Existing corners can be dragged to adjust,
    and clicking inside the finished quad moves the whole shape."""

    _CORNER_RADIUS = 8  # pixels — hit-test & draw radius

    def __init__(self, pixmap: QPixmap, current_aoi: list[list[float]] | None = None):
        super().__init__()
        self._pixmap = pixmap
        self.setPixmap(pixmap)
        self.setFixedSize(pixmap.size())

        # corners: list of [x, y] — ordered TL, TR, BR, BL (or any order)
        self._corners: list[list[float]] = []
        if current_aoi is not None and len(current_aoi) == 4:
            self._corners = [list(pt) for pt in current_aoi]

        self._drag_idx: int | None = None  # index of corner being dragged
        self._dragging_all = False  # True when moving the whole quad
        self._drag_offset: list[list[float]] = []  # per-corner offset for whole-quad drag

    @property
    def selected_quad(self) -> list[list[float]] | None:
        if len(self._corners) != 4:
            return None
        return [list(c) for c in self._corners]

    # ---- hit-testing helpers ---- #

    def _corner_at(self, px: float, py: float) -> int | None:
        """Return index of the corner near (px, py), or None."""
        for i, (cx, cy) in enumerate(self._corners):
            if (px - cx) ** 2 + (py - cy) ** 2 <= self._CORNER_RADIUS ** 2:
                return i
        return None

    @staticmethod
    def _point_in_quad(px: float, py: float, corners: list[list[float]]) -> bool:
        """Winding-number test for point-in-convex/concave quadrilateral."""
        n = len(corners)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = corners[i]
            xj, yj = corners[j]
            if ((yi > py) != (yj > py)) and (
                px < (xj - xi) * (py - yi) / (yj - yi) + xi
            ):
                inside = not inside
            j = i
        return inside

    # ---- mouse events ---- #

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        px, py = event.pos().x(), event.pos().y()

        # Priority 1: drag an existing corner
        idx = self._corner_at(px, py)
        if idx is not None:
            self._drag_idx = idx
            self.update()
            return

        # Priority 2: click inside completed quad — move whole shape
        if len(self._corners) == 4 and self._point_in_quad(px, py, self._corners):
            self._dragging_all = True
            self._drag_offset = [
                [c[0] - px, c[1] - py] for c in self._corners
            ]
            self.update()
            return

        # Priority 3: place a new corner (up to 4)
        if len(self._corners) < 4:
            self._corners.append([px, py])
            self.update()

    def mouseMoveEvent(self, event):
        px, py = event.pos().x(), event.pos().y()
        if self._drag_idx is not None:
            self._corners[self._drag_idx] = [px, py]
            self.update()
        elif self._dragging_all:
            for i, (ox, oy) in enumerate(self._drag_offset):
                self._corners[i] = [px + ox, py + oy]
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_idx = None
            self._dragging_all = False

    # ---- drawing ---- #

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._corners:
            return
        painter = QPainter(self)
        painter.setPen(QPen(QColor(Qt.GlobalColor.green), 2, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)

        n = len(self._corners)
        # Draw edges between placed corners (close polygon when 4 corners)
        edge_count = n if n == 4 else n - 1
        for i in range(edge_count):
            x1, y1 = self._corners[i]
            x2, y2 = self._corners[(i + 1) % n]
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Draw corner handles
        painter.setPen(QPen(QColor(Qt.GlobalColor.green), 1, Qt.PenStyle.SolidLine))
        painter.setBrush(QColor(0, 255, 0, 80))
        r = self._CORNER_RADIUS
        for cx, cy in self._corners:
            painter.drawEllipse(int(cx) - r, int(cy) - r, r * 2, r * 2)

        painter.end()


class ReferenceImageMapper(neon_player.Plugin):
    label = "Reference Image Mapper"

    def __init__(self):
        super().__init__()
        self.homographies = None
        self.ref_image = None
        self.ref_h = 0
        self.ref_w = 0

        self._reference_image_path = FilePath()
        self._start_time = 10.0
        self._stop_time = 30.0
        self._min_matches = 15
        self._aoi: list[list[float]] | None = None  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] in ref image coords
        self._outline_color = QColor(Qt.GlobalColor.green)
        self._gaze_color = QColor(Qt.GlobalColor.red)
        self._gaze_radius = 15
        self._font = QFont("Arial", 12)

        self.mapping_job = None

    # ------------------------------------------------------------------ #
    # Recording lifecycle
    # ------------------------------------------------------------------ #

    def on_recording_loaded(self, recording: NeonRecording) -> None:
        self._try_load_reference_image()

    # ------------------------------------------------------------------ #
    # Reference image loading
    # ------------------------------------------------------------------ #

    def _try_load_reference_image(self) -> None:
        """Attempt to load the reference image from the currently selected path."""
        self.ref_image = None
        self.homographies = None

        path = Path(self._reference_image_path)
        if not path.name:
            logger.info("No reference image selected")
            return
        if not path.is_file():
            logger.info("Reference image not found: %s", path)
            return

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            logger.info("Failed to read reference image: %s", path)
            return

        self.ref_image = img
        self.ref_h, self.ref_w = img.shape[:2]
        logger.info("Loaded reference image: %s", path.name)
        self._load_or_compute_cache()

    # ------------------------------------------------------------------ #
    # Homography cache
    # ------------------------------------------------------------------ #

    def _load_or_compute_cache(self) -> None:
        cache_file = self.get_cache_path() / "homographies.npy"

        if cache_file.exists():
            self._load_cache()
            return

        if self.app.headless:
            self.bg_compute_homographies()
            self._load_cache()
            return

        if self.mapping_job is not None:
            return

        self.mapping_job = self.job_manager.run_background_action(
            "Compute reference image mapping",
            "ReferenceImageMapper.bg_compute_homographies",
        )
        self.mapping_job.finished.connect(self._load_cache)

    def _load_cache(self) -> None:
        self.mapping_job = None
        cache_file = self.get_cache_path() / "homographies.npy"
        if cache_file.exists():
            self.homographies = np.load(str(cache_file), allow_pickle=True).tolist()

    # ------------------------------------------------------------------ #
    # Background computation
    # ------------------------------------------------------------------ #

    def bg_compute_homographies(self) -> T.Generator[ProgressUpdate, None, None]:
        """Detect the reference image in every scene frame and store the
        homography matrix (scene -> reference) for each frame.

        Uses ORB feature matching as primary method and falls back to
        Lucas-Kanade optical flow tracking from the last good detection
        when ORB fails, which helps bridge frames with motion blur or
        temporary occlusion.
        """

        orb = cv2.ORB_create(nfeatures=2000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        ref_gray = cv2.cvtColor(self.ref_image, cv2.COLOR_BGR2GRAY)
        kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)

        ref_corners = np.float32([
            [0, 0],
            [self.ref_w, 0],
            [self.ref_w, self.ref_h],
            [0, self.ref_h],
        ]).reshape(-1, 1, 2)

        # Optical flow parameters
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01
            ),
        )
        # State for optical flow fallback
        prev_gray: np.ndarray | None = None
        # Points in the *previous* scene frame that correspond to ref_pts_for_of
        prev_scene_pts: np.ndarray | None = None
        # Matching reference-image coordinates for those scene points
        ref_pts_for_of: np.ndarray | None = None
        of_miss_streak = 0  # consecutive frames tracked only by OF
        max_of_streak = 10  # reset OF tracking after this many frames

        homographies: list[dict[str, T.Any]] = []
        num_frames = len(self.recording.scene)
        scene_times = self.recording.scene.time
        rec_start = scene_times[0]
        start_ns = int(self._start_time * 1e9)
        stop_ns = int(self._stop_time * 1e9)

        for frame_idx, frame in enumerate(self.recording.scene):
            result: dict[str, T.Any] = {"found": False, "H": None, "corners": None}

            # Skip frames outside the start/stop time window
            elapsed = scene_times[frame_idx] - rec_start
            if elapsed < start_ns or elapsed > stop_ns:
                homographies.append(result)
                prev_gray = None
                prev_scene_pts = None
                ref_pts_for_of = None
                of_miss_streak = 0
                yield ProgressUpdate((frame_idx + 1) / num_frames)
                continue

            scene_gray = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2GRAY)
            frame_w, frame_h = frame.bgr.shape[1], frame.bgr.shape[0]
            orb_matched = False

            # ---- Primary: ORB feature matching ----
            kp_scene, des_scene = orb.detectAndCompute(scene_gray, None)

            if des_scene is not None and des_ref is not None and len(kp_scene) >= 2:
                matches = bf.knnMatch(des_ref, des_scene, k=2)

                good = []
                for pair in matches:
                    if len(pair) == 2:
                        m, n = pair
                        if m.distance < 0.75 * n.distance:
                            good.append(m)

                if len(good) >= self._min_matches:
                    src_pts = np.float32(
                        [kp_ref[m.queryIdx].pt for m in good]
                    ).reshape(-1, 1, 2)
                    dst_pts = np.float32(
                        [kp_scene[m.trainIdx].pt for m in good]
                    ).reshape(-1, 1, 2)

                    H, mask = cv2.findHomography(
                        dst_pts, src_pts, cv2.RANSAC, 3.0
                    )
                    inlier_ratio = (
                        mask.sum() / len(mask) if mask is not None else 0
                    )

                    if H is not None and inlier_ratio >= 0.4:
                        H_inv = np.linalg.inv(H)
                        scene_corners = cv2.perspectiveTransform(
                            ref_corners, H_inv
                        )
                        corners_2d = scene_corners.reshape(-1, 2)

                        if self._is_valid_detection(
                            corners_2d, frame_w, frame_h
                        ):
                            result = {
                                "found": True,
                                "H": H.tolist(),
                                "corners": corners_2d.tolist(),
                            }
                            orb_matched = True

                            # Seed optical-flow state with ORB inliers
                            inlier_mask = mask.ravel().astype(bool)
                            prev_scene_pts = dst_pts[inlier_mask].copy()
                            ref_pts_for_of = src_pts[inlier_mask].copy()
                            of_miss_streak = 0

            # ---- Fallback: Lucas-Kanade optical flow ----
            if (
                not orb_matched
                and prev_gray is not None
                and prev_scene_pts is not None
                and of_miss_streak < max_of_streak
            ):
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, scene_gray, prev_scene_pts, None, **lk_params
                )
                if next_pts is not None and status is not None:
                    good_mask = status.ravel() == 1
                    tracked_scene = next_pts[good_mask]
                    tracked_ref = ref_pts_for_of[good_mask]

                    if len(tracked_scene) >= self._min_matches:
                        H, mask = cv2.findHomography(
                            tracked_scene, tracked_ref, cv2.RANSAC, 5.0
                        )
                        inlier_ratio = (
                            mask.sum() / len(mask)
                            if mask is not None
                            else 0
                        )

                        if H is not None and inlier_ratio >= 0.35:
                            H_inv = np.linalg.inv(H)
                            scene_corners = cv2.perspectiveTransform(
                                ref_corners, H_inv
                            )
                            corners_2d = scene_corners.reshape(-1, 2)

                            if self._is_valid_detection(
                                corners_2d, frame_w, frame_h
                            ):
                                result = {
                                    "found": True,
                                    "H": H.tolist(),
                                    "corners": corners_2d.tolist(),
                                }

                                # Keep only inliers for next iteration
                                inlier_mask = mask.ravel().astype(bool)
                                prev_scene_pts = tracked_scene[
                                    inlier_mask
                                ].reshape(-1, 1, 2).copy()
                                ref_pts_for_of = tracked_ref[
                                    inlier_mask
                                ].reshape(-1, 1, 2).copy()
                                of_miss_streak += 1

            # Update previous-frame state
            if orb_matched or result["found"]:
                prev_gray = scene_gray
            elif prev_gray is not None:
                # Even on failure, advance prev_gray so OF can still
                # try on the next frame
                prev_gray = scene_gray

            if not result["found"] and not orb_matched:
                # Lost tracking — reset OF state after streak exceeded
                if of_miss_streak >= max_of_streak:
                    prev_scene_pts = None
                    ref_pts_for_of = None
                    of_miss_streak = 0

            homographies.append(result)
            yield ProgressUpdate((frame_idx + 1) / num_frames)

        destination = self.get_cache_path() / "homographies.npy"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, np.array(homographies, dtype=object))

    # ------------------------------------------------------------------ #
    # Detection validation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_valid_detection(
        corners: np.ndarray, frame_w: int, frame_h: int
    ) -> bool:
        """Reject geometrically implausible detections."""
        # 1. Convexity check — the quadrilateral must be convex
        #    (cross-product of consecutive edges must all have the same sign)
        n = len(corners)
        cross_signs = []
        for i in range(n):
            o = corners[i]
            a = corners[(i + 1) % n] - o
            b = corners[(i + 2) % n] - corners[(i + 1) % n]
            cross = a[0] * b[1] - a[1] * b[0]
            cross_signs.append(cross)

        if not (all(c > 0 for c in cross_signs) or all(c < 0 for c in cross_signs)):
            return False

        # 2. Area check — detected region shouldn't be too small or too large
        #    relative to the frame
        area = 0.5 * abs(
            sum(
                corners[i][0] * corners[(i + 1) % n][1]
                - corners[(i + 1) % n][0] * corners[i][1]
                for i in range(n)
            )
        )
        frame_area = frame_w * frame_h
        area_ratio = area / frame_area
        if area_ratio < 0.005 or area_ratio > 0.95:
            return False

        # 3. Aspect ratio sanity — no edge should be more than 10x longer
        #    than another (rejects extreme perspective distortions)
        edges = [
            np.linalg.norm(corners[(i + 1) % n] - corners[i]) for i in range(n)
        ]
        if max(edges) > 10 * min(edges):
            return False

        return True

    # ------------------------------------------------------------------ #
    # AOI helpers
    # ------------------------------------------------------------------ #

    def _aoi_corners(self) -> np.ndarray:
        """Return the 4 AOI corners as an array for perspectiveTransform."""
        return np.float32(self._aoi).reshape(-1, 1, 2)

    def _point_in_aoi(self, px: float, py: float) -> bool:
        """Check if a point is inside the AOI quadrilateral (ray-casting)."""
        corners = self._aoi
        n = len(corners)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = corners[i]
            xj, yj = corners[j]
            if ((yi > py) != (yj > py)) and (
                px < (xj - xi) * (py - yi) / (yj - yi) + xi
            ):
                inside = not inside
            j = i
        return inside

    # ------------------------------------------------------------------ #
    # Gaze helpers
    # ------------------------------------------------------------------ #

    def _get_gazes_for_scene(self, scene_idx: int):
        """Return gaze samples that fall within the given scene frame."""
        gaze_start_time = self.recording.scene[scene_idx].time
        after_mask = self.recording.gaze.time >= gaze_start_time

        if scene_idx < len(self.recording.scene) - 1:
            gaze_end_ts = self.recording.scene[scene_idx + 1].time
            before_mask = self.recording.gaze.time < gaze_end_ts
            time_mask = after_mask & before_mask
        else:
            time_mask = after_mask

        return self.recording.gaze[time_mask]

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def render(self, painter: QPainter, time_in_recording: int) -> None:
        if self.homographies is None or self.ref_image is None:
            return

        scene_idx = self.get_scene_idx_for_time(time_in_recording)
        if not (0 <= scene_idx < len(self.homographies)):
            return

        entry = self.homographies[scene_idx]
        if not entry["found"]:
            return

        H = np.array(entry["H"])
        H_inv = np.linalg.inv(H)

        # Determine the region to display: AOI or full reference image
        if self._aoi is not None:
            roi_corners = self._aoi_corners()
        else:
            roi_corners = np.float32([
                [0, 0],
                [self.ref_w, 0],
                [self.ref_w, self.ref_h],
                [0, self.ref_h],
            ]).reshape(-1, 1, 2)

        # Project AOI/reference corners into scene coordinates
        scene_roi = cv2.perspectiveTransform(roi_corners, H_inv)
        scene_pts = scene_roi.reshape(-1, 2)

        # Draw outline of the AOI in the scene
        pen = QPen(self.outline_color, 3, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        for i in range(4):
            x1, y1 = scene_pts[i]
            x2, y2 = scene_pts[(i + 1) % 4]
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Map gaze to reference image coordinates
        gaze_samples = self._get_gazes_for_scene(scene_idx)
        if len(gaze_samples.point) == 0:
            return

        # Use the mean gaze point for the current frame
        gaze_x, gaze_y = gaze_samples.point.mean(axis=0)
        gaze_pt = np.float32([[gaze_x, gaze_y]]).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(gaze_pt, H)
        ref_x, ref_y = mapped[0][0]

        # Check if mapped gaze is within the AOI (or full image if no AOI)
        if self._aoi is not None:
            in_region = self._point_in_aoi(ref_x, ref_y)
        else:
            in_region = 0 <= ref_x <= self.ref_w and 0 <= ref_y <= self.ref_h

        if in_region:
            # Draw gaze circle on the scene at the original gaze position
            painter.setPen(QPen(self.gaze_color, 2, Qt.PenStyle.SolidLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(
                int(gaze_x) - self.gaze_radius,
                int(gaze_y) - self.gaze_radius,
                self.gaze_radius * 2,
                self.gaze_radius * 2,
            )

            # Show mapped coordinates as text
            painter.setFont(self._font)
            painter.setPen(QPen(self.gaze_color, 1))
            label = f"AOI: ({ref_x:.0f}, {ref_y:.0f})"
            painter.drawText(int(gaze_x) + self.gaze_radius + 5, int(gaze_y) + 5, label)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def reference_image_path(self) -> FilePath:
        return self._reference_image_path

    @reference_image_path.setter
    def reference_image_path(self, value: FilePath) -> None:
        old = self._reference_image_path
        self._reference_image_path = value
        if old != value and self.recording is not None:
            # Clear old cache when a new image is selected
            cache_file = self.get_cache_path() / "homographies.npy"
            if cache_file.exists():
                cache_file.unlink()
            self._try_load_reference_image()

    @property
    @property_params(min=0, max=600, step=1)
    def start_time(self) -> float:
        return self._start_time

    @start_time.setter
    def start_time(self, value: float) -> None:
        self._start_time = value

    @property
    @property_params(min=1, max=600, step=1)
    def stop_time(self) -> float:
        return self._stop_time

    @stop_time.setter
    def stop_time(self, value: float) -> None:
        self._stop_time = value

    @neon_player.action
    def area_of_interest(self) -> None:
        """Open a window to select an AOI quadrilateral on the reference image."""
        if self.ref_image is None:
            logger.info("Load a reference image first")
            return

        # Convert BGR to RGB for Qt display
        rgb = cv2.cvtColor(self.ref_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        dialog = QDialog()
        dialog.setWindowTitle("Select Area of Interest (click 4 corners)")
        layout = QVBoxLayout(dialog)
        label = AOISelectionLabel(pixmap, self._aoi)
        layout.addWidget(label)
        dialog.setLayout(layout)

        if dialog.exec() == QDialog.DialogCode.Accepted or True:
            sel = label.selected_quad
            if sel is not None:
                self._aoi = sel
                logger.info("AOI set to: %s", self._aoi)
            else:
                self._aoi = None
                logger.info("AOI cleared")

    @neon_player.action
    def remap(self) -> None:
        """Clear cache and recompute homographies with current settings."""
        if self.ref_image is None:
            self._try_load_reference_image()
            return
        cache_file = self.get_cache_path() / "homographies.npy"
        if cache_file.exists():
            cache_file.unlink()
        self._load_or_compute_cache()

    @property
    @property_params(min=4, max=100, step=1)
    def min_matches(self) -> int:
        return self._min_matches

    @min_matches.setter
    def min_matches(self, value: int) -> None:
        self._min_matches = value

    @property
    def outline_color(self) -> QColor:
        return self._outline_color

    @outline_color.setter
    def outline_color(self, value: QColor) -> None:
        self._outline_color = value

    @property
    def gaze_color(self) -> QColor:
        return self._gaze_color

    @gaze_color.setter
    def gaze_color(self, value: QColor) -> None:
        self._gaze_color = value

    @property
    @property_params(min=2, max=50, step=1)
    def gaze_radius(self) -> int:
        return self._gaze_radius

    @gaze_radius.setter
    def gaze_radius(self, value: int) -> None:
        self._gaze_radius = value

    @property
    def font(self) -> QFont:
        return self._font

    @font.setter
    def font(self, value: QFont) -> None:
        self._font = value
