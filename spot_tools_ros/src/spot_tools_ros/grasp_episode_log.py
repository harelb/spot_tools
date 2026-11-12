"""Grasp episode log: everything an operator saw and answered during a pick,
plus every executor message around it, at the request rate (not video rate).

Layout, under the executor's output_dir:

    grasp_episodes/<YYYYmmdd-HHMMSS>_<n>/
        events.jsonl            one line per executor message, request, response,
                                image; {t, kind, ...}
        step_00/request.json    detection, source names, depth scales, camera
                                intrinsics/poses, ground plane, robot pose, walk-to
                                standoff/max distance, sizes of what was sent
        step_00/rgb_<i>_<source>.jpg    the exact JPEG sent to the operator
        step_00/depth_<i>_<source>.png  the exact 16-bit depth PNG sent (if any)
        step_00/response.json   the operator's answer (or preemption) and wait time
        semantic.png, mask.png  the executor's post-grasp feedback images

An episode opens at "Executing `pick` command" (or at a request arriving with
none open) and closes at "Pick skill success: ...". Logging never raises:
a failure is reported once through `warn` and the pick carries on.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time

PICK_START = "Executing `pick` command"
PICK_END = "Pick skill success:"


def _safe(name) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))[:60] or "image"


def _floats(values):
    return [float(v) for v in (values or [])]


class GraspEpisodeLog:
    def __init__(self, root: str, warn=print, clock=time.time):
        self.root = root
        self.warn = warn
        self.clock = clock
        self._lock = threading.RLock()
        self._dir = None
        self._steps = 0
        self._count = 0
        self._warned = False

    # ------------------------------------------------------------ internals
    def _guard(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001  logging must never break a pick
            if not self._warned:
                self._warned = True
                self.warn(f"grasp episode logging failed (continuing without it): {e!r}")
            return None

    def _open(self, reason):
        self._count += 1
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self.clock()))
        self._dir = os.path.join(self.root, f"{stamp}_{self._count:03d}")
        os.makedirs(self._dir, exist_ok=True)
        self._steps = 0
        self._event("episode_start", reason=reason)

    def _event(self, kind, **fields):
        if self._dir is None:
            return
        line = json.dumps({"t": self.clock(), "kind": kind, **fields}, default=str)
        with open(os.path.join(self._dir, "events.jsonl"), "a") as f:
            f.write(line + "\n")

    def _write(self, rel, data: bytes):
        path = os.path.join(self._dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return rel

    # -------------------------------------------------------------- public
    @property
    def directory(self):
        return self._dir

    def message(self, level: str, text: str):
        """Every executor feedback.print line; opens/closes episodes."""
        self._guard(self._message, level, str(text))

    def _message(self, level, text):
        with self._lock:
            if text.startswith(PICK_START):
                if self._dir is not None:
                    self._event("episode_end", outcome="superseded")
                self._open("pick")
            self._event("log", level=level, text=text)
            if text.startswith(PICK_END) and self._dir is not None:
                self._event("episode_end", outcome=text[len(PICK_END):].strip())
                self._dir = None

    def request(self, msg, semantic_class=None, raw_jpegs=None):
        """The ManipulationApprovalRequest exactly as published; returns the step
        id. `raw_jpegs` are JPEGs of the raw images when the request is not
        compressed (the message then carries sensor_msgs/Image)."""
        return self._guard(self._request, msg, semantic_class, raw_jpegs)

    def _request(self, msg, semantic_class, raw_jpegs):
        with self._lock:
            if self._dir is None:
                self._open("request")
            step = f"step_{self._steps:02d}"
            self._steps += 1
            names = list(msg.image_source_names)
            jpegs = [bytes(c.data) for c in msg.compressed_images] or list(raw_jpegs or [])
            images, total = [], 0
            for i, data in enumerate(jpegs):
                name = _safe(names[i] if i < len(names) else i)
                entry = {"index": i, "source": names[i] if i < len(names) else None}
                if data:
                    entry["rgb"] = self._write(f"{step}/rgb_{i}_{name}.jpg", data)
                    total += len(data)
                if i < len(msg.depth_images) and len(msg.depth_images[i].data):
                    depth = bytes(msg.depth_images[i].data)
                    entry["depth"] = self._write(f"{step}/depth_{i}_{name}.png", depth)
                    entry["depth_scale"] = float(msg.depth_scales[i])
                    total += len(depth)
                images.append(entry)
            n = len(images)
            intr, poses = _floats(msg.camera_intrinsics), _floats(msg.camera_poses)
            record = {
                "step": step,
                "t": self.clock(),
                "robot_id": msg.robot_id,
                "semantic_class": semantic_class,
                "has_detection": bool(msg.has_detection),
                "detection_image_index": int(msg.detection_image_index),
                "detection_xy": [int(msg.image_x), int(msg.image_y)],
                "images": images,
                "cameras": [
                    {"fx_fy_cx_cy": intr[4 * i : 4 * i + 4], "pose_xyz_wxyz": poses[7 * i : 7 * i + 7]}
                    for i in range(n)
                ],
                "ground_plane_point_normal": _floats(msg.ground_plane),
                "robot_pose_xy_yaw": _floats(msg.robot_pose),
                "walk_to_standoff": float(msg.walk_to_standoff),
                "walk_to_max_distance": float(msg.walk_to_max_distance),
                "bytes_sent": total,
                "frame": "vision (Spot fixed frame at capture time)",
            }
            with open(os.path.join(self._dir, step, "request.json"), "w") as f:
                json.dump(record, f, indent=1)
            self._event("request", step=step, images=n, detection=record["detection_xy"]
                        if record["has_detection"] else None)  # fmt: skip
            return step

    def response(self, step, sent_at, msg=None, preempted=False):
        """The operator's answer for `step` (msg: ManipulationApprovalResponse)."""
        self._guard(self._response, step, sent_at, msg, preempted)

    def _response(self, step, sent_at, msg, preempted):
        with self._lock:
            if self._dir is None or step is None:
                return
            now = self.clock()
            record = {"step": step, "t": now, "wait_s": round(now - sent_at, 2), "preempted": preempted}
            if msg is not None:
                record.update(
                    approve=bool(msg.approve),
                    walk_to=bool(msg.walk_to),
                    image_index=int(msg.image_index),
                    xy=[int(msg.image_x), int(msg.image_y)],
                    has_stand_point=bool(msg.has_stand_point),
                    stand_xy=[int(msg.stand_x), int(msg.stand_y)] if msg.has_stand_point else None,
                )
                record["decision"] = (
                    "approve" if msg.approve else "walk_to" if msg.walk_to else "reject"
                )
            else:
                record["decision"] = "preempted" if preempted else "none"
            os.makedirs(os.path.join(self._dir, step), exist_ok=True)
            with open(os.path.join(self._dir, step, "response.json"), "w") as f:
                json.dump(record, f, indent=1)
            self._event("response", **record)

    def image(self, name: str, png: bytes):
        """A feedback image (e.g. the post-grasp semantic/mask view)."""
        self._guard(self._image, name, png)

    def _image(self, name, png):
        with self._lock:
            if self._dir is None:
                return
            rel = self._write(f"{_safe(name)}_{self._steps:02d}.png", png)
            self._event("image", name=name, file=rel)
