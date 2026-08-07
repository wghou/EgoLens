#!/usr/bin/env python3
"""Read-only desktop visualizer for EgoAfford scenes and role masks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


SCENE_PATTERN = re.compile(r"^scene_(\d+)$")
STEP_PATTERN = re.compile(r"^step_(\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)
ROLE_NAMES = ("Object", "Instrument", "Destination")
ROLE_COLORS = np.asarray(
    ((255, 64, 64), (64, 200, 96), (64, 128, 255)), dtype=np.float32
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open EgoAfford scenes in a read-only desktop visualizer."
    )
    parser.add_argument(
        "data_root",
        type=Path,
        help="Dataset root containing scene_<id> directories.",
    )
    parser.add_argument(
        "--scene",
        type=int,
        default=None,
        help="Scene ID to open first. Defaults to the first available scene.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    return value if isinstance(value, dict) else {}


def load_sparse_masks(path: Path) -> np.ndarray:
    """Load the packed sparse-mask format used by EgoAfford."""
    with np.load(path, allow_pickle=False) as data:
        if "shape" not in data:
            raise ValueError(f"Missing 'shape' in {path.name}.")
        shape = tuple(int(value) for value in data["shape"])
        if len(shape) != 4 or shape[1] != 3:
            raise ValueError(f"Expected mask shape (N, 3, H, W), found {shape}.")

        masks = np.zeros(shape, dtype=bool)
        if "nonempty_idx" not in data or data["nonempty_idx"].size == 0:
            return masks

        required = ("packed", "nonempty_shape")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Missing {', '.join(missing)} in {path.name}.")

        indices = np.asarray(data["nonempty_idx"], dtype=np.int64)
        nonempty_shape = tuple(int(value) for value in data["nonempty_shape"])
        bit_count = int(np.prod(nonempty_shape))
        unpacked = np.unpackbits(data["packed"])[:bit_count]
        nonempty_masks = unpacked.reshape(nonempty_shape).astype(bool)
        masks[indices[:, 0], indices[:, 1]] = nonempty_masks
        return masks


def discover_scenes(data_root: Path) -> list[int]:
    scenes = []
    for path in data_root.iterdir():
        match = SCENE_PATTERN.fullmatch(path.name)
        if path.is_dir() and match:
            scenes.append(int(match.group(1)))
    return sorted(scenes)


def discover_images(scene_path: Path) -> list[tuple[int, Path]]:
    images = []
    for path in scene_path.iterdir():
        match = STEP_PATTERN.fullmatch(path.name)
        if path.is_file() and match:
            images.append((int(match.group(1)), path))
    return sorted(images)


class DatasetVisualizer(QMainWindow):
    def __init__(self, data_root: Path, initial_scene: int | None) -> None:
        super().__init__()
        self.data_root = data_root.resolve()
        self.scene_ids = discover_scenes(self.data_root)
        if not self.scene_ids:
            raise ValueError(f"No scene_<id> directories found under {self.data_root}.")

        self.scene_pos = 0
        if initial_scene is not None:
            if initial_scene not in self.scene_ids:
                raise ValueError(f"Scene {initial_scene} does not exist in the dataset root.")
            self.scene_pos = self.scene_ids.index(initial_scene)

        self.image_entries: list[tuple[int, Path]] = []
        self.state_pos = 0
        self.current_image: np.ndarray | None = None
        self.task: dict[str, Any] = {}
        self.object_meta: dict[str, Any] = {}
        self.alternatives: dict[str, Any] = {}
        self.single_masks: np.ndarray | None = None
        self.multi_masks: np.ndarray | None = None
        self.multi_entries: list[dict[str, Any]] = []

        self.setWindowTitle("EgoAfford Dataset Visualizer")
        self.resize(1400, 900)
        self._build_ui()
        self._load_scene(self.scene_pos)

    @property
    def scene_id(self) -> int:
        return self.scene_ids[self.scene_pos]

    @property
    def state_id(self) -> int:
        return self.image_entries[self.state_pos][0]

    @property
    def scene_path(self) -> Path:
        return self.data_root / f"scene_{self.scene_id}"

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll = QScrollArea()
        scroll.setWidget(self.image_label)
        scroll.setWidgetResizable(False)
        layout.addWidget(scroll, 4)

        panel = QWidget()
        panel.setMaximumWidth(430)
        panel_layout = QVBoxLayout(panel)
        layout.addWidget(panel, 1)

        scene_nav = QHBoxLayout()
        previous_scene = QPushButton("Previous (A)")
        next_scene = QPushButton("Next (D)")
        previous_scene.clicked.connect(lambda: self._change_scene(-1))
        next_scene.clicked.connect(lambda: self._change_scene(1))
        scene_nav.addWidget(previous_scene)
        scene_nav.addWidget(next_scene)
        panel_layout.addLayout(scene_nav)

        jump_layout = QHBoxLayout()
        self.scene_input = QLineEdit()
        self.scene_input.setPlaceholderText("Scene ID")
        go_button = QPushButton("Go")
        go_button.clicked.connect(self._jump_to_scene)
        jump_layout.addWidget(self.scene_input)
        jump_layout.addWidget(go_button)
        panel_layout.addLayout(jump_layout)

        state_nav = QHBoxLayout()
        previous_state = QPushButton("Previous state (Q)")
        next_state = QPushButton("Next state (E)")
        previous_state.clicked.connect(lambda: self._change_state(-1))
        next_state.clicked.connect(lambda: self._change_state(1))
        state_nav.addWidget(previous_state)
        state_nav.addWidget(next_state)
        panel_layout.addLayout(state_nav)

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        panel_layout.addWidget(self.info_label)

        panel_layout.addWidget(QLabel("Action candidate"))
        self.candidate_box = QComboBox()
        self.candidate_box.currentIndexChanged.connect(self._refresh_display)
        panel_layout.addWidget(self.candidate_box)

        self.show_masks = QCheckBox("Show mask overlay")
        self.show_masks.setChecked(True)
        self.show_masks.stateChanged.connect(self._refresh_display)
        panel_layout.addWidget(self.show_masks)

        self.role_checks: list[QCheckBox] = []
        for role, color in zip(ROLE_NAMES, ("red", "green", "blue")):
            checkbox = QCheckBox(f"{role} ({color})")
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self._refresh_display)
            panel_layout.addWidget(checkbox)
            self.role_checks.append(checkbox)

        panel_layout.addWidget(QLabel("Task"))
        self.task_label = QLabel()
        self.task_label.setWordWrap(True)
        self.task_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        panel_layout.addWidget(self.task_label)

        panel_layout.addWidget(QLabel("Objects"))
        self.objects_label = QLabel()
        self.objects_label.setWordWrap(True)
        self.objects_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        panel_layout.addWidget(self.objects_label)
        panel_layout.addStretch()

    def _load_scene(self, scene_pos: int) -> None:
        scene_path = self.data_root / f"scene_{self.scene_ids[scene_pos]}"
        image_entries = discover_images(scene_path)
        if not image_entries:
            QMessageBox.warning(self, "Missing Images", f"No step images found in {scene_path}.")
            return

        try:
            task = load_json(scene_path / "task.json")
            object_meta = load_json(scene_path / "obj_meta.json")
            alternatives = load_json(scene_path / "alternatives.json")
            single_masks = self._load_optional_masks(scene_path / "masks.npz")
            multi_masks = self._load_optional_masks(scene_path / "masks_multi.npz")
            multi_index = load_json(scene_path / "masks_multi_index.json")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            QMessageBox.critical(self, "Scene Load Failed", str(error))
            return

        self.scene_pos = scene_pos
        self.image_entries = image_entries
        self.state_pos = 0
        self.task = task
        self.object_meta = object_meta
        self.alternatives = alternatives
        self.single_masks = single_masks
        self.multi_masks = multi_masks
        entries = multi_index.get("entries", [])
        self.multi_entries = entries if isinstance(entries, list) else []
        self.scene_input.setText(str(self.scene_id))
        self._load_state()

    @staticmethod
    def _load_optional_masks(path: Path) -> np.ndarray | None:
        return load_sparse_masks(path) if path.is_file() else None

    def _load_state(self) -> None:
        _, image_path = self.image_entries[self.state_pos]
        try:
            self.current_image = np.asarray(Image.open(image_path).convert("RGB"))
        except OSError as error:
            QMessageBox.critical(self, "Image Load Failed", str(error))
            return
        self._refresh_candidates()
        self._refresh_text()
        self._refresh_display()

    def _refresh_candidates(self) -> None:
        self.candidate_box.blockSignals(True)
        self.candidate_box.clear()
        state = self.state_id

        candidates = []
        if self.multi_masks is not None:
            for entry in self.multi_entries:
                if int(entry.get("state", -1)) != state:
                    continue
                mask_index = int(entry.get("mask_index", -1))
                if not 0 <= mask_index < len(self.multi_masks):
                    continue
                if "cand_key" in entry:
                    label = str(entry["cand_key"])
                else:
                    action = int(entry.get("action", -1))
                    steps = self.task.get("steps", [])
                    label = str(steps[action]) if 0 <= action < len(steps) else f"Action {action}"
                candidates.append((label, "multi", mask_index))

            steps = self.task.get("steps", [])
            base_text = steps[state] if state < len(steps) else None
            candidate_map = self.alternatives.get("candidate_names", {})
            preferred = candidate_map.get(base_text, []) if base_text is not None else []
            if isinstance(preferred, list):
                order = {str(label): position for position, label in enumerate(preferred)}
                candidates.sort(key=lambda item: order.get(item[0], len(order)))

        if not candidates and self.single_masks is not None and state < len(self.single_masks):
            steps = self.task.get("steps", [])
            label = str(steps[state]) if state < len(steps) else f"State {state}"
            candidates.append((label, "single", state))

        if candidates:
            for label, source, mask_index in candidates:
                self.candidate_box.addItem(label, (source, mask_index))
        else:
            self.candidate_box.addItem("No mask annotation", None)
        self.candidate_box.blockSignals(False)

    def _selected_masks(self) -> np.ndarray | None:
        selection = self.candidate_box.currentData()
        if selection is None:
            return None
        source, index = selection
        masks = self.multi_masks if source == "multi" else self.single_masks
        if masks is None or not 0 <= index < len(masks):
            return None
        return masks[index]

    def _refresh_display(self, *_args: object) -> None:
        if self.current_image is None:
            return
        display = self.current_image.astype(np.float32).copy()
        masks = self._selected_masks()

        if self.show_masks.isChecked() and masks is not None:
            if masks.shape[1:] != display.shape[:2]:
                self.statusBar().showMessage(
                    f"Mask size {masks.shape[1:]} does not match image size {display.shape[:2]}."
                )
            else:
                self.statusBar().clearMessage()
                for channel, checkbox in enumerate(self.role_checks):
                    if checkbox.isChecked():
                        mask = masks[channel].astype(bool)
                        display[mask] = display[mask] * 0.5 + ROLE_COLORS[channel] * 0.5

        display = np.ascontiguousarray(np.clip(display, 0, 255).astype(np.uint8))
        height, width, channels = display.shape
        image = QImage(
            display.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image)
        self.image_label.setPixmap(pixmap)
        self.image_label.resize(pixmap.size())

    def _refresh_text(self) -> None:
        steps = self.task.get("steps", [])
        state_text = str(steps[self.state_id]) if self.state_id < len(steps) else "-"
        self.info_label.setText(
            f"Scene {self.scene_id} ({self.scene_pos + 1}/{len(self.scene_ids)})\n"
            f"State {self.state_id} ({self.state_pos + 1}/{len(self.image_entries)})\n"
            f"Default next action: {state_text}"
        )

        task_text = self.task.get("main_task")
        self.task_label.setText(str(task_text))

        meta = self.object_meta.get("object_meta", {})
        objects = meta.get("objects", []) if isinstance(meta, dict) else []
        lines = []
        for item in objects:
            if isinstance(item, dict):
                number = item.get("number", "")
                name = item.get("object", item.get("name", ""))
                lines.append(f"{number} {name}".strip())
            else:
                lines.append(str(item))
        self.objects_label.setText("\n".join(lines) if lines else "Not provided")

    def _change_scene(self, delta: int) -> None:
        target = self.scene_pos + delta
        if 0 <= target < len(self.scene_ids):
            self._load_scene(target)

    def _jump_to_scene(self) -> None:
        try:
            scene_id = int(self.scene_input.text())
            scene_pos = self.scene_ids.index(scene_id)
        except ValueError:
            QMessageBox.warning(self, "Scene Not Found", "Enter an available numeric scene ID.")
            return
        self._load_scene(scene_pos)

    def _change_state(self, delta: int) -> None:
        target = self.state_pos + delta
        if 0 <= target < len(self.image_entries):
            self.state_pos = target
            self._load_state()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_A:
            self._change_scene(-1)
        elif event.key() == Qt.Key_D:
            self._change_scene(1)
        elif event.key() == Qt.Key_Q:
            self._change_state(-1)
        elif event.key() == Qt.Key_E:
            self._change_state(1)
        else:
            super().keyPressEvent(event)


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser()
    if not data_root.is_dir():
        print(f"Dataset root does not exist: {data_root}", file=sys.stderr)
        return 2

    app = QApplication(sys.argv)
    try:
        window = DatasetVisualizer(data_root, args.scene)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
