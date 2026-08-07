# EgoAfford Dataset Visualizer

This directory provides a read-only desktop application for exploring EgoAfford scenes. It displays step images, task and object metadata, and role-specific segmentation masks. The application does not create, modify, or delete dataset files.

## Installation

Create a dedicated environment and install the lightweight GUI dependencies:

```bash
conda create -n egoafford-visualizer python=3.11 -y
conda activate egoafford-visualizer
python -m pip install numpy Pillow PyQt5
```

SAM2, PyTorch, and a GPU are not required.

## Dataset layout

Pass a directory that contains `scene_<id>` subdirectories. The visualizer reads the following files when available:

- `step_<n>.png` or common JPEG variants: scene observations.
- `task.json`: the task description and default action sequence.
- `obj_meta.json`: object metadata shown in the side panel.
- `masks.npz`: single-reference object, instrument, and destination masks.
- `alternatives.json`, `masks_multi.npz`, and `masks_multi_index.json`: multi-reference action candidates and masks in test scenes.

The public dataset does not need `object_bbox.json`; bounding boxes are not loaded or displayed.

## Launch

Run the visualizer with the dataset root as its first argument:

```bash
python dataset_visualizer.py /path/to/EgoAfford
```

To open a particular scene initially:

```bash
python dataset_visualizer.py /path/to/EgoAfford --scene 42
```

## Controls

- Use **Previous** and **Next**, or `A` and `D`, to move between available scenes.
- Use the state buttons, or `Q` and `E`, to move between observations in a scene.
- Select an action candidate to view its masks. Train scenes show the single reference from `masks.npz`; test scenes show candidates from the multi-reference files when available.
- Toggle the complete overlay or individual semantic roles. Colors are red for object, green for instrument, and blue for destination.

All dataset access is read-only. The visualizer has no annotation, correction, or save controls.
