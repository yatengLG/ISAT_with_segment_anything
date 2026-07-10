# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
# Install in editable mode
pip install -e .

# Install dependencies only
pip install -r requirements.txt

# Run (root-level entry point)
python main.py

# Run (if installed as package)
isat-sam

# Build source distribution
python setup.py sdist

# Build Windows EXE (PyInstaller)
./build_exe.bat
```

There is no formal test suite. The `test/` directory contains two ad-hoc scripts (`coco_display.py`, `f32_vs_bf16.py`). Test coverage is verified by running the GUI app manually.

## Architecture

ISAT is a PyQt5 desktop app for semi-automatic image segmentation annotation, integrating Meta's Segment Anything Model (SAM/SAM2/SAM3/HQ-SAM/Mobile-SAM/Edge-SAM/Med2D-SAM). It publishes on PyPI as `isat-sam`.

### Entry point & startup sequence

`main.py` → `ISAT/main.py:main()` → `ISAT/widgets/mainwindow.py:MainWindow`

`MainWindow.__init__()` runs: `setupUi()` (Qt Designer UI) → `init_ui()` (docks/dialogs) → `reload_cfg()` (YAML configs) → `init_connect()` (signal wiring) → `InitSegAnyThread` (async SAM model load) → `CheckLatestVersionThread`.

### ui/ vs widgets/ split

- **`ISAT/ui/`** — auto-generated from Qt Designer `.ui` files. Pure layout classes (`Ui_MainWindow`, `Ui_SettingDialog`, etc.). Never hand-edit the `.py` files; edit the `.ui` files in Qt Designer.
- **`ISAT/widgets/`** — hand-written logic classes. Pattern: `class SomeWidget(QDialog, Ui_SomeWidget)` inherits layout and adds behavior.

### Two-layer data model

- **`ISAT/annotation.py`** — disk representation. `Object` (category, group, segmentation vertices, area, bbox, layer, iscrowd, note) and `Annotation` (image metadata + list of Objects, load/save as JSON).
- **`ISAT/widgets/polygon.py`** — visual representation. `Polygon` is a `QGraphicsItem` with `Vertex` children. Conversion: `Polygon.to_object()` for save, `Polygon.load_object()` for display.

### Annotation data flow

**Create:** user draws or SAM predicts mask → `cv2.findContours` → `Polygon` on scene → `mainwindow.polygons` list → on save: `polygon.to_object()` → `Annotation.objects` → `Annotation.save_annotation()`

**Load:** `MainWindow.show_image()` → `Annotation(image_path, label_path)` → `annotation.load_annotation()` → for each Object, create `Polygon` with `load_object()` → add to scene + `mainwindow.polygons`

### Canvas interaction

`ISAT/widgets/canvas.py` has two classes:
- **`AnnotationScene`** (`QGraphicsScene`) — mouse/key event handling. Mode machine: `STATUSMode` (VIEW/CREATE/EDIT/REPAINT) × `DRAWMode` (POLYGON/SEGMENTANYTHING_POINT/SEGMENTANYTHING_BOX).
- **`AnnotationView`** (`QGraphicsView`) — zoom, pan, fit-to-window.

### SAM model integration

`ISAT/segment_any/segment_any.py` is the **facade**: `SegAny` for images, `SegAnyVideo` for video (SAM2/SAM2.1/SAM3). It auto-detects model type from the checkpoint filename string (e.g., `"mobile_sam"`, `"sam2"`, `"sam3"`, `"edge_sam"`) and dispatches to the correct sub-package predictor. All 8 SAM variants are **vendored directly** in the source tree under `ISAT/segment_any/` — they are not pip dependencies.

`ISAT/segment_any/model_zoo.py` — registry of 20 model entries with download URLs (HuggingFace + ModelScope mirrors), memory/param counts, and image/video capability flags.

### Config files

- **`ISAT/software.yaml`** — software settings (mask_alpha, language, contour_mode, auto_save, bfloat16), keyboard shortcuts, and category labels with colors.
- **`ISAT/isat.yaml`** — category labels only (name + hex color).
- **`ISAT/configs.py`** — `ISAT_ROOT`, `CHECKPOINT_PATH`, enums (`STATUSMode`, `DRAWMode`, `MAPMode`, `CONTOURMode`, `CONTOURMethod`), YAML load/save helpers.
- SAM2/SAM2.1 model architectures are defined as Hydra `@package _global_` YAML configs in `ISAT/segment_any/sam2/configs/`.

Note: `*.yaml` is in `.gitignore`, so config changes are not tracked by default.

### Concurrency model

All heavy work runs on `QThread` subclasses communicating via `pyqtSignal`:
- `InitSegAnyThread` — SAM model loading (several GB)
- `SegAnyThread` — image encoder, caches features for ±1 adjacent images
- `SegAnyVideoThread` — video frame-by-frame propagation
- `DownloadThread` — checkpoint download with HTTP range resume
- `GPUResource_Thread` — polls `nvidia-smi` for status bar display
- `CheckLatestVersionThread` — PyPI version check
- `AutoSegmentThread` — batch auto-segmentation

### Format converters

`ISAT/formats/` — all inherit from base `ISAT` class in `formats/isat.py`: `COCO`, `YOLO`, `LABELME`, `VOC`, `VOCDetect`. The `Converter` class in `widgets/converter_dialog.py` orchestrates them.

### Plugin system

Plugins are discovered via `entry_points` group `isat.plugins`, must subclass `ISAT/plugin_base.py:PluginBase`. `MainWindow` fires lifecycle hooks: `trigger_application_start/shutdown`, `trigger_before/after_image_open`, `trigger_before_annotations_save`, `trigger_after_annotation_changed`, `trigger_after_sam_encode_finished`.

### Key dependencies

`torch>=2.3.0`, `pyqt5`, `opencv_python_headless`, `shapely`, `pycocotools`, `hydra-core`, `timm`, `einops`, `pydicom`, `fuzzywuzzy`, `imgviz`, `orjson`
