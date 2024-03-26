<h1 align='center'>ISAT_with_segment_anything</h1>
<h2 align='center'>A Interactive Semi-automatic Annotation Tool based Segment Anything</h2>
<p align='center'>
    <a href='https://github.com/yatengLG/ISAT_with_segment_anything' target="_blank"><img alt="GitHub forks" src="https://img.shields.io/github/stars/yatengLG/ISAT_with_segment_anything"></a>
    <a href='https://github.com/yatengLG/ISAT_with_segment_anything' target="_blank"><img alt="GitHub forks" src="https://img.shields.io/github/forks/yatengLG/ISAT_with_segment_anything"></a>
    <a href='https://pypi.org/project/isat-sam/' target="_blank"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/isat-sam"></a>
    <a href='https://pypi.org/project/isat-sam/' target="_blank"><img alt="Pepy Total Downlods" src="https://img.shields.io/pepy/dt/isat-sam"></a>
</p>
<p align='center'>
    <a href='README-cn.md'><b>[中文]</b></a>
    <a href='README.md'><b>[English]</b></a>
</p>
<p align='center'><img src="./display/标注.gif" alt="标注.gif"'></p>

Our tool enables interactive use of [segment anything](https://github.com/facebookresearch/segment-anything) for rapid image segmentation with low RAM requirements (optional bf16 mode).

Demo Video：[YouTube](https://www.youtube.com/watch?v=yLdZCPmX-Bc)

---

# Features
## Annotaion modes
- **Semi-automatic Annotation**: utilizes SAM with point and bbox prompts.
- **Manual Annotation**:  click or drag to draw polygons (0.15s per point).

## Annotation adjustments
- **Polygon Adjustments**: delete points and adjust object occlusions.
- **Polygon Visualization**: Preview groups and semantic/instance segmentation masks.

## Export annotations
- Supported formats: **MSCOCO**, **YOLO**, **LabelMe**, and **VOC** (also xml)

For more features, see the [Features Description](./docs/features%20description.md).

---

# Installation
There are three ways to install ISAT-SAM:
1. from source code (recommended)
2. pip install
3. from .exe

## Option 1: from source code
### (1) Create environment
```shell
conda create -n isat_env python=3.8
conda activate isat_env
```

### (2) Install ISAT_with_segment_anything and its dependencies
**To use GPU, please install [Pytorch-GPU](https://pytorch.org/) on Windows OS frist.**
```shell
git clone https://github.com/yatengLG/ISAT_with_segment_anything.git
cd ISAT_with_segment_anything
pip install -r requirements.txt
```

### (3) Download Segment anything pretrained checkpoint.

Download the checkpoint, and save in under: ISAT_with_segment_anything/ISAT/checkpoints

**After version 0.0.3, you can manage checkpoints within GUI, click [menubar]-[SAM]-[Model manage] to open the GUI.**

Now support [SAM](https://github.com/facebookresearch/segment-anything), [sam-hq](https://github.com/SysCV/sam-hq), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), and [EdgeSAM](https://github.com/chongzhou96/EdgeSAM) etc.

|  | pretrained checkpoint | memory | size |
|----|----|----|----|
|    SAM     | [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)           | 7305M | 2.6G |
|            | [sam_vit_l_0b3195.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)           | 5855M | 2.6G |
|            | [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)           | 4149M | 375M |
|   sam-hq   | [sam_hq_vit_h.pth](https://huggingface.co/lkeab/hq-sam/blob/main/sam_hq_vit_h.pth)                     | 7393M | 2.6G |
|            | [sam_hq_vit_l.pth](https://huggingface.co/lkeab/hq-sam/blob/main/sam_hq_vit_l.pth)                     | 5939M | 1.3G |
|            | [sam_hq_vit_b.pth](https://huggingface.co/lkeab/hq-sam/blob/main/sam_hq_vit_b.pth)                     | 4207M | 379M |
|            | [sam_hq_vit_tiny.pth](https://huggingface.co/lkeab/hq-sam/blob/main/sam_hq_vit_tiny.pth)               | 1463M |  43M |
| mobile-sam | [mobile_sam.pt](https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt)          | 1375M |  40M |
|  edge-sam  | [edge_sam.pth](https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam.pth)      |  960M |  39M |
|            | [edge_sam_3x.pth](https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_3x.pth)|  960M |  39M |
|   sam-med  | [sam-med2d_b.pth](https://drive.google.com/file/d/1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl/view?usp=drive_link)|1500M |  2.4G |

### (4) Run
```shell
python main.py
```
<br>

## Option 2: pip install
**Note that the version may be lower than source code version if installed with pip**
### (1) Create environment
```shell
conda create -n isat_env python=3.8
conda activate isat_env
```
### (2) pip install isat-sam
**To use GPU, please install [Pytorch-GPU](https://pytorch.org/) on Windows OS frist.**
```shell
pip install isat-sam
```
### (3) Run
```shell
isat-sam
```

<br>

## Option 3: install with .exe
### (1) download the .exe
**The version of exe maybe older than source code.**

Download three .zip files, total 2.7G

|        | Download link                                                      |
|--------|-----------------------------------------------------------|
| Baidu Netdisk | link：https://pan.baidu.com/s/1vD19PzvIT1QAJrAkSVFfhg code：ISAT |

Click main.exe to run the tool.

### (2) Download Segment anything pretrained checkpoint.

The download zip files, container sam_hq_vit_tiny.pth, but somebody told the model not support cpu.
You can download [mobile_sam.pt](https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt) to test the tool.

If you want use other models, see[Download Segment anything pretrained checkpoint](https://github.com/yatengLG/ISAT_with_segment_anything/blob/master/docs/README-en.md#3-download-segment-anything-pretrained-checkpoint)


---

# Usage
## 1. Annotation
```text
1. Choose the categories in left window of software.
    Edit the category in Toolbar-File-Setting.
    
2. Start annotating
    2.1 semi-automatic annotate with SAM.
        Click button named [Segment anything] start annotate(shortcut Q).
        Click interested area with left button of mouse, click uninterested area with right button of mouse, SAM will calcute masks.
    2.2 draw polygons manually.
        Click the button [Draw polygon] to start annotation (shortcut C).
        Left click to add point into the polygon.
        Hold the left click and drag will automaticly add point into the polygon (time interval of 0.15 seconds).
    2.3 Undo
        Click the button [Backspace] to return to the previous state (shortcut Z).
3. Finish the annotation with [Annotate finished] or shortcut E.
4. Save the annotation with [Save] or shortcut S
```
## 2. Polygon Modification
```text
1. Modify polygons coordinates
    Drag and drop polygon vertices to modify the shape of the polygon.
    Drag the polygon to adjust its position.
2. Modify polygons category
    Choose the polygon and click [Edit] or double click the polygon, and choose the new category in editing window. 
3. Occlusion modification
    Choose the polygon and click [To top] (shortcut T) or [To bottom] (shortcut B).
4. Delete polygon
    Choose the polygon and click [Delete] to delete the polygon.
```
## 3. Visualization
```text
1. Preview annotations
    Click the [Bit map] to preview semantic and instance annotation masks (shortcut space).
    The order of swithing is polygons-semantic-instance.
2. Image window
    Click [Zoom in], [Zoom out], [Fit window] (shortcut F) to adjust the zooming distances.
3. Show / hide polygons
    Click [Visible] to show / hide polygons (shortcut V).
4. Mask aplha (only effective when using SAM)
    Drag the [mask aplha] bar to adjust the mask transparency.
```
## 4. Convet annotations
ISAT has a specific format with .json. You can use export it to other formarts.
```text
1. ISAT to VOC
    Convert ISAT jsons to PNG images.
2. ISAT to COCO
    Convert ISAT jsons to COCO json.
3. ISAT to LABELME
    Convert ISAT jsons to LABELME jsons.
4. COCO to ISAT
    Convert COCO json to ISAT jsons.
```

---

# Star History

**Please support us with a star—it's like a virtual coffee!**
[![Star History Chart](https://api.star-history.com/svg?repos=yatengLG/ISAT_with_segment_anything&type=Date)](https://star-history.com/#yatengLG/ISAT_with_segment_anything&Date)


# Contributors

<table border="0">
<tr>
    <td><img alt="yatengLG" src="https://avatars.githubusercontent.com/u/31759824?v=4" width="60" height="60" href="">
    <td><img alt="Alias-z" src="https://avatars.githubusercontent.com/u/66273343?v=4" width="60" height="60" href="">
    <td>...
</td>
</tr>
<tr>
  <td><a href="https://github.com/yatengLG">yatengLG</a>
  <td><a href="https://github.com/Alias-z">Alias-z</a>
    <td><a href="https://github.com/yatengLG/ISAT_with_segment_anything/graphs/contributors">...</a>
</tr>
</table>


# Citation
```text
@misc{ISAT with segment anything,
  title={{ISAT with segment anything}: A Interactive Semi-automatic Annotation Tool based Segment Anything},
  url={https://github.com/yatengLG/ISAT_with_segment_anything},
  note={Open source software available from https://github.com/yatengLG/ISAT_with_segment_anything},
  author={yatengLG, Alias-z and horffmanwang},
  year={2023},
}
```
