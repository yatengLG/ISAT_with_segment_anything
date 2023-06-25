# ISAT with segment anything
# Interactive semi-automatic annotation tool for image segmentation.

![annotate.gif](./display/标注%20-big-original.gif)

Quick annotate for image segmentation by [segment anything](https://github.com/facebookresearch/segment-anything)

**Project updating，suggestions welcome**

Demo Video：[youtube](https://www.youtube.com/watch?v=yLdZCPmX-Bc)

演示视频：[bilibili](https://www.bilibili.com/video/BV1Lk4y1J7uB/)

[中文](README.md)         [English](README-en.md)

# Feature

1. Support semantic segmentation and instance segmentation.
2. Integrating SAM (segment anything model) for interactive semi-automatic annotation of image segmentation.
3. Interactive correction mask, by clicking on the area of interest (not of interest) with the left (right) mouse button.
4. Support manual creation of polygons.
5. Support modifying polygons.
6. Support for adjusting polygon occlusion.
7. Support preview annotation result.
8. ISAT format json, contains more information.
9. Support opening JSON files annotated by labelme (please backup a copy before opening), and modify.
10. Support exporting ISAT format json to VOC as single-channel png images.
11. Support exporting ISAT format jsons to COCO format json.
12. Support exporting ISAT format json to LabelMe format json.
13. Support exporting COCO format json to ISAT format jsons.

# INSTALL （recommend to use mamba or conda）
## 1. Run the source code
### (1) Create environment
```shell
conda create -n ISAT_with_segment_anything python==3.8 -y
conda activate ISAT_with_segment_anything
```

### (2) Install Segment anything
```shell
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
cd ..
```

### (3) Install ISAT_with_segment_anything
```shell
git clone https://github.com/yatengLG/ISAT_with_segment_anything.git
cd ISAT_with_segment_anything
pip install -r requirements.txt
```
### (3) （in case the installation above fails）Manually install ISAT_with_segment_anything with GPU support
```shell
git clone https://github.com/yatengLG/ISAT_with_segment_anything.git
cd ISAT_with_segment_anything
```
Windows
```shell
mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
mamba install -c fastai opencv-python-headless -y
mamba install imgviz mahotas numpy pillow pyqt pyyaml pycocotools -y
```
Mac
```shell
conda install pytorch::pytorch torchvision torchaudio -c pytorch -y
pip install opencv-python-headless imgviz mahotas pycocotools 
conda install numpy pillow pyqt pyyaml -y
```
### (4) Download Segment anything pretrained checkpoint.

Download the checkpoint，and save in the path: ISAT_with_segment_anything/segment_any
- H-checkpoint:[sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
    
    H checkpoint has best effect, but need more resources.VRAM needs at least 8G.
- L-checkpoint:[sam_vit_l_0b3195.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
    
    L checkpoint has normal effect and normal resources.VRAM needs at least 7G.
- B-checkpoint:[sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
    
    B checkpoint has pool effect, but need less resources.VRAM needs at least 6G.

### (5) Run
```shell
python main.py
```

# Explain
1. The software needs to run well under the GPU and enought VRAM.
2. If you don't have a GPU or don't have enought VARM, please manually draw polygons for labeling by [ISAT](https://github.com/yatengLG/ISAT).
3. 


# Citation
```text
@misc{ISAT with segment anything,
  title={{ISAT with segment anything}: Image segmentation annotation tool with segment anything},
  url={https://github.com/yatengLG/ISAT_with_segment_anything},
  note={Open source software available from https://github.com/yatengLG/ISAT_with_segment_anything},
  author={yatengLG and horffmanwang},
  year={2023},
}
```