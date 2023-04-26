# ISAT with segment anything
## Image segmentation annotation tool with segment anything.
![annotate.gif](./display/标注%20-big-original.gif)

Quick annotate for image segmentation by [segment anything](https://github.com/facebookresearch/segment-anything)

**Project updating，suggestions welcome**

Demo Video：[youtube](https://www.youtube.com/watch?v=yLdZCPmX-Bc)

演示视频：[bilibili](https://www.bilibili.com/video/BV1Lk4y1J7uB/)

[中文](README.md)         [English](README-en.md)

# Feature

1. Support semantic segmentation and instance segmentation
2. Get the mask by SAM (segment anything model) and convert the mask into polygon.
3. Interactive correction mask, by clicking on the area of interest (not of interest) with the left (right) mouse button.
4. Support manual creation of polygons.
5. Support modifying polygons.
6. Support for adjusting polygon occlusion.
7. Support opening JSON files annotated by labelme (please backup a copy before opening).
8. Support exporting annotations as single-channel png images.

# INSTALL
## 1. Run the source code
### (1) Create environment
```shell
conda create -n ISAT_with_segment_anything python==3.8
conda activate ISAT_with_segment_anything
```

### (2) Install Segment anything
```shell
git clone git@github.com:facebookresearch/segment-anything.git
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
