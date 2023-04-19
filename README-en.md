# ISAT with segment anything
# Image segmentation annotation tool with segment anything.

Quick annotate image by [segment anything](https://github.com/facebookresearch/segment-anything).

[中文](README.md)         [English](README-en.md)



## INSTALL
### 1. 源码运行
```shell
# 创建虚拟环境
conda create -n ISAT_with_segment_anything python==3.8
conda activate ISAT_with_segment_anything
```

```shell
# Install Segment anything
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
cd ..
```

```shell
# Install ISAT_with_segment_anything
git clone https://github.com/yatengLG/ISAT_with_segment_anything.git
cd ISAT_with_segment_anything
pip install -r requirements.txt
```

```text
# Download Segment anything pretrained checkpoint.
Download the checkpoint，and save in the path: ISAT_with_segment_anything/segment_any
checkpoint link：
   https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
   https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
The checkpoint named sam_vit_h_4b8939 has best effect, but need more resources；
The checkpoint named sam_vit_b_01ec64 has poor effect, but need less resources；
```

```shell
# Run the software
python main.py
```