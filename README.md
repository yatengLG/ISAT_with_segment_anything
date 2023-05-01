# ISAT with segment anything
## ISAT 图像分割标注工具(集成segment anything)

![标注.gif](./display/标注%20-big-original.gif)

集成[segment anything](https://github.com/facebookresearch/segment-anything)，实现图片分割快速标注。

**项目持续更新中，[更新日志](./UpdateLog.md)，欢迎大家提出建议**

演示视频：[bilibili](https://www.bilibili.com/video/BV1Lk4y1J7uB/)

Demo Video：[youtube](https://www.youtube.com/watch?v=yLdZCPmX-Bc)

[中文](README.md)         [English](README-en.md)

# 特点

1. 支持同时标注语义分割与实例分割
2. 基于SAM(segment anything model)快速获取目标mask(掩码)，并将mask转换为多边形。
3. 交互式修正mask，通过鼠标左(右)键点击感兴趣(不感兴趣)区域，指引模型修正mask。
4. 支持手动创建多边形。
5. 支持修改多边形。
6. 支持调整多边形遮挡关系。
7. 支持打开labelme标注的json文件（打开前请先备份一份）。
8. 支持导出标注为单通道png图片。

# 安装
## 1. 源码运行
### (1) 创建虚拟环境
```shell
conda create -n ISAT_with_segment_anything python==3.8
conda activate ISAT_with_segment_anything
```
### (2) 安装Segment anything
```shell
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
cd ..
```
### (3) 安装ISAT_with_segment_anything
```shell
git clone https://github.com/yatengLG/ISAT_with_segment_anything.git
cd ISAT_with_segment_anything
pip install -r requirements.txt
```
### (4) 下载Segment anything预训练模型
下载任一模型，并将模型存放于ISAT_with_segment_anything/segment_any目录下
请按照硬件下载合适的模型.

- H模型:[sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
    
    模型最大，效果也最好，显存至少需求8G，演示时软件实际占用7305M；
- L模型:[sam_vit_l_0b3195.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
    
    模型适中，效果也适中，显存至少需求8G，演示时软件实际占用5855M；
- B模型:[sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
    
    模型最小，效果也最差，显存至少需求6G，演示时软件实际占用4149M；

### (5) 运行软件
```shell
python main.py
```


# 标注操作

1. 通过鼠标左键（或右键）提示感兴趣区域（或不感兴趣区域），自动形成目标分割掩码。
2. 可通过多次左右键提示，提升掩码质量。
3. E键结束标注，选择类别，得到多边形标注区域。
4. 拖拽多边形顶点，精细化调整标注。

# 注意事项
1. 自动分割效果受segment anything模型分割效果限制，如需更为精确的分割效果，可通过手动绘制多边形实现。
2. 如果没有GPU或只需要使用手动绘制多边形标注，推荐使用[ISAT](https://github.com/yatengLG/ISAT)。
3. 软件对GPU显存有最低限制：
    - h模型最大，效果也最好，显存至少需求8G，演示时软件实际占用7305M；
    - l模型适中，效果也适中，显存至少需求8G，演示时软件实际占用5855M；
    - b模型最小，效果也最差，显存至少需求6G，演示时软件实际占用4149M；

# 引用
```text
@misc{ISAT with segment anything,
  title={{ISAT with segment anything}: Image segmentation annotation tool with segment anything},
  url={https://github.com/yatengLG/ISAT_with_segment_anything},
  note={Open source software available from https://github.com/yatengLG/ISAT_with_segment_anything},
  author={yatengLG and horffmanwang},
  year={2023},
}
```