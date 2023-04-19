# ISAT with segment anything
# ISAT 图像分割标注工具(集成segment anything)

集成[segment anything]()，实现图片分割快速标注。

## 特点
1. segment anything简化标注过程

    通过鼠标左右键提示感兴趣区域，调用segment anything自动计算分割掩码。不必再手动进行目标轮廓选取。

2. 便捷地修改标注结果

    segment anything只是协助生成目标掩码，最终标注以多边形呈现，通过拖拽多边形顶点，可快速修改标注区域。

## 安装
### 1. 源码运行
```shell
# 创建虚拟环境
conda create -n ISAT_with_segment_anything python==3.8
conda activate ISAT_with_segment_anything
```

```shell
# 安装Segment anything
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
cd ..
```

```shell
# 安装ISAT_with_segment_anything
git clone https://github.com/yatengLG/ISAT_with_segment_anything.git
cd ISAT_with_segment_anything
pip install -r requirements.txt
```

```text
# 下载Segment anything预训练模型
下载任一模型，并将模型存放于ISAT_with_segment_anything/segment_any目录下
模型链接：
   https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
   https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
h模型最大，效果也最好；
b模型最小，效果也最差；
请按照硬件下载合适的模型，h模型在示例样本上的显存需求约8G.
```

```shell
# 运行软件
python main.py
```


## 标注操作

1. 通过鼠标左键（或右键）提示感兴趣区域（或不感兴趣区域），自动形成目标分割掩码
2. 可通过多次左右键提示，提升掩码质量
3. E键结束标注，选择类别，得到多边形标注区域
4. 拖拽多边形顶点，精细化调整标注

## 注意事项
1. 自动分割效果受segment anything模型分割效果限制，如需更为精确的分割效果，可通过手动绘制多边形实现。
2. 如只需要使用手动绘制多边形标注，推荐使用[ISAT](https://github.com/yatengLG/ISAT)。
