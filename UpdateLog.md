# 

- 添加了对labelme格式json的支持（只支持标注的多边形）

    **在进行修改之前，先备份一份！！！**
    现在可以打开并编辑之前用labeme生成的标注文件了，记得通过图层高低调整遮挡关系。
    但最终保存还会以ISAT格式的json保存。

- 添加了显示/隐藏按钮（快捷键V），用于显示或隐藏所有多边形
- 添加了GPU显存占用

# 

- 标注时隐藏所有多边形
- 修改windows中文路径下，图片打开的bug
- bug修复

# 
- 添加转换voc格式png图片的功能（单通道png）
- 添加转换coco格式json的功能（ISAT jsons to COCO json）
- 添加转换coco格式json转ISAT格式json的功能（COCO json to ISAT jsons）

# 
- 优化了转换界面，以显示详细的转换进度

#
- 添加了ISAT格式json转LabelMe格式json的功能
- 优化部分界面

# 新版本2.0

1. 更新了界面，现在左侧选择类别，试用SAM时，直接快捷键Q，鼠标提示，E完成标注即可，不再选择类别与组。
2. 菜单栏添加了mask转polygon的方式选择，分为a.保存所有外轮廓(单个多边形)，b.保存顶点数最多的外轮廓(单个多边形)，c.保存所有轮廓（内轮廓多边形类别默认为__background__）
3. 菜单栏添加了模型选择，列出segment_any文件夹下的所有.pth权重文件。
4. 支持了SAM-HQ