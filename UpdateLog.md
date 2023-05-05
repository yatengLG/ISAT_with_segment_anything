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