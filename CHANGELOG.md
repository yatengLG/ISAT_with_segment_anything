# 更新日志

## [0.0.1]

* 发布测试版，版本号0.0.1

## 
* 更新下载功能（后续需优化下载链接）
* 支持多选多边形，现在可以批量删除了

    按下Ctrl点击多边形；按下Ctrl点击右侧标注栏；按下Shift点击右侧标注栏；点击右侧标注栏，Ctrl+A全选。
    
* 添加转换脚本 - utils目录下
    
    现支持 ISAT <-> COCO, ISAT <-> YOLO, ISAT -> XML(目标检测) 

* 添加了对[segment-anything-fast](https://github.com/pytorch-labs/segment-anything-fast)的支持
    
    **现支持SAM系列模型，sam-hq以及mobile-sam等后续更新**
    
    sam-fast要求torch版本不低于2.1.1;低于版本要求时，默认导入sam
    
    sam_vit_h_4b8939.pth encode计算时间缩短大概4倍，显存占用缩小到5.6G
    
| | sam | sam-fast |
|----|----|----|
| 0 | 0.698307991027832 | 0.19336390495300293 | 
| 1 | 0.7048919200897217 | 0.21175742149353027 | 
| 2 | 0.766636848449707 | 0.2573261260986328 | 
| 3 | 0.8198366165161133 | 0.22284531593322754 | 

* 添加了饱和度调整
    
    通过拖动工具栏中的饱和度调整条，对图片进行饱和度调整。（只与显示有关，不影响sam）
    
* 添加track模式
    
    在"自动auto"和"手动manual"模式外，为组模式中添加了"跟踪track"模式。该模式下，使用[TAB]或者[`]切换目标时，组id会显示为设置为当前多边形的组id。
    
## [0.0.2]

* 添加了模型管理界面

    现在可以方便的管理与使用模型了。
    
    **由于sam-hq以及mobile-sam的权重链接，需要科学上网才可以访问。这两类模型下载时会经常失败**
    **有推荐的比较好用的大文件托管服务，可以联系我**

* 整合了数据转换界面并提供了新功能
    
    - COCO <-> ISAT
    - YOLO <-> ISAT
    - LABELME <-> ISAT
    - ISAT -> VOC(png单通道图)
    - ISAT -> VOC for object detection(xml目标检测)

* 添加了linux系统对[**segment-anything-fast**](https://github.com/pytorch-labs/segment-anything-fast)的支持
    
    该功能可以保持sam分割效果的情况下，减少显存占用并提升分割速度。（目前只对sam系列模型有效）
    
    由于windows下需需torch版本为2.2.0+dev且需要安装较多其他依赖，因而暂时关闭了windows系统下对该功能的支持。
    

* 修复了遗留bug
    
    - 修复了转VOC后第一行一直为0的问题
    - 轮廓保存模式中-只保存最大轮廓现在严格保存面积最大的轮廓(之前使用顶点数量进行粗估计)