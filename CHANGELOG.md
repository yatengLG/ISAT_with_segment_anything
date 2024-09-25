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

## 
* 添加了对[EdgeSAM](https://github.com/chongzhou96/EdgeSAM)的支持。
* 修复转coco后，类别ID从0开始的问题。（现在第一类的类别id为1）
* 修复sam标注过程中，切换图片文件夹后，闪退的问题
* 添加了模型国内下载链接

* 减少模型显存占用
    
    使用bfloat16模型后，显存需求降低，特征计算时间略微增加，最终分割效果无显著变化。

| checkpoint | mem(float) | mem(bfloat16) | cost(float)| cost(bfloat16) |
|----:|----:|----:|----:|----:|
| edge_sam.pth          | 360M | 304M | 0.0212 | 0.0239 |
| edge_sam_3x.pth       | 360M | 304M | 0.0212 | 0.0239 |
| mobile_sam.pt         | 534M | 390M | 0.0200 | 0.0206 |
| sam_hq_vit_tiny.pth   | 598M | 392M | 0.0196 | 0.0210 |
| sam_hq_vit_b.pth      | 3304M | 1762M | 0.1496 | 0.1676 |
| sam_hq_vit_l.pth      | 5016M | 2634M | 0.3766 | 0.4854 |
| sam_hq_vit_h.pth      | 6464M | 3378M | 0.6764 | 0.9282 |
| sam_vit_b_01ec64.pth  | 3302M | 1760M | 0.1539 | 0.1696 |
| sam_vit_l_0b3195.pth  | 5016M | 2634M | 0.3776 | 0.4833 |
| sam_vit_h_4b8939.pth  | 6462M | 3378M | 0.6863 | 0.9288 |

## 0.0.3

* 更新了项目结构
* 填加了sam_med2d模型，实现对医疗数据更好的分割效果

## 0.0.4

* 更新了基于bounding box的自动分割功能

基于标注完成的VOC格式目标检测数据，使用标注框进行sam框提示，自动分割图像并存储为ISAT格式json。


## 1.1.0

- 添加了isat 标注检查器，用于检测存在问题的标注文件。
  
  [Alias-z](https://github.com/Alias-z)在使用[sahi](https://github.com/obss/sahi)过程中发现了该问题
  
- 添加了对多边形的交集、并集、差集、异或计算。

  感谢[XieDeWu](https://github.com/XieDeWu)提的建议。

- 添加了自动保存模式
  
  开启后，切换图片自动保存标注。  

- 功能修复
  
  模型下载，点击按钮后，进度现在更直观了。

## 1.1.1
- 添加了sam2模型的图像分割功能
- 添加拖拽打开文件功能

## 1.2.0
- 添加了对sam2图像分割的支持

## 1.2.1
- 修复pip安装时的依赖项
