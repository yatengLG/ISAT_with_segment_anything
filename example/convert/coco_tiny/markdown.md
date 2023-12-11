# COCO tiny - used to test convert scripts

random choices five images from coco/instances_val2017, contain polygon and RLE(run-length encoding, be used to label crowed objects).

- images - coco tiny images
- coco_tiny.json - coco json file.
- yolo - yolo annotation files convert from coco.json by [yolov8 script](https://github.com/ultralytics/JSON2YOLO/blob/c38a43f342428849c75c103c6d060012a83b5392/general_json2yolo.py)
- isat - annotation files labeled by isat.

# 微小COCO数据

从COCO val2017中随机选取五个样本，样本中包含了多边形标注与rle编码标注。
在coco分割数据集中，rle编码用来标注拥挤目标。

- images - coco tiny图片存放文件夹
- coco_tiny.json - coco注释文件。摘自coco/annotations/instances_val2017.json
- yolo - yolo注释文件。通过[yolov8 提供的coco转换yolo脚本](https://github.com/ultralytics/JSON2YOLO/blob/c38a43f342428849c75c103c6d060012a83b5392/general_json2yolo.py)转换coco_tiny.json得到。
- isat - isat注释文件。由ISAT标注得到。
