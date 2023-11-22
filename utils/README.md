# Convert Scripts

- isat.py - ISAT format explain, and this also is the base class for other convert, contain functions: read_from_ISAT, save_to_ISAT.
- coco.py - COCO convert derived class. contain functions: read_from_COCO, save_to_COCO and all functions of ISAT.
- yolo.py - YOLO convert derived class. contain functions: read_from_YOLO, save_to_YOLO and all functions of ISAT.
- xml_detection.py - This is for object detection, contain functions: save_to_XML and all functions of ISAT.

# 转换脚本

- isat.py - ISAT格式说明，同时ISAT类也是其他转换类的基类。
- coco.py - COCO数据转换类
- yolo.py - YOLO数据转换类
- xml_detection.py - XML数据转换类，针对目标检测任务。将ISAT标注的多边形转换为目标检测包围盒。