Features
====================================

Annotation Modes
------------------------------------------------------------------------
Semi-automatic Annotation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Utilizes SAM with point and bounding box prompts.

Manual Annotation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Click or drag to draw polygons (0.15s per point).

------------------------------------------------------------------------

Annotation Adjustments
------------------------------------------------------------------------
Polygon Adjustments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Delete points and adjust object occlusions to refine the annotation.

Polygon Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Preview groups and semantic/instance segmentation masks.

------------------------------------------------------------------------

Export Annotations
------------------------------------------------------------------------
Supported Formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Export annotations in multiple formats including MSCOCO, YOLO, LabelMe, and VOC (also XML).

For more features, see the below

------------------------------------------------------------------------


Language switching
------------------------------------------------------------------------
The software provides two interfaces, Chinese and English, which can be switched at any time.

.. image:: ../../display/双语界面.gif
   :alt: Bilingual Interface

------------------------------------------------------------------------

Model switching
------------------------------------------------------------------------
Switch model among your downloaded models.

.. image:: ../../display/模型切换.gif
   :alt: Model switching

------------------------------------------------------------------------

Contour mode
------------------------------------------------------------------------
When using SAM for semi-automatic annotation, convert SAM mask to contours using OpenCV, then convert contours to polygons.

1. **Max only**: Usually, the contour with the highest number of vertices also has the largest area. Other contours are deleted as noise.
2. **External**: Saves external contours; internal holes in masks will be filled.
3. **All**: Saves all contours; will add polygons with category __background__ for holes.

.. image:: ../../display/轮廓保存模式.gif
   :alt: Contour saving modes

------------------------------------------------------------------------

Occlusion adjustment
------------------------------------------------------------------------
Adjust occlusion for polygons with overlapping parts, choosing either **to top** or **to bottom** adjustment.

.. image:: ../../display/图层调整遮挡关系.gif
   :alt: Adjusting layer occlusion

------------------------------------------------------------------------

Quick jump to image
------------------------------------------------------------------------
Input an image name or index to quickly navigate to it.

.. image:: ../../display/图片快速跳转.gif
   :alt: Quick image jump

------------------------------------------------------------------------

Statusbar information
------------------------------------------------------------------------
Obtain position and pixel value information. If using SAM, CUDA memory usage is also displayed on the status bar.

.. image:: ../../display/状态栏信息.gif
   :alt: Statusbar Information

------------------------------------------------------------------------

Preview annotation result
------------------------------------------------------------------------
Click the **Bit map** button to preview semantic and instance annotation results.

.. image:: ../../display/实时预览.gif
   :alt: Real-time preview

------------------------------------------------------------------------

Import/export config file
------------------------------------------------------------------------
Save categories and other parameters using a configuration file.

.. image:: ../../display/配置文件导入导出.gif
   :alt: Config file import/export

------------------------------------------------------------------------

Drag to draw
------------------------------------------------------------------------
Create polygons by keeping the mouse left button pressed and dragging. Use a click for straight lines and dragging for curves.

.. image:: ../../display/拖动绘制.gif
   :alt: Drag to draw

------------------------------------------------------------------------

Quick browsing
------------------------------------------------------------------------
Select a group ID from a dropdown to view the target, supporting quick switching between different groups via the scroll wheel.

.. image:: ../../display/快速浏览.gif
   :alt: Quick browsing

------------------------------------------------------------------------

Detail inspection
------------------------------------------------------------------------
Switch annotations one by one using group IDs, adapting the view to the size of the annotation for detailed inspection.

.. image:: ../../display/细节检查.gif
   :alt: Detail inspection

------------------------------------------------------------------------

Move and delete vertices
------------------------------------------------------------------------
Select vertices using CTRL and move or delete them.

.. image:: ../../display/顶点批量移动与删除.gif
   :alt: Move and delete vertices

------------------------------------------------------------------------

SAM features cache
------------------------------------------------------------------------
Features are encoded using QThread to speed up image switching when using large models. Automatically encodes features for the current, previous, and next images.

.. image:: ../../display/sam缓存.gif
   :alt: SAM features cache

------------------------------------------------------------------------

Model manager
------------------------------------------------------------------------
Supports multithreaded downloading with options to pause and resume transfers.

.. image:: ../../display/模型下载.gif
   :alt: Model manager

------------------------------------------------------------------------

Repaint
------------------------------------------------------------------------
Use the shortcut 'R' to switch to repaint mode. Start by selecting one vertex and end by selecting another vertex.

.. image:: ../../display/重绘.gif
   :alt: Repaint

------------------------------------------------------------------------

Intersection, Union, Difference, and XOR
------------------------------------------------------------------------

Provides operations for the intersection, union, difference, and XOR of two polygons.

Intersection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Calculate and display the intersection of two polygons.

.. image:: ../../display/交集.gif
   :alt: Intersection

------------------------------------------------------------------------

Union
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Calculate and display the union of two polygons.

.. image:: ../../display/并集.gif
   :alt: Union

------------------------------------------------------------------------

Difference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Calculate and display the difference between two polygons.

.. image:: ../../display/差集.gif
   :alt: Difference

------------------------------------------------------------------------

XOR (Symmetric Difference)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Calculate and display the symmetric difference (XOR) between two polygons.

.. image:: ../../display/异或.gif
   :alt: XOR


