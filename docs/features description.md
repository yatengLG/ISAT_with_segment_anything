# Menu

1. [Language switching](https://github.com/yatengLG/ISAT_with_segment_anything/blob/master/docs/features%20description.md#1language-switching)
2. [Model switching](https://github.com/yatengLG/ISAT_with_segment_anything/blob/master/docs/features%20description.md#2model-switching)
3. [Contour mode](https://github.com/yatengLG/ISAT_with_segment_anything/blob/master/docs/features%20description.md#3contour-mode)
4. [Occlusion adjustment](https://github.com/yatengLG/ISAT_with_segment_anything/blob/master/docs/features%20description.md#4occlusion-adjustment)
5. [Quick jump to image](https://github.com/yatengLG/ISAT_with_segment_anything/blob/master/docs/features%20description.md#5quick-jump-to-image)
6. [Statusbar information](https://github.com/yatengLG/ISAT_with_segment_anything/blob/master/docs/features%20description.md#6statusbar-information)
7. [Preview annotation result](https://github.com/yatengLG/ISAT_with_segment_anything/blob/master/docs/features%20description.md#7preview-annotation-result)
8. [Import/export config file](https://github.com/yatengLG/ISAT_with_segment_anything/blob/master/docs/features%20description.md#8importexport-config-file)
9. [Drag to draw](https://github.com/yatengLG/ISAT_with_segment_anything/blob/master/docs/features%20description.md#9drag-to-draw)
10. [Quick browsing](https://github.com/yatengLG/ISAT_with_segment_anything/blob/master/docs/features%20description.md#10quick-browsing)
11. [Detail inspection](https://github.com/yatengLG/ISAT_with_segment_anything/blob/master/docs/features%20description.md#11detail-inspection)
12. [Move and delete vertexs](https://github.com/yatengLG/ISAT_with_segment_anything/blob/master/docs/features%20description.md#12move-and-delete-vertexs)
13. [Sam features cache](https://github.com/yatengLG/ISAT_with_segment_anything/blob/master/docs/features%20description.md#13Sam-features-cache)
14. [Model manager](https://github.com/yatengLG/ISAT_with_segment_anything/blob/master/docs/features%20description.md#14Model-manager)

# 1.Language switching
The software provides two interfaces, Chinese and English, which can be switched at any time.

![双语界面.gif](../display/双语界面.gif)

# 2.Model switching
Switching model in your download models.

![模型切换.gif](../display/模型切换.gif)

# 3.Contour mode
When using SAM for semi-automatic annotation,convert SAM mask to contours by opencv, and then convert contours to polygons.

1. Max only.
```text
Usually,the contour with the highest number of vertices also has the largest area.
Other contours will delete as noise.
```
2. External.
```text
Save external contours, the hole of masks will be filled.
```
3. All.
```text
Save all contours, will add polygons with category __background__ for holes.
```
![轮廓保存模式.gif](../display/轮廓保存模式.gif)


# 4.Occlusion adjustment
For polygons with overlapping parts,adjustment occlusion with **to top** or  **to bottom**.

![图层调整遮挡关系.gif](../display/图层调整遮挡关系.gif)


# 5.Quick jump to image
Input image name or index to jump the image.

![图片快速跳转.gif](../display/图片快速跳转.gif)

# 6.Statusbar information
Get position and value of pixel. 
If use SAM also show cuda memory in statusbar.

![状态栏信息.gif](../display/状态栏信息.gif)

# 7.Preview annotation result
Click **Bit map** button to preview semantic and instance annotation result.

![实时预览.gif](../display/实时预览.gif)

# 8.Import/export config file
Use config file save categories and other args.

![配置文件导入导出.gif](../display/配置文件导入导出.gif)

# 9.Drag to draw
Keep mouse left button and drag, create polygon like drawing.
You can use click to draw straight line, and use drag to draw curve.

![拖动绘制.gif](../display/拖动绘制.gif)

# 10.Quick browsing

Drop down and select group ID to view the target, also support quickly switch between different groups through the scroll wheel.

![快速浏览.gif](../display/快速浏览.gif)

# 11.Detail inspection
Switch Annotations one by one with group IDs, and at the same time, adapt to the annotation size for easy inspection details.

![细节检查.gif](../display/细节检查.gif)

# 12.Move and delete vertexs
Select vertexs through CTRL, and then move or delete them.

![顶点批量移动与删除.gif](../display/顶点批量移动与删除.gif)

# 13.Sam features cache

Now sam encode features by qthread, so switching images faster when use large model.

Will auto encode features for current image and the prior and the next.(You can adjust it in widgets/mainwindow.py > SegAnyThread > run() )

Add encoding state before file list.
- yello: encoding features
- green: features encoded
- gray:  no features

![sam缓存.gif](../display/sam缓存.gif)

**The features of image will not be encoded, when the speed switch images faster than the speed of sam encoding. It`s easy to solve, click the image or switch image for sam encoding** 

# 14.Model manager

![模型下载.gif](../display/模型下载.gif)
- support multithread download.
- support pause and breakpoint transmission.
