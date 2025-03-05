# -*- coding: utf-8 -*-
# @Author  : LG

from pycocotools import coco as mscoco
import os
import matplotlib.pyplot as plt
import PIL.Image as Image

coco_json = ''
coco_image_root = ''

coco = mscoco.COCO(coco_json)

imgids = coco.getImgIds()
for imgId in imgids:
    img = coco.loadImgs([imgId])[0]
    annids = coco.getAnnIds(imgIds=img['id'])
    anns = coco.loadAnns(annids)

    img_path = os.path.join(coco_image_root, img["file_name"])

    plt.imshow(Image.open(img_path))
    coco.showAnns(anns)
    plt.show()

