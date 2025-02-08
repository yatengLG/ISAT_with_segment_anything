Welcome to ISAT-SAM's documentation!
====================================

**ISAT-SAM** stands for Interactive Semi-Automatic Annotation Tool with `Segment Anything Model <https://github.com/facebookresearch/segment-anything>`_

.. image:: ../../display/标注.gif
   :alt: Software demo

|

Annotation JSON File Structure
====================================

| The annotations are stored in ISAT json format, similar to MSCOCO:
|

* **info**:
    * **description**: Always 'ISAT' for the software to recoginize
    * **folder**: The directory where the images are stored
    * **name**: The name (path) of the image file
    * **width**, **height**, **depth**: The dimensions of the image; depth is 3 for RGB images
    * **note**: An optional field for any additional notes related to the image

* **objects**:
    * **category**: The class label of the object. If the category_id from MSCOCO does not have a corresponding entry, 'unknown' is used
    * **group**: An identifier that groups objects based on overlapping bounding boxes. If an object's bounding box is within another, they share the same group number. Group numbering starts at 1
    * **segmentation**: A list of [x, y] coordinates forming the polygon around the object, e.g. [[x1, y1], [x2, y2], ..., [xn, yn]]
    * **area**: The area covered by the object in pixels
    * **layer**: A float indicating the sequence of the object. It increments within the same group, starting at 1.0
    * **bbox**: The bounding box coordinates in the format [x_min, y_min, x_max, y_max]
    * **iscrowd**: A boolean value indicating if the object is part of a crowd
    * **note**: An optional field for any additional notes related to the annotation mask



Supported Models
====================================

Now support `SAM2.1 <https://github.com/facebookresearch/segment-anything-2>`_, `SAM <https://github.com/facebookresearch/segment-anything>`_, `Sam-HQ <https://github.com/SysCV/sam-hq>`_, 
`MedSAM <https://github.com/bowang-lab/MedSAM>`_, `MobileSAM <https://github.com/ChaoningZhang/MobileSAM>`_, and `EdgeSAM <https://github.com/chongzhou96/EdgeSAM>`_.


.. list-table:: Pretrained Checkpoints
   :header-rows: 1

   * - Model
     - Pretrained Checkpoint
     - Memory
     - Size
   
   * - SAM-HQ
     - `sam_hq_vit_h.pth <https://huggingface.co/lkeab/hq-sam/blob/main/sam_hq_vit_h.pth>`_
     - 7393M
     - 2.6G
   * - 
     - `sam_hq_vit_l.pth <https://huggingface.co/lkeab/hq-sam/blob/main/sam_hq_vit_l.pth>`_
     - 5939M
     - 1.3G
   * - 
     - `sam_hq_vit_b.pth <https://huggingface.co/lkeab/hq-sam/blob/main/sam_hq_vit_b.pth>`_
     - 4207M
     - 379M
   * - 
     - `sam_hq_vit_tiny.pth <https://huggingface.co/lkeab/hq-sam/blob/main/sam_hq_vit_tiny.pth>`_
     - 1463M
     - 43M

   * - SAM2.1
     - `sam2.1_hiera_large.pt <https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt>`_
     - 4000M
     - 900M
   * - 
     - `sam2.1_hiera_base_plus.pt <https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt>`_
     - 4000M
     - 324M
   * - 
     - `sam2.1_hiera_small.pt <https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt>`_
     - 3000M
     - 185M
   * - 
     - `sam2.1_hiera_tiny.pt <https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt>`_
     - 2400M
     - 156M

   * - MedSAM
     - `sam-med2d_b.pth <https://drive.google.com/file/d/1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl/view?usp=drive_link>`_
     - 1500M
     - 2.4G

   * - SAM
     - `sam_vit_h_4b8939.pth <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth>`_
     - 7305M
     - 2.6G
   * - 
     - `sam_vit_l_0b3195.pth <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth>`_
     - 5855M
     - 1.3G
   * - 
     - `sam_vit_b_01ec64.pth <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth>`_
     - 4149M
     - 375M

   * - Edge-SAM
     - `edge_sam.pth <https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam.pth>`_
     - 960M
     - 39M
   * - 
     - `edge_sam_3x.pth <https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_3x.pth>`_
     - 960M
     - 39M

   * - Mobile-SAM
     - `mobile_sam.pt <https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt>`_
     - 1375M
     - 40M

   * - SAM2
     - `sam2_hiera_large.pt <https://huggingface.co/yatengLG/ISAT_with_segment_anything_checkpoints/resolve/main/sam2_hiera_large.pt>`_
     - 4000M
     - 900M
   * - 
     - `sam2_hiera_base_plus.pt <https://huggingface.co/yatengLG/ISAT_with_segment_anything_checkpoints/resolve/main/sam2_hiera_base_plus.pt>`_
     - 4000M
     - 324M
   * - 
     - `sam2_hiera_small.pt <https://huggingface.co/yatengLG/ISAT_with_segment_anything_checkpoints/resolve/main/sam2_hiera_small.pt>`_
     - 3000M
     - 185M
   * - 
     - `sam2_hiera_tiny.pt <https://huggingface.co/yatengLG/ISAT_with_segment_anything_checkpoints/resolve/main/sam2_hiera_tiny.pt>`_
     - 2400M
     - 156M



.. note::
   Check :doc:`usage` to get started! Or :doc:`features` for full features demonstration.

   This project is under active development. Feedbacks are Welcome!

Contents
====================================
.. toctree::
   usage
   features


