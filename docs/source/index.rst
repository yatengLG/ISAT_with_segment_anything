Welcome to ISAT-SAM's documentation!
====================================

**ISAT-SAM** stands for Interactive Semi-Automatic Annotation Tool with `Segment Anything Model <https://github.com/facebookresearch/segment-anything>`_

.. image:: ../../display/标注.gif
   :alt: Software demo

| The annotations are stored in ISAT json format, similar to MSCOCO:


* **info**:
    * **description**: Always 'ISAT'
    * **folder**: The directory where the images are stored
    * **name**: The name of the image file
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


Check :doc:`features` for full features demonstration. Or click :doc:`usage` to **get started!**


.. note::

   This project is under active development. Feedbacks are Welcome!

Contents
====================================
.. toctree::
   features
   usage


