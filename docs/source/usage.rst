Installation
====================================

There are three ways to install ISAT-SAM:

1. from source code (recommended)
2. pip install
3. from .exe

------------------------------------------------------------------------

Option 1: From Source Code
------------------------------------------------------------------------

1. **Create environment**
   Use conda to set up a new environment:

   .. code-block:: bash

      conda create -n isat_env python=3.8
      conda activate isat_env

2. **Install ISAT_with_segment_anything and its dependencies**
   To use GPU, please install `Pytorch-GPU <https://pytorch.org/>`_ on Windows OS first.

   .. code-block:: bash

      git clone https://github.com/yatengLG/ISAT_with_segment_anything.git
      cd ISAT_with_segment_anything
      pip install -r requirements.txt

3. **Download Segment anything pretrained checkpoint**
   Download the checkpoint, and save it under: ``ISAT_with_segment_anything/ISAT/checkpoints``

   After version 0.0.3, you can manage checkpoints within the GUI, click [menubar]-[SAM]-[Model manage] to open the GUI.

   Now support `SAM <https://github.com/facebookresearch/segment-anything>`_, `Sam-HQ <https://github.com/SysCV/sam-hq>`_, `MobileSAM <https://github.com/ChaoningZhang/MobileSAM>`_, and `EdgeSAM <https://github.com/chongzhou96/EdgeSAM>`_ etc.

   .. list-table:: Pretrained Checkpoints
      :header-rows: 1

      * - Model
        - Pretrained Checkpoint
        - Memory
        - Size
      * - SAM
        - `sam_vit_h_4b8939.pth <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth>`_
        - 7305M
        - 2.6G
      * - 
        - `sam_vit_l_0b3195.pth <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth>`_
        - 5855M
        - 2.6G
      * - 
        - `sam_vit_b_01ec64.pth <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth>`_
        - 4149M
        - 375M
      * - sam-hq
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
      * - mobile-sam
        - `mobile_sam.pt <https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt>`_
        - 1375M
        - 40M
      * - edge-sam
        - `edge_sam.pth <https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam.pth>`_
        - 960M
        - 39M
      * - 
        - `edge_sam_3x.pth <https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_3x.pth>`_
        - 960M
        - 39M
      * - sam-med
        - `sam-med2d_b.pth <https://drive.google.com/file/d/1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl/view?usp=drive_link>`_
        - 1500M
        - 2.4G

4. **Run**
   Execute the main application:

   .. code-block:: bash

      python main.py

------------------------------------------------------------------------

Option 2: Pip Install
------------------------------------------------------------------------

1. **Create environment**
   Use conda to create and activate a new environment:

   .. code-block:: bash

      conda create -n isat_env python=3.8
      conda activate isat_env

2. **Install ISAT-SAM using pip**
   To use GPU, install `Pytorch-GPU <https://pytorch.org/>`_ on Windows OS first:

   .. code-block:: bash

      pip install isat-sam

3. **Run**
   Start the application via the command line:

   .. code-block:: bash

      isat-sam

------------------------------------------------------------------------

Option 3: Install with .exe
------------------------------------------------------------------------

1. **Download the .exe**
   The .exe version may be older than the source code version.

   - Download three .zip files, total 2.7G
   - Download link: `Baidu Netdisk <https://pan.baidu.com/s/1vD19PzvIT1QAJrAkSVFfhg>`_ Code: ISAT

   Click `main.exe` to run the tool.

2. **Download Segment anything pretrained checkpoint**
   The download zip files contain `sam_hq_vit_tiny.pth`, but note this model may not support CPU.
   You can download `mobile_sam.pt <https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt>`_ to test the tool.

   For using other models, refer to `Download Segment anything pretrained checkpoint <https://github.com/yatengLG/ISAT_with_segment_anything/blob/master/docs/README-en.md#3-download-segment-anything-pretrained-checkpoint>`_.



