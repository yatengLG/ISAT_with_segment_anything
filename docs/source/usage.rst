Installation
====================================

There are three ways to install ISAT-SAM:

1. (recommended) from source code 
2. pip install
3. (old version) from .exe
4. (new) local + GPU server

------------------------------------------------------------------------

Option 1: From Source Code
------------------------------------------------------------------------

1. **Create environment**


   Use `conda <https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html/>`_ to set up a new environment:

   .. code-block:: bash

      conda create -n isat_env python=3.10
      conda activate isat_env


2. **Install ISAT_with_segment_anything and its dependencies**


   To use GPU, please install `Pytorch-GPU <https://pytorch.org/>`_ on Windows OS first.

   **Note**: the next step requires 'git' in environment path. If not, you can download the code zip file `Here <https://github.com/yatengLG/ISAT_with_segment_anything/archive/refs/heads/master.zip>`_. 
   And you need to change the folder name 'ISAT_with_segment_anything-master' to 'ISAT_with_segment_anything'. 
   Doing this, you can skip the `git clone` step

   .. code-block:: bash

      git clone https://github.com/yatengLG/ISAT_with_segment_anything.git
      cd ISAT_with_segment_anything
      pip install -r requirements.txt


   **For macOS users**:

   It is important to follow the installation order to ensure SAM can be load on CPU

   .. code-block:: bash

      git clone https://github.com/yatengLG/ISAT_with_segment_anything.git
      cd ISAT_with_segment_anything
      pip timm imgviz scikit-image opencv-python pillow pyyaml pycocotools shapely hydra-core tqdm fuzzywuzzy python-Levenshtein iopath
      conda install conda-forge::pyqt 
      conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -c pytorch

3. **Download Segment anything pretrained checkpoint**


   | Download the model checkpoint with the GUI, click [menubar]-[SAM]-[Model manage] to open the GUI. 

   | Model checkpoints are stored under: ``ISAT_with_segment_anything/ISAT/checkpoints`` 



4. **Run**


   Execute the main application:

   .. code-block:: bash

      python main.py

------------------------------------------------------------------------

Option 2: Pip Install
------------------------------------------------------------------------

1. **Create environment**


   Use `conda <https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html/>`_ to set up a new environment:

   .. code-block:: bash

      conda create -n isat_env python=3.10
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
   - Click `main.exe` to run the tool.



2. **Download Segment anything pretrained checkpoint**


   The download zip files contain `sam_hq_vit_tiny.pth`, but note this model may not support CPU.
   You can download `mobile_sam.pt <https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt>`_ to test the tool.



------------------------------------------------------------------------

Option 4: GPU server + local GUI
------------------------------------------------------------------------

1. Ensure ensure these lines are enabled in ```/etc/ssh/sshd_config```

   .. code-block:: bash

      sudo nano /etc/ssh/sshd_config
      
      X11Forwarding yes
      X11DisplayOffset 10
      X11UseLocalhost yes

      sudo apt-get install xauth x11-xserver-utils
      sudo systemctl restart sshd



2. Install local X forwarding softwares

   - Windows: `MobaXterm <https://mobaxterm.mobatek.net/download.html>`_
   - macOS: `XQuartz <https://www.xquartz.org>`_

| 

3. Test the X forwarding

   .. code-block:: bash

      ssh -X user_name@host_name
      
      echo $DISPLAY
      xeyes

| You should be able to a running eyes annimation
| 
| 

4. Build and run the docker image

| The Dockerfile and entrypoint file are under ``/ISAT_with_segment_anything/docker/``.  
| Please change the user and mounted volume as you desire.
| 

.. code-block:: bash

    docker build --network=host -t isat .
    
    docker run -it \
      --user $(id -u):$(id -g) \
      --gpus all \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      -v $HOME/.Xauthority:$HOME/.Xauthority \
      -v ~/projects/ISAT_SAM:/ISAT_SAM \
      -e DISPLAY=$DISPLAY \
      --network host \
      isat \
      --rm