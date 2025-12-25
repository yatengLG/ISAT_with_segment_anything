Installation
====================================

Option 1: Install with Pip
------------------------------------

1.  Create a new environment (recommended, optional)

    Use conda create a new environment

    .. code-block:: bash

       conda create -n isat_env python=3.10

    activate new environment

    .. code-block:: bash

        conda activate isat_env

2.  Install ISAT-SAM using pip

    .. code-block:: bash

        pip install isat-sam

    .. attention::

        To use GPU on Windows OS, install torch and torchvision from `Pytorch <https://pytorch.org/>`_

3.  Run

    Start the application via the command line:

    .. code-block:: bash

        isat-sam

.. hint:: Recommended install isat-sam using pip

Option 2: Install from Source Code
------------------------------------

1.  Get source code from github

    Use git clone project and then into project

    .. code-block:: bash

        git clone https://github.com/yatengLG/ISAT_with_segment_anything.git
        cd ISAT_with_segment_anything

    Or download zip from `Github <https://github.com/yatengLG/ISAT_with_segment_anything/archive/refs/heads/master.zip>`_ , unzip and into project

    .. code-block:: bash

        cd ISAT_with_segment_anything-master

2.  Create a new environment (recommended, optional)

    Use conda create a new environment

    .. code-block:: bash

        conda create -n isat_env python=3.10

    activate new environment

    .. code-block:: bash

        conda activate isat_env

3.  Install dependencies

    .. code-block:: bash

        pip install -r requirements.txt

    .. attention::

        To use GPU on Windows OS, install torch and torchvision from `Pytorch <https://pytorch.org/>`_

4.  Run

    Start the application via the command line:

    .. code-block:: bash

        python main.py

    Or

    .. code-block:: bash

        # install isat as a package
        python setup.py install
        # run
        isat-sam

