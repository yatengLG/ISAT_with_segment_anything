Plugin Development
====================================

1. Plugin Base Class
------------------------------------

.. code-block:: python
   :caption: ISAT.widgets.plugin_base.py
   :emphasize-lines: 6, 14, 23, 32, 40, 48

    class PluginBase(ABC):
        def __init__(self):
            self.enabled = False

        @abstractmethod
        def init_plugin(self, mainwindow):
            """
            接受主程序调用，进行初始化
            :param mainwindow: 传入mainwindow，便于访问ISAT主程序的各种方法与属性
            """
            pass

        @abstractmethod
        def enable_plugin(self):
            """
            启动插件
            :param mainwindow:
            :return:
            """
            pass

        @abstractmethod
        def disable_plugin(self):
            """
            禁用插件
            :param mainwindow:
            :return:
            """
            pass

        @abstractmethod
        def get_plugin_author(self) -> str:
            """
            获取插件作者
            :return:
            """
            pass

        @abstractmethod
        def get_plugin_version(self) -> str:
            """
            获取插件版本
            :return:
            """
            pass

        @abstractmethod
        def get_plugin_description(self) -> str:
            """
            获取插件描述
            :return:
            """
            pass

        def get_plugin_name(self) -> str:
            """
            获取插件名
            :return:
            """
            return self.__class__.__name__

        def activate_state_changed(self, checkbox_state):
            if checkbox_state:
                self.enable_plugin()
            else:
                self.disable_plugin()

        def before_image_open_event(self, image_path: str):
            """
            当图片打开前调用
            :param image_path: 图片路径
            :return:
            """
            pass

        def after_image_open_event(self):
            """
            当图片打开后调用
            :param:
            :return:
            """
            pass

        def before_annotation_start_event(self):
            """
            当开始标注前调用
            :return:
            """
            pass

        def after_annotation_created_event(self):
            """
            当标注创建完成后调用
            :return:
            """
            pass

        def after_annotation_changed_event(self):
            """
            当标注发生变化后调用，包括创建新标注，拖动顶点，移动多边形，改变多边形属性等
            :return:
            """
            pass

        def before_annotations_save_event(self):
            """
            当保存标注文件前调用
            :return:
            """
            pass

        def after_annotations_saved_event(self):
            """
            当保存标注文件后调用
            :return:
            """
            pass

        def after_sam_encode_finished_event(self, index):
            """
            当sam编码完成后调用
            注意ISAT在单独线程对前后图像进行预编码，通过self.mainwindow.seganythread.results_dict中获取编码结果
            :param index: 编码完成的图片index
            :return:
            """
            pass


        def on_mouse_move_event(self, scene_pos):
            """
            当鼠标移动时调用
            :param scene_pos: 坐标，限制在图像范围内
            :return:
            """
            pass

        def on_mouse_press_event(self, scene_pos):
            """
            当鼠标按下时调用
            :param scene_pos: 坐标，限制在图像范围内
            :return:
            """
            pass

        def on_mouse_release_event(self, scene_pos):
            """
            当鼠标释放时调用
            :param scene_pos: 坐标，限制在图像范围内
            :return:
            """
            pass

        def on_mouse_pressed_and_mouse_move_event(self, scene_pos):
            """
            当鼠标拖动时调用
            :param scene_pos: 坐标，限制在图像范围内
            :return:
            """
            pass

        def application_start_event(self):
            """
            软件启动后触发
            :return:
            """
            pass

        def application_shutdown_event(self):
            """
            软件关闭前触发
            :return:
            """
            pass

.. tip:: PluginBase is the base class for all ISAT plugin classes

2. Plugin Entry Point
------------------------------------

.. code-block:: python
   :caption: setup.py

    ...
    entry_points={
        "isat.plugins": []
    }
    ...

.. important:: All ISAT plugins are released in the form of Python packages.

               **Plugin must use setup.py to add the plugin classes to the isat.plugins entry_points.**

3. Plugin Get ISAT Data
------------------------------------

The data returned from the events of the plugin base class is limited.

Generally, when developing ISAT plugins, you can get ISAT data through ``self.mainwindow`` .

4. Create Your First Plugin
------------------------------------

4.1 Plugin Project Structure
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

It is recommended to use the following structure as the plugin project.

::

    ProjectName
    ├── MANIFEST.in
    ├── requirements.txt
    ├── README.md
    ├── setup.py
    ├── ...
    └── PluginPackage
       ├── __init__.py
       ├── main.py
       └── ...

4.2 Write Plugin
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

-   ``ProjectName``/``PluginPackage``/``__init__.py``

    Including information such as version, author, and package description.

    .. code-block:: python

        __author__ = "Your name"
        __version__ = "0.0.1"
        __description__ = "A short description of the plugin's functionality."

-   ``ProjectName``/``PluginPackage``/``main.py``

    Implement the plugin class.

    .. code-block:: python

        from ISAT.widgets.plugin_base import PluginBase

        class CustomPlugin(PluginBase):
            def __init__(self):
                super().__init__()

            def init_plugin(self, mainwindow):
                self.mainwindow = mainwindow
                ...
            ...

            # Handle events
            def after_image_open_event(self):
                # do something

            ...

4.3 Write the Packaging File
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

-   ``ProjectName``/``setup.py``

    .. code-block:: python

        from setuptools import setup, find_packages

        def get_version():
            try:
                from {PluginPackage}.__init__ import __version__
                return __version__

            except FileExistsError:
                FileExistsError('__init__.py not exists.')

        version, author = get_version()

        setup(
            name="isat-plugin-{custom}",                 # It is recommended that the package name start with "isat-plugin".
            version={version},
            author={author},
            keywords=["isat-sam", "isat plugin", ...],

            packages=find_packages(),
            include_package_data=True,

            python_requires=">=3.8",
            install_requires=[
                'isat-sam>=1.4.0',
            ],

            entry_points={
                "isat.plugins": [
                    "{custom_plugin} = {PluginPackage}.main:{CustomPlugin}",
                ]
            }
        )

4.4 Packaging
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

-   Package as a source distribution

    .. code-block:: shell

        cd {ProjectName}
        python setup.py sdist

-   Install

    .. code-block:: shell

        pip install dist/{isat_plugin_custom}-{version}.tar.gz

.. tip:: You can refer to the plugins: `AutoAnnotatePlugin <https://github.com/yatengLG/ISAT_plugin_auto_annotate>`_ and `MaskExportPlugin <https://github.com/yatengLG/ISAT_plugin_mask_export>`_ to develop your own plugins.

4.5 Share
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

-  You can directly share the package file dist/xxx.tar.gz with others and then install with:

   .. code-block:: shell

      pip install xxx.tar.gz

-  Or upload the package file to `pypi <https://pypi.org/>`_, and then install with:

   .. code-block:: shell

      pip install {package name}

