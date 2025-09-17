# -*- coding: utf-8 -*-
# @Author  : LG

import os
import codecs
from setuptools import setup, find_packages


def get_version():
    try:
        from ISAT.__init__ import __version__
        return __version__

    except FileExistsError:
        FileExistsError('ISAT/__init__.py not exists.')


setup(
    name="isat-sam",                                        # 包名
    version=get_version(),                                  # 版本号
    author="yatengLG",
    author_email="yatenglg@foxmail.com",
    description="Interactive semi-automatic annotation tool for image segmentation based on SAM(segment anything model).",
    long_description=(codecs.open("README.md", encoding='utf-8').read()),
    long_description_content_type="text/markdown",

    url="https://github.com/yatengLG/ISAT_with_segment_anything",  # 项目相关文件地址

    keywords=["annotation tool", "segment anything", "image annotation", "semantic segmentation", 'instance segmentation'],
    license="Apache2.0",

    packages=find_packages(),
    include_package_data=True,

    python_requires=">=3.8",                            # python 版本要求
    install_requires=[                                  # 必须直接指明，不然pip包不会自动安装
        'imgviz',
        'scikit-image',
        'numpy',
        'opencv_python_headless',
        'pillow',
        'pyqt5',
        'pyyaml',
        'torch>=2.1.1',
        'torchvision',
        'pycocotools',
        'timm',
        'shapely',
        'hydra-core>=1.3.2',
        'tqdm>=4.66.1',
        'fuzzywuzzy',
        'python-Levenshtein',
        'iopath',
        'orjson',
        'pydicom'
        ],

    classifiers=[
        "Intended Audience :: Developers",              # 目标用户:开发者
        "Intended Audience :: Science/Research",        # 目标用户:学者
        'Development Status :: 5 - Production/Stable',
        "Natural Language :: Chinese (Simplified)",
        "Natural Language :: English",
        'License :: OSI Approved :: Apache Software License',

        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={
        "console_scripts": [
            "isat-sam=ISAT.main:main",
        ],
        "isat.plugins": []  # 插件注册
    },
)
