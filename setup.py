# -*- coding: utf-8 -*-
# @Author  : LG

import os
import codecs
from setuptools import setup, find_packages


def get_version():
    try:
        from ISAT.__init__ import __version__
        return __version__

    except:
        FileExistsError('ISAT/__init__.py not exists.')


def get_install_requires():
    requirements_file = 'requirements.txt'
    requirements_list = []
    if os.path.exists(requirements_file):
        with open(requirements_file) as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip('\n')
                if line != '':
                    requirements_list.append(line)
    return requirements_list

setup(
    name="isat-sam",                                        # 包名
    version=get_version(),                              # 版本号
    author="yatengLG",
    author_email="yatenglg@foxmail.com",
    description="Interactive semi-automatic annotation tool for image segmentation based on SAM(segment anything model).",
    long_description=(codecs.open("README.md", encoding='utf-8').read() + "\n" +
                      codecs.open("CHANGELOG.md", encoding='utf-8').read()),
    long_description_content_type="text/markdown",

    url="https://github.com/yatengLG/ISAT_with_segment_anything",  # 项目相关文件地址

    keywords=["annotation tool", "segment anything", "image annotation", "semantic segmentation", 'instance segmentation'],
    license="Apache2.0",

    packages=find_packages(),
    include_package_data=True,

    python_requires=">=3.7",                            # python 版本要求
    install_requires=[
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
        ],

    classifiers=[
        "Intended Audience :: Developers",              # 目标用户:开发者
        "Intended Audience :: Science/Research",        # 目标用户:学者
        'Development Status :: 4 - Beta',
        "Natural Language :: Chinese (Simplified)",
        "Natural Language :: English",
        'License :: OSI Approved :: Apache Software License',

        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={
        "console_scripts": [
            "isat-sam=ISAT.main:main",
        ],
    },
)