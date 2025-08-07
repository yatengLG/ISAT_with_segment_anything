<h1 align='center'>ISAT_with_segment_anything [isat-sam]</h1>
<h2 align='center'>An Interactive Semi-Automatic Annotation Tool Based on Segment Anything</h2>
<p align='center'>
    <a href='https://github.com/yatengLG/ISAT_with_segment_anything' target="_blank"><img alt="GitHub forks" src="https://img.shields.io/github/stars/yatengLG/ISAT_with_segment_anything"></a>
    <a href='https://github.com/yatengLG/ISAT_with_segment_anything' target="_blank"><img alt="GitHub forks" src="https://img.shields.io/github/forks/yatengLG/ISAT_with_segment_anything"></a>
    <a href='https://pypi.org/project/isat-sam/' target="_blank"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/isat-sam?style=social&logo=pypi"></a>
    <a href='https://pypi.org/project/isat-sam/' target="_blank"><img alt="Pepy Total Downlods" src="https://img.shields.io/pepy/dt/isat-sam?style=social&logo=pypi"></a>
</p>
<p align='center'>
    <a href='README-cn.md'><b>[中文]</b></a>
    <a href='README.md'><b>[English]</b></a>
</p>
<p align='center'><img src="./display/software.gif" alt="software.gif"></p>

Focusing on the field of image segmentation, we strive to create the best image segmentation annotation software.

Please refers to our latest [Documentation in English](https://isat-sam.readthedocs.io/en/latest/#) or [中文文档](https://isat-sam.readthedocs.io/zh-cn/latest/)

---

# News
- &#x1F389;: **In version 1.4.0, ISAT add a plugin system.** You can use a small amount of code to extend the functionality of ISAT.
  
    Here are some official plugin examples:
  - [ISAT_plugin_auto_annotate](https://github.com/yatengLG/ISAT_plugin_auto_annotate) ![PyPI - Version](https://img.shields.io/pypi/v/isat-plugin-auto-annotate?style=social&logo=pypi)
 ![Pepy Total Downloads](https://img.shields.io/pepy/dt/isat-plugin-auto-annotate?style=social) : An auto-annotation function based on the YOLO model, implemented with just 240 lines of code.
  - [ISAT_plugin_mask_export](https://github.com/yatengLG/ISAT_plugin_mask_export) ![PyPI - Version](https://img.shields.io/pypi/v/isat-plugin-mask-export?style=social&logo=pypi)
![Pepy Total Downloads](https://img.shields.io/pepy/dt/isat-plugin-mask-export?style=social) : A mask export function, implemented with just 160 lines of code.

- For other versions and the release note, please refer to [releases](https://github.com/yatengLG/ISAT_with_segment_anything/releases)

# Install

- Create a conda environment(recommended, optional)
    ```shell
    # create environment
    conda create -n isat_env python=3.8
    
    # activate environment
    conda activate isat_env
    ```

- Install
    ```shell
    pip install isat-sam
    ```

- Run
    ```shell
    isat-sam
    ```

# Star History

**Please support us with a star—it's like a virtual coffee!**
[![Star History Chart](https://api.star-history.com/svg?repos=yatengLG/ISAT_with_segment_anything&type=Date)](https://star-history.com/#yatengLG/ISAT_with_segment_anything&Date)


# Contributors

<table border="0">
<tr>
    <td><img alt="yatengLG" src="https://avatars.githubusercontent.com/u/31759824?v=4" width="60" height="60" href="">
    <td><img alt="Alias-z" src="https://avatars.githubusercontent.com/u/66273343?v=4" width="60" height="60" href="">
    <td>...
</td>
</tr>
<tr>
  <td><a href="https://github.com/yatengLG">yatengLG</a>
  <td><a href="https://github.com/Alias-z">Alias-z</a>
    <td><a href="https://github.com/yatengLG/ISAT_with_segment_anything/graphs/contributors">...</a>
</tr>
</table>


# Citation
```text
@misc{ISAT_with_segment_anything,
  title={{ISAT with Segment Anything: An Interactive Semi-Automatic Annotation Tool}},
  author={Ji, Shuwei and Zhang, Hongyuan},
  url={https://github.com/yatengLG/ISAT_with_segment_anything},
  note={Updated on 2025-02-07},
  year={2024},
  version={1.33}
}
```
