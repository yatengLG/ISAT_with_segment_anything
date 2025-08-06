<h1 align='center'>ISAT_with_segment_anything</h1>
<h2 align='center'>一款基于SAM的交互式半自动图像分割标注工具</h2>
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
<p align='center'><img src="./display/标注.gif" alt="标注.gif"></p>

集成[segment anything](https://github.com/facebookresearch/segment-anything)及bf16模式，实现低显存图片分割快速标注。

演示视频：[bilibili](https://www.bilibili.com/video/BV1or4y1R7EJ/)

请查阅我们最新的[中文文档](https://isat-sam.readthedocs.io/zh-cn/latest/)

---

# 软件特点及安装
- &#x1F389;: **V1.4.0版本添加了插件系统。** 可以使用较少量的代码，扩展ISAT的功能。
  
    以下是一些插件示例:
  - [ISAT_plugin_auto_annotate](https://github.com/yatengLG/ISAT_plugin_auto_annotate) ![PyPI - Version](https://img.shields.io/pypi/v/isat-plugin-auto-annotate?style=social&logo=pypi)
 ![Pepy Total Downloads](https://img.shields.io/pepy/dt/isat-plugin-auto-annotate?style=social) : 仅用240行代码实现的**自动标注**功能（使用yolo模型）。
  - [ISAT_plugin_mask_export](https://github.com/yatengLG/ISAT_plugin_mask_export) ![PyPI - Version](https://img.shields.io/pypi/v/isat-plugin-mask-export?style=social&logo=pypi)
![Pepy Total Downloads](https://img.shields.io/pepy/dt/isat-plugin-mask-export?style=social) : 仅用160行代码实现的**mask导出**功能。
  

## 安装
- 新建conda环境（推荐，可选）
```shell
# 创建环境
conda create -n isat_env python=3.8

# 激活环境
conda activate isat_env
```

- 安装
```shell
pip install isat-sam
```

- 运行
```shell
isat-sam
```

# Star History

**请给该项目一个star，您的点赞就是对我最大的支持与鼓励**
[![Star History Chart](https://api.star-history.com/svg?repos=yatengLG/ISAT_with_segment_anything&type=Date)](https://star-history.com/#yatengLG/ISAT_with_segment_anything&Date)


# 核心贡献者

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


# 引用
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
