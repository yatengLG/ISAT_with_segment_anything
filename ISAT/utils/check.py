# -*- coding: utf-8 -*-
# @Author  : LG

import re

def has_chinese(text: str) -> bool:
    """判断字符串中是否包含汉字"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def calculate_area(points: list[tuple[float, float]]) -> float:
    """计算多边形面积"""
    area = 0
    num_points = len(points)
    for i in range(num_points):
        p1 = points[i]
        p2 = points[(i + 1) % num_points]
        d = p1[0] * p2[1] - p2[0] * p1[1]
        area += d
    return abs(area) / 2
