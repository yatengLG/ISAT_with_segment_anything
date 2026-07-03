# -*- coding: utf-8 -*-
# @Author  : LG

import re

def has_chinese(text: str) -> bool:
    """判断字符串中是否包含汉字"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))