# -*- coding: utf-8 -*-
"""chembl_parse.py —— 引号感知的 mysqldump extended-insert 元组解析器。
处理: 单引号字符串(可含逗号/括号/换行转义)、嵌套括号。
"""
import re, csv

def split_tuples(body):
    """把 'INSERT INTO x VALUES (..),(..);' 的 VALUES 部分切成元组字符串列表(去外层括号)。"""
    tuples = []
    buf = ''
    depth = 0
    in_str = False
    i = 0
    n = len(body)
    while i < n:
        ch = body[i]
        if ch == "'":
            # 处理转义单引号 \'
            if in_str and i + 1 < n and body[i+1] == "'":
                buf += "''"
                i += 2
                continue
            in_str = not in_str
            buf += ch
        elif not in_str and ch == '(':
            depth += 1
            if depth == 1:
                buf = ''
            else:
                buf += ch
        elif not in_str and ch == ')':
            depth -= 1
            if depth == 0:
                tuples.append(buf)
            else:
                buf += ch
        else:
            if depth >= 1:
                buf += ch
        i += 1
    return tuples

def parse_fields(tup):
    """元组字符串(无外层括号) → 字段列表。用 csv 处理单引号字符串。"""
    try:
        return next(csv.reader([tup], quotechar="'", skipinitialspace=True))
    except Exception:
        # fallback: 简单逗号分割(去掉单引号)
        return [f.strip().strip("'") for f in tup.split(',')]
