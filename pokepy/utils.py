# -*- coding: utf-8 -*-
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_DOWN

def round_half_up(v: float) -> int:
    """四捨五入した値を返す"""
    return int(Decimal(str(v)).quantize(Decimal('0'),rounding=ROUND_HALF_UP))

def round_half_down(v: float) -> int:
    """五捨五超入した値を返す"""
    return int(Decimal(str(v)).quantize(Decimal('0'),rounding=ROUND_HALF_DOWN))

def push(dict: dict, key: str, value: int|float):
    """dictに要素を追加する。すでにkeyがある場合はvalueを加算する"""
    if key not in dict:
        dict[key] = value
    else:
        dict[key] += value

def zero_ratio(dict: dict) -> float:
    """keyがゼロのvalueを全valueの合計値で割った値を返す"""
    n, n0 = 0, 0
    for key in dict:
        n += dict[key]
        if float(key) == 0:
            n0 += dict[key]
    return n0/n

def offset_hp_keys(hp_dict: dict, v: int) -> dict:
    """hp dictのすべてのkeyにvを加算したdictを返す"""
    result = {}
    for hp in hp_dict:
        h = int(float(hp))
        new_hp = '0' if h == 0 else str(max(0, h+v))
        if new_hp != '0' and hp[-2:] == '.0':
            new_hp += '.0'
        push(result, new_hp, hp_dict[hp])
    return result

def to_hankaku(text: str) -> str:
    """全角英数字を半角に変換した文字列を返す"""
    return text.translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)})).replace('・','･')

def average(ls: list[float]) -> float:
    return sum(ls)/len(ls)

def frac(v: float) -> float:
    return v - int(v)