# 功能：序列化研究流程中的小型消息与材料。
"""Serialization of small research messages; originals stay in EvidenceStore."""
import json


def encoded(value):
    return json.dumps(value, ensure_ascii=False)
