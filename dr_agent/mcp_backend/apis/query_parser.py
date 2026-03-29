# -*- coding: utf-8 -*-
"""
查询解析模块
提供查询字符串解析功能
"""

import re
from typing import Optional, List


def parse_search_call(search_call: str) -> dict:
    """
    解析search函数调用字符串
    
    Args:
        search_call: 例如 'search(query="NCT02838420", prefer_url="...", reason="...")'
        
    Returns:
        包含query, prefer_url, reason的字典
    """
    # 提取query参数
    query_match = re.search(r'query=["\']([^"\']+)["\']', search_call)
    query = query_match.group(1) if query_match else ""
    
    # 提取prefer_url参数（可选）
    prefer_url_match = re.search(r'prefer_url=["\']([^"\']+)["\']', search_call)
    prefer_url = prefer_url_match.group(1) if prefer_url_match else None
    
    # 提取reason参数（可选）
    reason_match = re.search(r'reason=["\']([^"\']+)["\']', search_call)
    reason = reason_match.group(1) if reason_match else None
    
    return {
        "query": query,
        "prefer_url": prefer_url,
        "reason": reason
    }


def extract_nct_id(query: str) -> Optional[str]:
    """
    从query中提取NCT编号
    
    Args:
        query: 查询字符串
    
    Returns:
        NCT编号，如果未找到则返回None
    """
    nct_match = re.search(r'NCT\d{8}', query, re.IGNORECASE)
    return nct_match.group(0).upper() if nct_match else None


def extract_ingredients(query: str) -> List[str]:
    """
    从query中提取药物成分
    支持格式：
    - "ingredients ingredient1 ingredient2"
    - "ingredient1, ingredient2"
    
    Args:
        query: 查询字符串
    
    Returns:
        药物成分列表
    """
    # 移除"ingredients"关键词
    cleaned = re.sub(r'\bingredients?\s*:?\s*', '', query, flags=re.IGNORECASE)
    
    # 按逗号或换行分割
    ingredients = []
    for part in re.split(r'[,;\n]', cleaned):
        part = part.strip()
        if part:
            ingredients.append(part)
    
    return ingredients if ingredients else [cleaned.strip()] if cleaned.strip() else []

