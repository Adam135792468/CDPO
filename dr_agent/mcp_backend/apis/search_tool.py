# -*- coding: utf-8 -*-
"""
搜索工具模块
内联了search_anything的功能，不再依赖外部文件夹
"""

from typing import Dict, Any, Optional
from .query_parser import parse_search_call, extract_nct_id, extract_ingredients
from .function_caller import create_function_caller
from .result_formatter import format_search_result

# 导出主要函数，保持向后兼容
__all__ = ['execute_search_call', 'format_search_result', 'search']


def search(query: str, prefer_url: Optional[str] = None, reason: Optional[str] = None) -> Dict[str, Any]:
    """
    执行search操作，根据query内容调用相应的工具函数
    
    Args:
        query: 搜索查询内容
        prefer_url: 偏好URL（用于说明数据来源）
        reason: 搜索原因
    
    Returns:
        工具函数执行结果
    """
    caller = create_function_caller()
    
    # 检查是否包含NCT编号
    nct_id = extract_nct_id(query)
    if nct_id:
        # 调用get_trial_info
        result = caller.get_trial_info(nct_id)
        result["_search_metadata"] = {
            "query": query,
            "prefer_url": prefer_url,
            "reason": reason,
            "function_called": "get_trial_info"
        }
        return result
    
    # 检查是否包含ingredients关键词
    if "ingredient" in query.lower():
        ingredients = extract_ingredients(query)
        if ingredients:
            # 根据reason判断调用哪个函数
            if reason:
                reason_lower = reason.lower()
                # 检查专利（支持中英文）
                if "patent" in reason_lower or "专利" in reason_lower:
                    result = caller.get_drug_patents(ingredients)
                    func_name = "get_drug_patents"
                # 检查独占期（支持中英文）
                elif "exclusivity" in reason_lower or "独占" in reason_lower:
                    result = caller.get_drug_exclusivities(ingredients)
                    func_name = "get_drug_exclusivities"
                # 检查批准/公司（支持中英文）
                elif "approval" in reason_lower or "批准" in reason_lower or "fda" in reason_lower or "company" in reason_lower or "公司" in reason_lower or "申请" in reason_lower:
                    result = caller.get_drug_approvals(ingredients)
                    func_name = "get_drug_approvals"
                else:
                    # 默认调用approvals
                    result = caller.get_drug_approvals(ingredients)
                    func_name = "get_drug_approvals"
            else:
                # 没有reason，默认调用approvals
                result = caller.get_drug_approvals(ingredients)
                func_name = "get_drug_approvals"
            
            result["_search_metadata"] = {
                "query": query,
                "prefer_url": prefer_url,
                "reason": reason,
                "function_called": func_name
            }
            return result
    
    # 如果无法识别，返回错误
    return {
        "success": False,
        "error": f"无法识别查询类型: {query}",
        "query": query,
        "prefer_url": prefer_url,
        "reason": reason
    }


def execute_search_call(search_call: str) -> Dict[str, Any]:
    """
    执行search函数调用字符串
    
    Args:
        search_call: 例如 'search(query="NCT02838420", prefer_url="...", reason="...")'
    
    Returns:
        工具函数执行结果
    """
    parsed = parse_search_call(search_call)
    return search(
        query=parsed["query"],
        prefer_url=parsed.get("prefer_url"),
        reason=parsed.get("reason")
    )

