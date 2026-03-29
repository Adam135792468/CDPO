# -*- coding: utf-8 -*-
"""
搜索结果格式化模块
"""

from typing import Dict, Any


def format_search_result(result: Dict[str, Any]) -> str:
    """
    将search结果格式化为可读文本
    
    Args:
        result: 搜索结果字典
    
    Returns:
        格式化后的字符串
    """
    if not result.get("success", False):
        return f"查询失败：{result.get('error', '未知错误')}"
    
    lines = []
    
    # 基本信息
    if "nct_id" in result:
        lines.append(f"NCT编号: {result['nct_id']}")
    
    # 处理ingredients（可能是列表或字符串）
    if "ingredients" in result:
        ingredients = result['ingredients']
        if isinstance(ingredients, list):
            if ingredients:
                lines.append(f"药物成分: {', '.join(ingredients)}")
        else:
            if ingredients:
                lines.append(f"药物成分: {ingredients}")
    
    if "sponsor" in result:
        lines.append(f"赞助商: {result['sponsor']}")
    
    if "recruitment_status" in result:
        lines.append(f"招募状态: {result['recruitment_status']}")
    elif "status" in result:
        # 兼容旧格式
        lines.append(f"招募状态: {result['status']}")
    
    # 批准信息
    if "approvals" in result and result["approvals"]:
        approvals = result["approvals"]
        lines.append(f"\n批准信息（共{len(approvals)}条）:")
        for i, appr in enumerate(approvals[:5], 1):
            lines.append(f"  {i}. 产品名称: {appr.get('product_name', 'N/A')}")
            lines.append(f"     活性成分: {appr.get('active_ingredient', 'N/A')}")
            lines.append(f"     批准日期: {appr.get('approval_date', 'N/A')}")
            lines.append(f"     申请公司: {appr.get('marketing_authorisation_holder', 'N/A')}")
    
    # 专利信息
    if "patents" in result and result["patents"]:
        patents = result["patents"]
        lines.append(f"\n专利信息（共{len(patents)}条）:")
        for i, pat in enumerate(patents[:5], 1):
            lines.append(f"  {i}. 专利号: {pat.get('number', 'N/A')}")
            lines.append(f"     过期日期: {pat.get('expiry_date', 'N/A')}")
    
    # 独占期信息
    if "exclusivities" in result and result["exclusivities"]:
        exclusivities = result["exclusivities"]
        lines.append(f"\n独占期信息（共{len(exclusivities)}条）:")
        for i, excl in enumerate(exclusivities[:5], 1):
            lines.append(f"  {i}. 类型: {excl.get('type', 'N/A')}")
            lines.append(f"     结束日期: {excl.get('end_date', 'N/A')}")
    
    # 数据来源
    if "sources" in result:
        lines.append(f"\n数据来源: {', '.join(result['sources'])}")
    elif "source_url" in result:
        # 兼容旧格式
        lines.append(f"\n数据来源: {result['source_url']}")
    
    return "\n".join(lines) if lines else str(result)

