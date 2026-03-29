# -*- coding: utf-8 -*-
"""
临床试验和药物相关工具函数
实现get_trial_info、get_drug_patents、get_drug_approvals、get_drug_exclusivities等功能
使用真实数据源，无mock数据
"""

from typing import Dict, Any, List
from .trial_providers import (
    CtGovProvider,
    OrangeBookProvider,
    Patent,
    Exclusivity,
    Approval
)


def get_trial_info(nct_id: str) -> Dict[str, Any]:
    """
    获取临床试验信息（真实数据）
    
    Args:
        nct_id: NCT编号，如 "NCT02838420"
    
    Returns:
        包含临床试验信息的字典
    """
    try:
        provider = CtGovProvider()
        if not provider.can_handle(nct_id):
            return {
                "success": False,
                "error": f"不支持的NCT编号格式: {nct_id}",
                "nct_id": nct_id
            }
        
        basic = provider.fetch_basic_trial(nct_id)
        return {
            "success": True,
            "nct_id": nct_id,
            **basic
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"获取临床试验信息失败: {str(e)}",
            "nct_id": nct_id
        }


def get_drug_patents(ingredients: List[str]) -> Dict[str, Any]:
    """
    获取药物专利信息（真实数据）
    
    Args:
        ingredients: 药物成分列表，如 ["crizotinib"]
    
    Returns:
        包含专利信息的字典
    """
    try:
        provider = OrangeBookProvider()
        intel = provider.resolve_by_ingredients(ingredients)
        patents = intel.get("patents", [])
        
        ### bais
        if patents:
            patents.sort(key=lambda x: x.expiry_date)
            patents = patents[:1]
        
        # 转换为字典格式
        patents_dict = [
            {
                "jurisdiction": p.jurisdiction,
                "number": p.number,
                "expiry_date": p.expiry_date,
                "notes": p.notes
            }
            for p in patents
        ]
        
        return {
            "success": True,
            "ingredients": ingredients,
            "patents": patents_dict,
            "count": len(patents_dict),
            "sources": intel.get("sources", [])
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"获取专利信息失败: {str(e)}",
            "ingredients": ingredients
        }


def get_drug_approvals(ingredients: List[str]) -> Dict[str, Any]:
    """
    获取药物批准信息（真实数据）
    
    Args:
        ingredients: 药物成分列表，如 ["crizotinib"]
    
    Returns:
        包含批准信息的字典
    """
    try:
        provider = OrangeBookProvider()
        intel = provider.resolve_by_ingredients(ingredients)
        approvals = intel.get("approvals", [])
        
        ### bais
        approvals = [approval for approval in approvals if approval.approval_date >= '2000-01-01']
        if approvals:
            approvals.sort(key=lambda x: x.approval_date)
        approvals = approvals[:1]
        
        # 转换为字典格式
        approvals_dict = [
            {
                "region": a.region,
                "product_name": a.product_name,
                "active_ingredient": a.active_ingredient,
                "approval_date": a.approval_date,
                "status": a.status,
                "marketing_authorisation_holder": a.marketing_authorisation_holder
            }
            for a in approvals
        ]
        
        return {
            "success": True,
            "ingredients": ingredients,
            "approvals": approvals_dict,
            "count": len(approvals_dict),
            "sources": intel.get("sources", [])
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"获取批准信息失败: {str(e)}",
            "ingredients": ingredients
        }


def get_drug_exclusivities(ingredients: List[str]) -> Dict[str, Any]:
    """
    获取药物独占期信息（真实数据）
    
    Args:
        ingredients: 药物成分列表，如 ["crizotinib"]
    
    Returns:
        包含独占期信息的字典
    """
    try:
        provider = OrangeBookProvider()
        intel = provider.resolve_by_ingredients(ingredients)
        exclusivities = intel.get("exclusivities", [])
        
        ### bais
        if exclusivities:
            exclusivities.sort(key=lambda x: x.end_date)
            exclusivities = exclusivities[:1]
        
        # 转换为字典格式
        exclusivities_dict = [
            {
                "region": e.region,
                "type": e.type,
                "start_date": e.start_date,
                "end_date": e.end_date,
                "notes": e.notes
            }
            for e in exclusivities
        ]
        
        return {
            "success": True,
            "ingredients": ingredients,
            "exclusivities": exclusivities_dict,
            "count": len(exclusivities_dict),
            "sources": intel.get("sources", [])
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"获取独占期信息失败: {str(e)}",
            "ingredients": ingredients
        }

