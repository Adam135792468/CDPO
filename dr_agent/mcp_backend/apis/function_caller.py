# -*- coding: utf-8 -*-
"""
函数调用器模块
提供FunctionCaller类和create_function_caller函数
"""

from typing import List
from .trial_tools import (
    get_trial_info,
    get_drug_patents,
    get_drug_approvals,
    get_drug_exclusivities
)


class FunctionCaller:
    """
    函数调用器类
    封装所有工具函数的调用
    """
    
    def __init__(self):
        """初始化函数调用器"""
        pass
    
    def get_trial_info(self, nct_id: str):
        """
        获取临床试验信息
        
        Args:
            nct_id: NCT编号
        
        Returns:
            临床试验信息字典
        """
        return get_trial_info(nct_id)
    
    def get_drug_patents(self, ingredients: List[str]):
        """
        获取药物专利信息
        
        Args:
            ingredients: 药物成分列表
        
        Returns:
            专利信息字典
        """
        return get_drug_patents(ingredients)
    
    def get_drug_approvals(self, ingredients: List[str]):
        """
        获取药物批准信息
        
        Args:
            ingredients: 药物成分列表
        
        Returns:
            批准信息字典
        """
        return get_drug_approvals(ingredients)
    
    def get_drug_exclusivities(self, ingredients: List[str]):
        """
        获取药物独占期信息
        
        Args:
            ingredients: 药物成分列表
        
        Returns:
            独占期信息字典
        """
        return get_drug_exclusivities(ingredients)


def create_function_caller() -> FunctionCaller:
    """
    创建函数调用器实例
    
    Returns:
        FunctionCaller实例
    """
    return FunctionCaller()

