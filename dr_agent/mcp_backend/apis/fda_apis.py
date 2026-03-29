# -*- coding: utf-8 -*-
"""
FDA API - 药物信息检索
从 FDA Drug Label API 获取药物信息，并使用 LLM 提取相关信息
"""
import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlencode

import aiohttp
import requests


class FDASearchResult:
    """FDA 搜索结果数据模型"""
    
    def __init__(
        self,
        keyword: str,
        focus: str,
        extracted_info: List[str],
        search_strategy: str,
        raw_results: Optional[Dict[str, Any]] = None,
    ):
        self.keyword = keyword
        self.focus = focus
        self.extracted_info = extracted_info
        self.search_strategy = search_strategy
        self.raw_results = raw_results


def _get_proxy() -> Optional[str]:
    """获取代理设置"""
    return os.getenv("https_proxy") or os.getenv("HTTPS_PROXY") or os.getenv("http_proxy") or os.getenv("HTTP_PROXY")


async def _call_openrouter_llm_async(
    prompt: str,
    api_key: Optional[str] = None,
    model_name: str = "bytedance-seed/seed-1.6",
    max_retries: int = 3,
) -> str:
    """
    异步调用 OpenRouter API 进行信息提取

    Args:
        prompt: 提示词
        api_key: OpenRouter API Key (可选，优先从环境变量读取)
        model_name: 模型名称
        max_retries: 最大重试次数

    Returns:
        提取的信息
    """
    # 优先从环境变量读取 API key
    env_api_key = os.getenv("OPENROUTER_API_KEY")
    api_key = env_api_key or api_key

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY 环境变量未设置")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    messages = [{"role": "user", "content": prompt}]
    payload = {"model": model_name, "messages": messages}

    base_sleep_time = 10
    proxy = _get_proxy()

    async with aiohttp.ClientSession() as session:
        for retry_count in range(max_retries):
            try:
                async with session.post(
                    url, headers=headers, json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                    proxy=proxy
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        assistant_message = result['choices'][0]['message']
                        extracted_info = assistant_message.get('content', '').strip()
                        return extracted_info
                    else:
                        print(f"调用 OpenRouter 失败 (尝试 {retry_count + 1}/{max_retries}): {response.status}")
                        if retry_count < max_retries - 1:
                            sleep_time = base_sleep_time * (2 ** retry_count)
                            await asyncio.sleep(sleep_time)

            except Exception as e:
                print(f"调用 OpenRouter 出错 (尝试 {retry_count + 1}/{max_retries}): {str(e)}")
                if retry_count < max_retries - 1:
                    sleep_time = base_sleep_time * (2 ** retry_count)
                    await asyncio.sleep(sleep_time)

    # 如果所有重试都失败，返回空字符串
    return ""


def _call_openrouter_llm(
    prompt: str,
    api_key: Optional[str] = None,
    model_name: str = "bytedance-seed/seed-1.6",
    max_retries: int = 3,
) -> str:
    """
    同步调用 OpenRouter API 进行信息提取（保留用于向后兼容）

    Args:
        prompt: 提示词
        api_key: OpenRouter API Key (可选，优先从环境变量读取)
        model_name: 模型名称
        max_retries: 最大重试次数

    Returns:
        提取的信息
    """
    # 优先从环境变量读取 API key
    env_api_key = os.getenv("OPENROUTER_API_KEY")
    api_key = env_api_key or api_key

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY 环境变量未设置")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    messages = [{"role": "user", "content": prompt}]
    payload = {"model": model_name, "messages": messages}

    base_sleep_time = 10

    for retry_count in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)

            if response.status_code == 200:
                result = response.json()
                assistant_message = result['choices'][0]['message']
                extracted_info = assistant_message.get('content', '').strip()
                return extracted_info
            else:
                print(f"调用 OpenRouter 失败 (尝试 {retry_count + 1}/{max_retries}): {response.status_code}")
                if retry_count < max_retries - 1:
                    sleep_time = base_sleep_time * (2 ** retry_count)
                    time.sleep(sleep_time)

        except Exception as e:
            print(f"调用 OpenRouter 出错 (尝试 {retry_count + 1}/{max_retries}): {str(e)}")
            if retry_count < max_retries - 1:
                sleep_time = base_sleep_time * (2 ** retry_count)
                time.sleep(sleep_time)

    # 如果所有重试都失败，返回空字符串
    return ""


def search_fda_drug_label(
    keyword: str,
    focus: str,
    limit: int = 5,
    use_llm_extraction: bool = True,
    api_key: Optional[str] = None,
    model_name: str = "bytedance-seed/seed-1.6",
) -> Dict[str, Any]:
    """
    从 FDA API 获取药物信息（三级搜索策略）
    
    Args:
        keyword: 药物名称
        focus: 关注点（用于提取相关信息，如 "adverse reactions", "indications" 等）
        limit: 返回结果数量限制
        use_llm_extraction: 是否使用 LLM 提取相关信息
        api_key: OpenRouter API Key (用于 LLM 提取)
        model_name: LLM 模型名称
        
    Returns:
        Dictionary containing:
        - keyword: 搜索的药物名称
        - focus: 关注点
        - search_strategy: 成功的搜索策略 (BRAND_NAME, GENERIC_NAME, FINDALL_NAME)
        - extracted_info: 提取的相关信息列表
        - data: 原始 FDA 数据（如果不使用 LLM 提取）
        - error: 错误信息（如果搜索失败）
    """
    fda_api_key = "nwhIkQlmveH44at4S8uBKk94UWs1e7KC3A7nTNv8"
    
    # 优先搜索策略：品牌名和通用名
    priority_searches = [
        {
            "name": "BRAND_NAME",
            "base_url": "https://api.fda.gov/drug/label.json",
            "limit": 1,
            "search": f'(openfda.brand_name:("{keyword}"))',
            "api_key": fda_api_key
        },
        {
            "name": "GENERIC_NAME",
            "base_url": "https://api.fda.gov/drug/label.json",
            "limit": 1,
            "search": f'(openfda.generic_name:("{keyword}"))',
            "api_key": fda_api_key
        },
    ]
    
    # 回退搜索策略
    fallback_search = {
        "name": "FINDALL_NAME",
        "base_url": "https://api.fda.gov/drug/label.json",
        "limit": min(limit, 3),
        "search": f'"{keyword}"',
        "api_key": fda_api_key
    }
    
    raw_results = None
    search_strategy = None
    
    # 先尝试优先搜索策略
    for search_config in priority_searches:
        try:
            max_retries = 3
            retry_delay = 1
            query_params = {k: v for k, v in search_config.items() if k not in ['name', 'base_url']}
            query_str = urlencode(query_params, quote_via=quote)
            url = f"{search_config['base_url']}?{query_str}"
            
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, timeout=60)
                    response.raise_for_status()
                    result_data = response.json()
                    
                    if 'results' in result_data and result_data['results']:
                        raw_results = result_data
                        search_strategy = search_config['name']
                        print(f"✅ {search_config['name']} 成功找到结果")
                        break
                        
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        print(f"⚠️ {search_config['name']} 失败，{retry_delay}秒后重试...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        
            if raw_results:
                break
                
        except Exception as e:
            print(f"❌ {search_config['name']} 出错: {e}")
            continue
    
    # 如果优先搜索没有找到结果，使用回退搜索
    if not raw_results:
        print("⚠️ 优先搜索未找到结果，使用回退搜索...")
        try:
            max_retries = 3
            retry_delay = 1
            query_params = {k: v for k, v in fallback_search.items() if k not in ['name', 'base_url']}
            query_str = urlencode(query_params, quote_via=quote)
            url = f"{fallback_search['base_url']}?{query_str}"
            
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, timeout=60)
                    response.raise_for_status()
                    result_data = response.json()
                    
                    if 'results' in result_data and result_data['results']:
                        raw_results = result_data
                        search_strategy = fallback_search['name']
                        print("✅ 回退搜索成功")
                    else:
                        print("⚠️ 回退搜索也未找到结果")
                    break
                    
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        print(f"⚠️ 回退搜索失败，{retry_delay}秒后重试...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        
        except Exception as e:
            print(f"❌ 回退搜索出错: {e}")
    
    # 如果所有搜索都失败
    if not raw_results:
        return {
            "keyword": keyword,
            "focus": focus,
            "search_strategy": None,
            "extracted_info": [],
            "data": [],
            "error": f"No FDA data found for drug: {keyword}"
        }
    
    # 处理搜索结果
    results_list = raw_results['results']
    is_fallback = search_strategy == "FINDALL_NAME"
    
    # 准备待处理的数据
    related_parts = []
    if is_fallback:
        # 回退搜索：返回多个完整的结果
        related_parts = results_list[:limit]
    else:
        # 精确搜索：提取所有字段（排除 table 字段）
        for information_dict in results_list:
            for key, value in information_dict.items():
                if "_table" not in key:
                    related_parts.append({key: value})
        related_parts = [str(related_parts)]
    
    # 如果不使用 LLM 提取，直接返回原始数据
    if not use_llm_extraction:
        return {
            "keyword": keyword,
            "focus": focus,
            "search_strategy": search_strategy,
            "extracted_info": [],
            "data": related_parts,
            "error": None
        }
    
    # 使用 LLM 提取相关信息
    all_results = []
    for information_dict in related_parts:
        # 构造提示词
        prompt = f"You need to extract information from FDA drug data based on a specific goal.\n\n"
        prompt += f"Drug/Keyword: {keyword}\n"
        prompt += f"Goal: {focus}\n\n"
        prompt += f"Instructions:\n"
        prompt += f"- Extract ONLY the information directly relevant to the goal\n"
        prompt += f"- Use original text from the data (do not paraphrase or add explanations)\n"
        prompt += f"- Combine relevant excerpts into a coherent response\n"
        prompt += f"- If no relevant information is found, return 'No relevant information found'\n\n"
        
        if is_fallback:
            prompt += f"Drug description: {information_dict.get('description', '')}\n\n"
        
        prompt += f"FDA Data:\n{information_dict}\n\n"
        prompt += f"Extracted information based on the goal:"
        
        # 调用 LLM 提取信息
        extracted_info = _call_openrouter_llm(
            prompt=prompt,
            api_key=api_key,
            model_name=model_name,
            max_retries=3
        )
        
        # 如果 LLM 调用失败，使用原始数据
        if not extracted_info:
            extracted_info = str(information_dict)
        
        # 如果是回退搜索，添加药物描述前缀
        if is_fallback:
            fall_back_title = information_dict.get('description', '')
            extracted_info = f"Description of Drug: {fall_back_title}, Related information: {extracted_info}"
        
        all_results.append(extracted_info)
    
    return {
        "keyword": keyword,
        "focus": focus,
        "search_strategy": search_strategy,
        "extracted_info": all_results,
        "data": all_results,  # For backward compatibility
        "error": None
    }


async def search_fda_drug_label_async(
    keyword: str,
    focus: str,
    limit: int = 5,
    use_llm_extraction: bool = True,
    api_key: Optional[str] = None,
    model_name: str = "bytedance-seed/seed-1.6",
) -> Dict[str, Any]:
    """
    异步版本：从 FDA API 获取药物信息（三级搜索策略）

    Args:
        keyword: 药物名称
        focus: 关注点（用于提取相关信息，如 "adverse reactions", "indications" 等）
        limit: 返回结果数量限制
        use_llm_extraction: 是否使用 LLM 提取相关信息
        api_key: OpenRouter API Key (用于 LLM 提取)
        model_name: LLM 模型名称

    Returns:
        Dictionary containing:
        - keyword: 搜索的药物名称
        - focus: 关注点
        - search_strategy: 成功的搜索策略 (BRAND_NAME, GENERIC_NAME, FINDALL_NAME)
        - extracted_info: 提取的相关信息列表
        - data: 原始 FDA 数据（如果不使用 LLM 提取）
        - error: 错误信息（如果搜索失败）
    """
    fda_api_key = "nwhIkQlmveH44at4S8uBKk94UWs1e7KC3A7nTNv8"

    # 优先搜索策略：品牌名和通用名
    priority_searches = [
        {
            "name": "BRAND_NAME",
            "base_url": "https://api.fda.gov/drug/label.json",
            "limit": 1,
            "search": f'(openfda.brand_name:("{keyword}"))',
            "api_key": fda_api_key
        },
        {
            "name": "GENERIC_NAME",
            "base_url": "https://api.fda.gov/drug/label.json",
            "limit": 1,
            "search": f'(openfda.generic_name:("{keyword}"))',
            "api_key": fda_api_key
        },
    ]

    # 回退搜索策略
    fallback_search = {
        "name": "FINDALL_NAME",
        "base_url": "https://api.fda.gov/drug/label.json",
        "limit": min(limit, 3),
        "search": f'"{keyword}"',
        "api_key": fda_api_key
    }

    raw_results = None
    search_strategy = None
    proxy = _get_proxy()

    # 使用 aiohttp 进行异步 HTTP 请求
    async with aiohttp.ClientSession() as session:
        # 先尝试优先搜索策略
        for search_config in priority_searches:
            try:
                max_retries = 3
                retry_delay = 1
                query_params = {k: v for k, v in search_config.items() if k not in ['name', 'base_url']}
                query_str = urlencode(query_params, quote_via=quote)
                url = f"{search_config['base_url']}?{query_str}"

                for attempt in range(max_retries):
                    try:
                        async with session.get(
                            url, timeout=aiohttp.ClientTimeout(total=60), proxy=proxy
                        ) as response:
                            if response.status == 200:
                                result_data = await response.json()

                                if 'results' in result_data and result_data['results']:
                                    raw_results = result_data
                                    search_strategy = search_config['name']
                                    print(f"✅ {search_config['name']} 成功找到结果")
                                    break
                            else:
                                raise aiohttp.ClientError(f"HTTP {response.status}")

                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        if attempt < max_retries - 1:
                            print(f"⚠️ {search_config['name']} 失败 ({type(e).__name__}: {e})，{retry_delay}秒后重试... (proxy={proxy})")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2

                if raw_results:
                    break

            except Exception as e:
                print(f"❌ {search_config['name']} 出错: {e}")
                continue

        # 如果优先搜索没有找到结果，使用回退搜索
        if not raw_results:
            print("⚠️ 优先搜索未找到结果，使用回退搜索...")
            try:
                max_retries = 3
                retry_delay = 1
                query_params = {k: v for k, v in fallback_search.items() if k not in ['name', 'base_url']}
                query_str = urlencode(query_params, quote_via=quote)
                url = f"{fallback_search['base_url']}?{query_str}"

                for attempt in range(max_retries):
                    try:
                        async with session.get(
                            url, timeout=aiohttp.ClientTimeout(total=60), proxy=proxy
                        ) as response:
                            if response.status == 200:
                                result_data = await response.json()

                                if 'results' in result_data and result_data['results']:
                                    raw_results = result_data
                                    search_strategy = fallback_search['name']
                                    print("✅ 回退搜索成功")
                                else:
                                    print("⚠️ 回退搜索也未找到结果")
                                break
                            else:
                                raise aiohttp.ClientError(f"HTTP {response.status}")

                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        if attempt < max_retries - 1:
                            print(f"⚠️ 回退搜索失败 ({type(e).__name__}: {e})，{retry_delay}秒后重试... (proxy={proxy})")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2

            except Exception as e:
                print(f"❌ 回退搜索出错: {e}")

    # 如果所有搜索都失败
    if not raw_results:
        return {
            "keyword": keyword,
            "focus": focus,
            "search_strategy": None,
            "extracted_info": [],
            "data": [],
            "error": f"No FDA data found for drug: {keyword}"
        }

    # 处理搜索结果
    results_list = raw_results['results']
    is_fallback = search_strategy == "FINDALL_NAME"

    # 准备待处理的数据
    related_parts = []
    if is_fallback:
        # 回退搜索：返回多个完整的结果
        related_parts = results_list[:limit]
    else:
        # 精确搜索：提取所有字段（排除 table 字段）
        for information_dict in results_list:
            for key, value in information_dict.items():
                if "_table" not in key:
                    related_parts.append({key: value})
        related_parts = [str(related_parts)]

    # 如果不使用 LLM 提取，直接返回原始数据
    if not use_llm_extraction:
        return {
            "keyword": keyword,
            "focus": focus,
            "search_strategy": search_strategy,
            "extracted_info": [],
            "data": related_parts,
            "error": None
        }

    # 使用 LLM 提取相关信息（异步）
    all_results = []
    for information_dict in related_parts:
        # 构造提示词
        prompt = f"You need to extract information from FDA drug data based on a specific goal.\n\n"
        prompt += f"Drug/Keyword: {keyword}\n"
        prompt += f"Goal: {focus}\n\n"
        prompt += f"Instructions:\n"
        prompt += f"- Extract ONLY the information directly relevant to the goal\n"
        prompt += f"- Use original text from the data (do not paraphrase or add explanations)\n"
        prompt += f"- Combine relevant excerpts into a coherent response\n"
        prompt += f"- If no relevant information is found, return 'No relevant information found'\n\n"

        if is_fallback:
            prompt += f"Drug description: {information_dict.get('description', '')}\n\n"

        prompt += f"FDA Data:\n{information_dict}\n\n"
        prompt += f"Extracted information based on the goal:"

        # 异步调用 LLM 提取信息
        extracted_info = await _call_openrouter_llm_async(
            prompt=prompt,
            api_key=api_key,
            model_name=model_name,
            max_retries=3
        )

        # 如果 LLM 调用失败，使用原始数据
        if not extracted_info:
            extracted_info = str(information_dict)

        # 如果是回退搜索，添加药物描述前缀
        if is_fallback:
            fall_back_title = information_dict.get('description', '')
            extracted_info = f"Description of Drug: {fall_back_title}, Related information: {extracted_info}"

        all_results.append(extracted_info)

    return {
        "keyword": keyword,
        "focus": focus,
        "search_strategy": search_strategy,
        "extracted_info": all_results,
        "data": all_results,  # For backward compatibility
        "error": None
    }


if __name__ == "__main__":
    # 测试
    result = search_fda_drug_label(
        keyword="aspirin",
        focus="adverse reactions",
        limit=3,
        use_llm_extraction=True
    )
    
    print(f"\n搜索策略: {result['search_strategy']}")
    print(f"提取的信息:")
    for i, info in enumerate(result['extracted_info'], 1):
        print(f"\n{i}. {info[:500]}...")

