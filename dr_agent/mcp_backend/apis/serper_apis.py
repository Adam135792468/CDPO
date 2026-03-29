import json
import logging
import os
import threading
from typing import Callable, Dict, List, Optional, TypeVar, Union

import dotenv
import requests
from typing_extensions import TypedDict

from ..cache import cached

# Load environment variables
dotenv.load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
TIMEOUT = int(os.getenv("API_TIMEOUT", 10))

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SerperAPIKeyManager:
    """
    管理多个Serper API key，当一个key用完时自动切换到下一个。

    API key配置方式：
    1. 环境变量 SERPER_API_KEY: 单个key
    2. 环境变量 SERPER_API_KEYS: 多个key，用逗号分隔

    当API返回403（禁止访问，通常是额度用完）或429（请求过多）时，
    会自动切换到下一个key重试。
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._api_keys: List[str] = []
        self._current_index = 0
        self._key_lock = threading.Lock()
        self._exhausted_keys: set = set()  # 记录已用完的key
        self._load_api_keys()
        self._initialized = True

    def _load_api_keys(self):
        """从环境变量加载API keys"""
        # 优先从 SERPER_API_KEYS 加载多个key
        keys_str = os.getenv("SERPER_API_KEYS", "")
        if keys_str:
            self._api_keys = [k.strip() for k in keys_str.split(",") if k.strip()]

        # 如果没有多个key，尝试加载单个key
        if not self._api_keys:
            single_key = os.getenv("SERPER_API_KEY", "")
            if single_key:
                self._api_keys = [single_key]

        if self._api_keys:
            logger.info(f"Loaded {len(self._api_keys)} Serper API key(s)")
        else:
            logger.warning("No Serper API keys found in environment variables")

    def get_current_key(self) -> Optional[str]:
        """获取当前可用的API key"""
        with self._key_lock:
            if not self._api_keys:
                return None

            # 如果所有key都用完了，重置（可能key已经恢复额度）
            if len(self._exhausted_keys) >= len(self._api_keys):
                logger.warning("All API keys exhausted, resetting exhausted list")
                self._exhausted_keys.clear()

            # 找到一个未用完的key
            for _ in range(len(self._api_keys)):
                key = self._api_keys[self._current_index]
                if key not in self._exhausted_keys:
                    return key
                self._current_index = (self._current_index + 1) % len(self._api_keys)

            # 所有key都用完了，返回第一个（让调用方决定如何处理）
            return self._api_keys[0]

    def mark_key_exhausted(self, key: str):
        """标记一个key已用完额度"""
        with self._key_lock:
            self._exhausted_keys.add(key)
            logger.warning(f"API key exhausted: {key[:8]}... (exhausted: {len(self._exhausted_keys)}/{len(self._api_keys)})")
            # 切换到下一个key
            self._current_index = (self._current_index + 1) % len(self._api_keys)

    def get_available_key_count(self) -> int:
        """获取可用的key数量"""
        with self._key_lock:
            return len(self._api_keys) - len(self._exhausted_keys)

    def add_key(self, key: str):
        """动态添加新的API key"""
        with self._key_lock:
            if key and key not in self._api_keys:
                self._api_keys.append(key)
                logger.info(f"Added new API key: {key[:8]}...")


# 全局单例
_api_key_manager = SerperAPIKeyManager()


class QuotaExceededError(Exception):
    """Serper API key 额度用完时抛出的异常"""
    def __init__(self, message: str, api_key: str, status_code: int):
        super().__init__(message)
        self.api_key = api_key
        self.status_code = status_code


def _is_quota_exceeded_response(status_code: int, response_text: str) -> bool:
    """判断HTTP响应是否表示额度用完"""
    # 400: Bad Request - Serper 额度用完时返回 400 + "Not enough credits"
    # 403: Forbidden - 额度用完的另一种表现
    # 429: Too Many Requests - 请求频率限制或额度用完
    if status_code in (400, 403, 429):
        lower_text = response_text.lower()
        quota_keywords = ["quota", "limit", "exceeded", "exhausted", "credits", "insufficient", "not enough"]
        if any(keyword in lower_text for keyword in quota_keywords):
            return True
        # 403 和 429 本身就说明额度/限流问题
        if status_code in (403, 429):
            return True

    return False


def _check_response_quota(response, api_key: str):
    """检查API响应，如果是额度用完则抛出QuotaExceededError"""
    if response.status_code != 200:
        if _is_quota_exceeded_response(response.status_code, response.text):
            raise QuotaExceededError(
                f"API key {api_key[:8]}... quota exceeded (HTTP {response.status_code}): {response.text[:200]}",
                api_key=api_key,
                status_code=response.status_code,
            )
        # 非额度错误，抛出普通异常
        raise Exception(
            f"API request failed with status {response.status_code}: {response.text}"
        )


def _call_with_key_rotation(
    api_func: Callable[..., T],
    api_key: Optional[str] = None,
    **kwargs
) -> T:
    """
    执行API调用，如果key用完则自动轮换到下一个key重试。

    Args:
        api_func: 实际执行API调用的函数
        api_key: 指定的API key（如果提供则不使用轮换机制）
        **kwargs: 传递给api_func的其他参数

    Returns:
        API调用的结果
    """
    # 如果调用方指定了key，直接使用不轮换
    if api_key:
        return api_func(api_key=api_key, **kwargs)

    manager = _api_key_manager
    tried_keys = set()
    last_error = None

    while True:
        current_key = manager.get_current_key()

        if not current_key:
            raise ValueError(
                "SERPER_API_KEY or SERPER_API_KEYS environment variable is not set"
            )

        # 避免无限循环：如果已经尝试过所有key
        if current_key in tried_keys:
            if last_error:
                raise last_error
            raise ValueError("All API keys have been exhausted")

        tried_keys.add(current_key)

        try:
            return api_func(api_key=current_key, **kwargs)
        except QuotaExceededError as e:
            # 额度用完，标记key并切换到下一个
            logger.warning(f"API key quota exceeded, switching to next key. Key: {e.api_key[:8]}... Status: {e.status_code}")
            manager.mark_key_exhausted(e.api_key)
            last_error = e
            continue
        except Exception:
            # 其他错误直接抛出，不重试
            raise


class KnowledgeGraph(TypedDict, total=False):
    title: str
    type: str
    website: str
    imageUrl: str
    description: str
    descriptionSource: str
    descriptionLink: str
    attributes: Optional[Dict[str, str]]


class Sitelink(TypedDict):
    title: str
    link: str


class SearchResult(TypedDict):
    title: str
    link: str
    snippet: str
    position: int
    sitelinks: Optional[List[Sitelink]]
    attributes: Optional[Dict[str, str]]
    date: Optional[str]


class PeopleAlsoAsk(TypedDict):
    question: str
    snippet: str
    title: str
    link: str


class RelatedSearch(TypedDict):
    query: str


class SearchResponse(TypedDict, total=False):
    searchParameters: Dict[str, Union[str, int, bool]]
    knowledgeGraph: Optional[KnowledgeGraph]
    organic: List[SearchResult]
    peopleAlsoAsk: Optional[List[PeopleAlsoAsk]]
    relatedSearches: Optional[List[RelatedSearch]]


class ScholarResult(TypedDict):
    title: str
    link: str
    publicationInfo: str
    snippet: str
    year: Union[int, str]
    citedBy: int


class ScholarResponse(TypedDict):
    searchParameters: Dict[str, Union[str, int, bool]]
    organic: List[ScholarResult]


class WebpageContentResponse(TypedDict, total=False):
    url: str
    text: str
    markdown: str
    metadata: Dict[str, Union[str, int, bool]]
    credits: int


def _search_serper_impl(
    query: str,
    num_results: int,
    gl: str,
    hl: str,
    search_type: str,
    api_key: str,
) -> SearchResponse:
    """内部实现函数，执行实际的API调用"""
    url = "https://google.serper.dev/search"

    payload = json.dumps({"q": query, "num": num_results, "gl": gl, "hl": hl, "type": search_type})

    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, data=payload)
        _check_response_quota(response, api_key)
        return response.json()

    except QuotaExceededError:
        raise  # 让 _call_with_key_rotation 处理
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error performing Serper search: {str(e)}")


@cached()
def search_serper(
    query: str,
    num_results: int = 10,
    gl: str = "us",
    hl: str = "en",
    search_type: str = "search",  # Can be "search", "places", "news", "images"
    api_key: str = None,
) -> SearchResponse:
    """
    Search using Serper.dev API for general web search.
    支持多API key自动轮换：当一个key额度用完时自动切换到下一个key重试。

    Args:
        query: Search query string
        num_results: Number of results to return (default: 10)
        gl: Country code to boosts search results whose country of origin matches the parameter value (default: us)
        hl: Host language of user interface (default: en)
        search_type: Type of search to perform (default: "search")
                    Options: "search", "places", "news", "images"
        api_key: Serper API key (if not provided, will use key rotation from SERPER_API_KEYS or SERPER_API_KEY env var)

    Returns:
        SearchResponse containing:
        - searchParameters: Dict with search metadata
        - knowledgeGraph: Optional knowledge graph information
        - organic: List of organic search results
        - peopleAlsoAsk: Optional list of related questions
        - relatedSearches: Optional list of related search queries
    """
    return _call_with_key_rotation(
        _search_serper_impl,
        api_key=api_key,
        query=query,
        num_results=num_results,
        gl=gl,
        hl=hl,
        search_type=search_type,
    )


def _search_serper_scholar_impl(
    query: str,
    num_results: int,
    api_key: str,
) -> ScholarResponse:
    """内部实现函数，执行实际的API调用"""
    url = "https://google.serper.dev/scholar"

    payload = json.dumps({"q": query, "num": num_results})

    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, data=payload)
        _check_response_quota(response, api_key)
        return response.json()

    except QuotaExceededError:
        raise  # 让 _call_with_key_rotation 处理
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error performing Serper scholar search: {str(e)}")


@cached()
def search_serper_scholar(
    query: str,
    num_results: int = 10,
    api_key: str = None,
) -> ScholarResponse:
    """
    Search academic papers using Serper.dev Scholar API.
    支持多API key自动轮换：当一个key额度用完时自动切换到下一个key重试。

    Args:
        query: Academic search query string
        num_results: Number of results to return (default: 10)
        api_key: Serper API key (if not provided, will use key rotation from SERPER_API_KEYS or SERPER_API_KEY env var)

    Returns:
        ScholarResponse containing:
        - organic: List of academic paper results with:
            - title: Paper title
            - link: URL to the paper
            - publicationInfo: Author and publication details
            - snippet: Brief excerpt from the paper
            - year: Publication year
            - citedBy: Number of citations
    """
    return _call_with_key_rotation(
        _search_serper_scholar_impl,
        api_key=api_key,
        query=query,
        num_results=num_results,
    )


def _fetch_webpage_content_impl(
    url: str,
    include_markdown: bool,
    api_key: str,
) -> WebpageContentResponse:
    """内部实现函数，执行实际的API调用"""
    scrape_url = "https://scrape.serper.dev"

    payload = json.dumps({"url": url, "includeMarkdown": include_markdown})

    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    try:
        response = requests.post(scrape_url, headers=headers, data=payload)
        _check_response_quota(response, api_key)
        data = response.json()
        data["url"] = url
        return data

    except QuotaExceededError:
        raise  # 让 _call_with_key_rotation 处理
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching webpage content: {str(e)}")
    except json.JSONDecodeError as e:
        raise Exception(f"Error parsing API response: {str(e)}")


@cached()
def fetch_webpage_content(
    url: str,
    include_markdown: bool = True,
    api_key: str = None,
) -> WebpageContentResponse:
    """
    Fetch the content of a webpage using Serper.dev API.
    支持多API key自动轮换：当一个key额度用完时自动切换到下一个key重试。

    Args:
        url: The URL of the webpage to fetch
        include_markdown: Whether to include markdown formatting in the response (default: True)
        api_key: Serper API key (if not provided, will use key rotation from SERPER_API_KEYS or SERPER_API_KEY env var)

    Returns:
        WebpageContentResponse containing:
        - text: The webpage content as plain text
        - markdown: The webpage content formatted as markdown (if include_markdown=True)
        - metadata: Additional metadata about the webpage
    """
    return _call_with_key_rotation(
        _fetch_webpage_content_impl,
        api_key=api_key,
        url=url,
        include_markdown=include_markdown,
    )


# Example usage:
if __name__ == "__main__":
    # Regular search example
    try:
        results = search_serper("apple inc", num_results=5)
        print("Regular Search Results:")
        print(f"Found {len(results.get('organic', []))} results")
        if "knowledgeGraph" in results:
            print(f"Knowledge Graph: {results['knowledgeGraph']['title']}")
        print()
    except Exception as e:
        print(f"Search error: {e}")

    # Scholar search example
    try:
        scholar_results = search_serper_scholar(
            "attention is all you need", num_results=5
        )
        print("Scholar Search Results:")
        print(f"Found {len(scholar_results.get('organic', []))} academic papers")
        for paper in scholar_results.get("organic", [])[:2]:
            print(
                f"- {paper['title']} ({paper['year']}) - Cited by: {paper['citedBy']}"
            )
        print()
    except Exception as e:
        print(f"Scholar search error: {e}")
