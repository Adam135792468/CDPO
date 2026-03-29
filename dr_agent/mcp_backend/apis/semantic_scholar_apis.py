import os
import warnings
from pathlib import Path
from typing import Generic, List, Optional, TypeVar, Union
import time

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, HttpUrl
from requests.exceptions import RequestException

from ..cache import cached
from .data_model import (
    PaperSnippet,
    SemanticScholarAuthorData,
    SemanticScholarPaperData,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

S2_API_KEY = os.getenv("S2_API_KEY")
TIMEOUT = int(os.getenv("API_TIMEOUT", 10))
S2_SEARCH_LIMIT = 100

# 免费调用使用官方地址
S2_GRAPH_API_URL_FREE = "https://api.semanticscholar.org/graph/v1"
S2_RECOMMENDATIONS_API_URL_FREE = "https://api.semanticscholar.org/recommendations/v1"

# 付费调用使用代理地址
S2_GRAPH_API_URL_PAID = "https://ai4scholar.net/graph/v1"
S2_RECOMMENDATIONS_API_URL_PAID = "https://ai4scholar.net/recommendations/v1"

# 默认使用付费地址（向后兼容）
S2_GRAPH_API_URL = S2_GRAPH_API_URL_PAID
S2_RECOMMENDATIONS_API_URL = S2_RECOMMENDATIONS_API_URL_PAID

# 免费调用重试次数
FREE_RETRY_ATTEMPTS = 3

# authors.authorId,authors.paperCount,authors.citationCount
S2_PAPER_SEARCH_FIELDS = "paperId,corpusId,url,title,abstract,authors,authors.name,year,venue,citationCount,openAccessPdf,externalIds,isOpenAccess"
S2_PAPER_CITATION_FIELDS = (
    "paperId,corpusId,contexts,intents,isInfluential,title,abstract,venue,year,authors"
)
S2_PAPER_REFERENCE_FIELDS = (
    "paperId,corpusId,contexts,intents,isInfluential,title,abstract,venue,year,authors"
)
S2_PAPER_RECOMMENDATION_FIELDS = "paperId,corpusId,title,abstract,year,venue"

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    total: int
    offset: int = Field(..., description="Offset for pagination")
    next: int = Field(..., description="Next page offset")
    data: List[T] = Field(..., description="Data items")


# This is because the Semantic Scholar API right now doesn't support
# pagination for snippets, so we will ignore the offset and limit.
class PaperSnippetApiResponse(BaseModel, Generic[T]):
    data: List[T] = Field(..., description="Data items")


# This is a subset of the supported query parameters for the Semantic Scholar API.
# We include the most important ones and remove some for clarity. (e.g., there are year and
# publicationDateOrYear, which can be confusing to the model).
def _make_request_with_retry(
    url: str,
    params: dict,
    timeout: int,
    method: str = "GET",
    json_data: dict = None,
) -> dict:
    """
    先尝试免费调用（访问官方地址，不带 API key），重试 3 次。
    如果都失败，再使用付费调用（访问代理地址，带 API key）。
    """
    # 将付费地址的 URL 替换为免费地址
    free_url = url.replace(S2_GRAPH_API_URL_PAID, S2_GRAPH_API_URL_FREE)
    free_url = free_url.replace(S2_RECOMMENDATIONS_API_URL_PAID, S2_RECOMMENDATIONS_API_URL_FREE)
    
    # 获取代理设置（从环境变量）
    # requests 会自动从 http_proxy/https_proxy 环境变量读取
    # 但为了确保正确，我们显式设置
    proxies = None
    if os.getenv('http_proxy') or os.getenv('https_proxy'):
        proxies = {
            'http': os.getenv('http_proxy'),
            'https': os.getenv('https_proxy'),
        }
        print(f"🌐 使用代理: {proxies}")
    
    def _is_valid_response(response_json: dict) -> bool:
        """检查响应是否有效（不是错误消息）"""
        # 如果响应只包含 'message' 字段，通常是错误
        if 'message' in response_json and len(response_json) == 1:
            return False
        # 如果包含 'error' 字段，是错误
        if 'error' in response_json:
            return False
        # 其他情况认为是有效响应
        return True
    
    # 先尝试免费调用（使用官方地址）
    for attempt in range(FREE_RETRY_ATTEMPTS):
        try:
            if method.upper() == "POST":
                res = requests.post(
                    free_url,  # 使用免费地址
                    params=params,
                    json=json_data,
                    headers=None,  # 不带 API key
                    timeout=timeout,
                    proxies=proxies,  # 使用代理
                )
            else:
                res = requests.get(
                    free_url,  # 使用免费地址
                    params=params,
                    headers=None,  # 不带 API key
                    timeout=timeout,
                    proxies=proxies,  # 使用代理
                )
            
            res.raise_for_status()
            result = res.json()
            
            # 检查响应内容是否有效
            if not _is_valid_response(result):
                error_msg = result.get('message', result.get('error', 'Unknown error'))
                raise Exception(f"API 返回错误: {error_msg}")
            
            print(f"✓ 免费调用成功 (尝试 {attempt + 1}/{FREE_RETRY_ATTEMPTS}) - 地址: {free_url}")
            return result
        
        except Exception as e:
            print(f"✗ 免费调用失败 (尝试 {attempt + 1}/{FREE_RETRY_ATTEMPTS}): {str(e)}")
            if attempt < FREE_RETRY_ATTEMPTS - 1:
                time.sleep(1)  # 重试前等待 1 秒
            continue
    
    # 所有免费调用都失败，使用付费调用（使用代理地址）
    print(f"所有免费调用失败，切换到付费调用 - 地址: {url}")
    try:
        if method.upper() == "POST":
            res = requests.post(
                url,  # 使用付费地址（原 URL）
                params=params,
                json=json_data,
                # headers={"x-api-key": S2_API_KEY} if S2_API_KEY else None,
                headers = {'Authorization': f'Bearer {S2_API_KEY}'},
                timeout=timeout,
                proxies=proxies,  # 使用代理
            )
        else:
            res = requests.get(
                url,  # 使用付费地址（原 URL）
                params=params,
                # headers={"x-api-key": S2_API_KEY} if S2_API_KEY else None,
                headers = {'Authorization': f'Bearer {S2_API_KEY}'},
                timeout=timeout,
                proxies=proxies,  # 使用代理
            )
        
        res.raise_for_status()
        result = res.json()
        
        # 检查付费调用的响应是否有效
        if not _is_valid_response(result):
            error_msg = result.get('message', result.get('error', 'Unknown error'))
            raise Exception(f"API 返回错误: {error_msg}")
        
        print("✓ 付费调用成功")
        return result
    
    except Exception as e:
        print(f"✗ 付费调用也失败: {str(e)}")
        raise


class SemanticScholarSearchQueryParams(BaseModel):
    query: str = Field(
        ...,
        description="A plain-text search query string.",
        examples=["BERT"],
    )
    year: Optional[str] = Field(
        None,
        description="Restrict results to the given range of publication year (inclusive).",
        examples=["2015-2020", "2015-", "-2015"],
    )
    minCitationCount: Optional[int] = Field(
        None,
        description="Restrict results to only include papers with the minimum number of citations, inclusive.",
        examples=[100, 1000],
    )
    sort: Optional[str] = Field(
        None,
        description="Sort results by publicationDate and citationCount in ascending or descending order.",
        examples=[
            "citationCount:asc",
            "publicationDate:desc",
        ],
    )
    venue: Optional[str] = Field(
        None,
        description="Restrict results by venue. Input could be an ISO4 abbreviation.",
        examples=["ACL", "EMNLP"],
    )


################## NOT EXPLICITLY USED ##################
# fields: Optional[str] = Field(None, description="A comma-separated list of the fields to be returned.")
# publicationTypes: Optional[str] = Field(None, description="Restrict results by publication types, use a comma-separated list.")
# openAccessPdf: Optional[bool] = Field(None, description="Restrict results to only include papers with a public PDF.")
# publicationDateOrYear: Optional[str] = Field(None, description="""
# Restrict results to the given range of publication dates or years (inclusive).
# Accepts the format <startDate>:<endDate>. Prefixes supported for specific ranges.""")
# fieldsOfStudy: Optional[str] = Field(None, desc="Restrict results to given field-of-study, using the s2FieldsOfStudy paper field.")
# offset: Optional[int] = Field(0, description="Start with the element at this position in the list.")
# limit: Optional[int] = Field(100, description="The maximum number of results to return, must be <= 100.")


@cached()
def search_semantic_scholar_keywords(
    query_params: SemanticScholarSearchQueryParams,
    *,
    offset: int = 0,
    limit: int = 25,
    fields: str = S2_PAPER_SEARCH_FIELDS,
    timeout: int = TIMEOUT,
) -> ApiResponse[SemanticScholarPaperData]:

    results = _make_request_with_retry(
        url=f"{S2_GRAPH_API_URL}/paper/search",
        params={
            "offset": offset,
            "limit": limit,
            "fields": fields,
            **query_params.model_dump(exclude_none=True),
        },
        timeout=timeout,
        method="GET",
    )

    # For each paper, if we know their external arxiv ids, we can construct the open access pdf link
    # if it is not already provided.
    if "data" in results:
        for paper in results["data"]:
            if paper.get("openAccessPdf") is None:
                if paper["externalIds"]:
                    if "ArXiv" in paper["externalIds"]:
                        paper["openAccessPdf"] = dict(
                            url=f"https://arxiv.org/pdf/{paper['externalIds']['ArXiv']}",
                        )
                    if "ACL" in paper["externalIds"]:
                        paper["openAccessPdf"] = dict(
                            url=f"https://www.aclweb.org/anthology/{paper['externalIds']['ACL']}.pdf",
                        )
    else:
        results["data"] = []

    return results


class SemanticScholarSnippetSearchQueryParams(BaseModel):
    query: str = Field(
        ...,
        description="A plain-text search query string.",
        examples=["BERT"],
    )
    year: Optional[str] = Field(
        None,
        description="Restrict results to the given range of publication year (inclusive).",
        examples=["2015-2020", "2015-", "-2015"],
    )
    paperIds: Optional[Union[str, List[str]]] = Field(
        None,
        description="Restricts results to snippets from specific papers. To specify papers, provide a comma-separated list of their IDs. You can provide up to approximately 100 IDs.",
        examples=[
            "649def34f8be52c8b66281af98ae884c09aef38b",
            "649def34f8be52c8b66281af98ae884c09aef38b,CorpusId:215416146",
        ],
    )
    venue: Optional[str] = Field(
        None,
        description="Restrict results by venue. Input could be an ISO4 abbreviation.",
        examples=["ACL", "EMNLP"],
    )


@cached()
def search_semantic_scholar_snippets(
    query_params: SemanticScholarSnippetSearchQueryParams,
    *,
    offset: int = 0,
    limit: int = 10,
    timeout: int = TIMEOUT,
) -> PaperSnippetApiResponse[PaperSnippet]:
    if offset:
        warnings.warn(
            "Right now the API does not support pagination, so the offset will be ignored."
        )

    params = query_params.model_dump(exclude_none=True)
    if (query_params.paperIds or "paperIds" in query_params) and isinstance(
        query_params.paperIds, list
    ):
        params["paperIds"] = ",".join(query_params.paperIds)

    results = _make_request_with_retry(
        url=f"{S2_GRAPH_API_URL}/snippet/search",
        params={
            # "offset": offset,
            "limit": limit,
            **params,
        },
        timeout=timeout,
        method="GET",
    )
    return results


def search_semantic_scholar_bulk_api(
    query_params: SemanticScholarSearchQueryParams,
    *,
    fields: str = S2_PAPER_SEARCH_FIELDS,
    timeout: int = TIMEOUT,
) -> ApiResponse[SemanticScholarPaperData]:

    results = _make_request_with_retry(
        url=f"{S2_GRAPH_API_URL}/paper/search/bulk",
        params={
            "fields": fields,
            **query_params.model_dump(exclude_none=True),
        },
        timeout=timeout,
        method="GET",
    )

    # For each paper, if we know their external arxiv ids, we can construct the open access pdf link
    # if it is not already provided.
    if "data" in results:
        for paper in results["data"]:
            if paper.get("openAccessPdf") is None:
                if paper["externalIds"]:
                    if "ArXiv" in paper["externalIds"]:
                        paper["openAccessPdf"] = dict(
                            url=f"https://arxiv.org/pdf/{paper['externalIds']['ArXiv']}",
                        )
                    if "ACL" in paper["externalIds"]:
                        paper["openAccessPdf"] = dict(
                            url=f"https://www.aclweb.org/anthology/{paper['externalIds']['ACL']}.pdf",
                        )
    else:
        results["data"] = []

    return results


def download_paper_details(
    paper_id: str,
    *,
    fields: str = S2_PAPER_SEARCH_FIELDS,
    timeout: int = TIMEOUT,
):
    results = _make_request_with_retry(
        url=f"{S2_GRAPH_API_URL}/paper/{paper_id}",
        params={
            "fields": fields,
        },
        timeout=timeout,
        method="GET",
    )
    return results


def download_paper_references(
    paper_id: str,
    *,
    offset: int = 0,
    limit: int = 100,
    fields: str = S2_PAPER_REFERENCE_FIELDS,
    timeout: int = TIMEOUT,
):
    results = _make_request_with_retry(
        url=f"{S2_GRAPH_API_URL}/paper/{paper_id}/references",
        params={
            "offset": offset,
            "limit": limit,
            "fields": fields,
        },
        timeout=timeout,
        method="GET",
    )
    return results


def download_paper_citations(
    paper_id: str,
    *,
    offset: int = 0,
    limit: int = 100,
    fields: str = S2_PAPER_CITATION_FIELDS,
    timeout: int = TIMEOUT,
):
    results = _make_request_with_retry(
        url=f"{S2_GRAPH_API_URL}/paper/{paper_id}/citations",
        params={
            "offset": offset,
            "limit": limit,
            "fields": fields,
        },
        timeout=timeout,
        method="GET",
    )
    return results


def download_paper_details_batch(
    paper_ids: List[str],
    *,
    fields: str = S2_PAPER_SEARCH_FIELDS,
    timeout: int = TIMEOUT,
):
    results = _make_request_with_retry(
        url=f"{S2_GRAPH_API_URL}/paper/batch",
        params={"fields": fields},
        timeout=timeout,
        method="POST",
        json_data={"ids": paper_ids},
    )
    return results


# Example usage:
if __name__ == "__main__":
    result1 = search_semantic_scholar_snippets(
        SemanticScholarSnippetSearchQueryParams(
            query="how to set learning rate in state space models?",
        ),
    )
