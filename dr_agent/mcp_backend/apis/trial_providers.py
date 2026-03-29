# -*- coding: utf-8 -*-
"""
真实数据提供者模块
从ClinicalTrials.gov API和FDA Orange Book获取真实数据
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Protocol, Tuple, Iterable
import os
import sys
import io
import re
import zipfile
import csv
import hashlib
import time
import fcntl
import requests
from requests.adapters import HTTPAdapter, Retry
from dateutil import parser as dtparser
from bs4 import BeautifulSoup

# ---------- 常量 ----------
CTGOV_STUDY_URL = "https://clinicaltrials.gov/api/v2/studies/{}"
CTGOV_PAGE_URL = "https://clinicaltrials.gov/study/{}"

# 官方页（主数据页 + Orange Book 主页，两者任一可解析出下载链接）
ORANGE_BOOK_PAGE = "https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files"
ORANGE_BOOK_MAIN = "https://www.fda.gov/drugs/drug-approvals-and-databases/approved-drug-products-therapeutic-equivalence-evaluations-orange-book"

# 允许的下载链接模式：绝对或相对
MEDIA_ABS_RE = re.compile(r'https?://www\.fda\.gov/media/\d+/download(?:\?[^"\s>]*)?', re.I)
MEDIA_REL_RE = re.compile(r'^/media/\d+/download(?:\?[^"\s>]*)?$', re.I)

# 缓存配置
# 默认缓存目录：使用共享挂载路径（GPU和CPU机器都可以访问）
_DEFAULT_CACHE_DIR = "/mnt/bn/med-mllm-lfv2/linjh/project/rl-nips-cure/work/exp7_tool_eval_infra/assets/cache"
# 支持环境变量覆盖（如果路径不同，可以通过环境变量指定）
CACHE_DIR = os.getenv("ORANGE_BOOK_CACHE_DIR", _DEFAULT_CACHE_DIR)
CACHE_MAX_AGE = 24 * 60 * 60 * 365  # 缓存有效期：一年
MAX_RETRIES = 3  # 最大重试次数
RETRY_DELAY = 2  # 重试延迟（秒）

# ---------- 数据模型 ----------
@dataclass
class Patent:
    jurisdiction: str
    number: str
    expiry_date: Optional[str]
    notes: Optional[str] = None

@dataclass
class Exclusivity:
    region: str
    type: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    notes: Optional[str] = None

@dataclass
class Approval:
    region: str
    product_name: str
    active_ingredient: str
    approval_date: Optional[str] = None
    status: Optional[str] = "Approved"
    marketing_authorisation_holder: Optional[str] = None

@dataclass
class TrialCommercialProfile:
    nct_id: str
    ingredients: List[str]
    sponsor: str
    is_open_for_enrollment: bool
    recruitment_status: str
    patents: List[Patent]
    exclusivities: List[Exclusivity]
    approvals: List[Approval]
    sources: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ---------- Provider 接口 ----------
class TrialProvider(Protocol):
    def can_handle(self, nct_id: str) -> bool: ...
    def fetch_basic_trial(self, nct_id: str) -> Dict[str, Any]: ...

class DrugIntelProvider(Protocol):
    def resolve_by_ingredients(self, ingredients: List[str]) -> Dict[str, Any]: ...

# ---------- 工具函数 ----------
def _session_with_retry() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3, 
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    )
    s.headers.update({
        "User-Agent": "trial-profile-bot/1.1 (+github.com/your-org) Python-requests"
    })
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def _normalize_date(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    try:
        return dtparser.parse(s).date().isoformat()
    except Exception:
        return s

def _salt_stripper(name: str) -> str:
    # 只保留真正的盐类/酸根后缀，不包含元素名（magnesium、sodium、potassium、calcium）
    salts = {
        "hydrochloride", "sulfate", "phosphate", "mesylate", "maleate", "bitartrate",
        "acetate", "fumarate", "succinate", "tosylate", "besylate", "citrate", "oxalate"
    }
    x = name.lower()
    x = re.sub(r"[()]", " ", x)
    tokens = [t for t in re.split(r"[,\s;]+", x) if t]
    tokens = [t for t in tokens if t not in salts]
    return " ".join(tokens)

def _match_ob_ingredient(targets: Iterable[str], ob_ingredient: str) -> bool:
    ob_clean = _salt_stripper(ob_ingredient)
    # 如果剥离后为空字符串，回退到原始匹配（避免空字符串匹配所有内容）
    if not ob_clean:
        ob_clean = re.sub(r"[,\s;]+", " ", ob_ingredient.lower()).strip()
    # 如果仍然为空，直接返回False
    if not ob_clean:
        return False
    
    for t in targets:
        t_clean = _salt_stripper(t)
        # 如果剥离后为空字符串，回退到原始匹配
        if not t_clean:
            t_clean = re.sub(r"[,\s;]+", " ", t.lower()).strip()
        # 如果仍然为空，跳过
        if not t_clean:
            continue
        # 只有当两个字符串都不为空时才进行匹配
        if t_clean == ob_clean or (t_clean in ob_clean and len(t_clean) >= 3) or (ob_clean in t_clean and len(ob_clean) >= 3):
            return True
    return False

def _guess_ingredients_from_intervention_names(names: List[str]) -> List[str]:
    out = []
    for nm in names:
        low = nm.lower()
        for suffix in [" hydrochloride", " sulfate", " phosphate", " tablets", " capsule", " capsules"]:
            if low.endswith(suffix):
                low = low[: -len(suffix)]
        low = low.replace(" (alecensa)", "").replace("alecensa", "alectinib")
        cleaned = "".join(ch for ch in low if ch.isalpha() or ch in "- ")
        cleaned = cleaned.strip()
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out

# ---------- ClinicalTrials.gov v2 ----------
class CtGovProvider:
    """从ClinicalTrials.gov API获取真实临床试验数据"""
    
    def __init__(self):
        self.session = _session_with_retry()
    
    def can_handle(self, nct_id: str) -> bool:
        return nct_id.startswith("NCT") and len(nct_id) == 11
    
    def fetch_basic_trial(self, nct_id: str) -> Dict[str, Any]:
        """获取临床试验基本信息"""
        r = self.session.get(CTGOV_STUDY_URL.format(nct_id), timeout=30)
        r.raise_for_status()
        data = r.json()
        psec = data.get("protocolSection", {})
        if not psec:
            raise ValueError(f"Study not found or invalid response: {nct_id}")
        
        status_mod = psec.get("statusModule", {}) or {}
        overall = status_mod.get("overallStatus") or ""
        
        sponsor_mod = psec.get("sponsorCollaboratorsModule", {}) or {}
        sponsor = sponsor_mod.get("leadSponsor", {}).get("name") \
               or sponsor_mod.get("leadSponsor", {}).get("class") or ""
        
        inter_mod = psec.get("armsInterventionsModule", {}) or {}
        interventions = inter_mod.get("interventions", []) or []
        names = []
        for itv in interventions:
            nm = itv.get("name")
            if nm:
                names.append(nm.strip())
        
        ingredients = _guess_ingredients_from_intervention_names(names)
        
        # 如果一个成分含有多个单词，则忽略
        ingredients = [ingredient for ingredient in ingredients if len(ingredient.split()) <= 1]
        names = [name for name in names if len(name.split()) <= 1]
        
        return {
            "ingredients": ingredients or names,
            "sponsor": sponsor,
            "recruitment_status": overall,
            "sources": [CTGOV_PAGE_URL.format(nct_id)]
        }

# ---------- 缓存工具函数 ----------
def _ensure_cache_dir():
    """确保缓存目录存在"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR

def _get_cache_key(url: str) -> str:
    """根据URL生成缓存键（文件名）"""
    return hashlib.md5(url.encode()).hexdigest() + ".zip"

def _get_cache_path(url: str) -> str:
    """获取缓存文件路径"""
    cache_dir = _ensure_cache_dir()
    cache_key = _get_cache_key(url)
    return os.path.join(cache_dir, cache_key)

def _get_lock_path(url: str) -> str:
    """获取锁文件路径"""
    lock_dir = _ensure_cache_dir()
    cache_key = _get_cache_key(url)
    return os.path.join(lock_dir, cache_key + ".lock")

def _is_cache_valid(cache_path: str) -> bool:
    """检查缓存是否有效（存在且未过期）"""
    if not os.path.exists(cache_path):
        return False
    try:
        age = time.time() - os.path.getmtime(cache_path)
        return age < CACHE_MAX_AGE
    except Exception:
        return False

def _save_to_cache(url: str, content: bytes) -> None:
    """保存内容到缓存"""
    cache_dir = _ensure_cache_dir()
    cache_path = os.path.join(cache_dir, _get_cache_key(url))
    try:
        with open(cache_path, 'wb') as f:
            f.write(content)
    except Exception:
        # 如果保存失败，不影响主流程，只是下次无法使用缓存
        pass

def _find_cache_files() -> List[str]:
    """查找缓存目录中的所有ZIP文件（不需要知道URL）"""
    if not os.path.exists(CACHE_DIR):
        return []
    try:
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.zip')]
        return [os.path.join(CACHE_DIR, f) for f in cache_files]
    except Exception:
        return []

def _load_from_cache(url: str) -> Optional[bytes]:
    """从缓存加载内容（需要知道URL）"""
    cache_path = os.path.join(CACHE_DIR, _get_cache_key(url))
    if _is_cache_valid(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                content = f.read()
                # 调试信息：仅在环境变量启用时打印
                if os.getenv("ORANGE_BOOK_DEBUG_CACHE"):
                    print(f"[缓存] 从缓存加载: {cache_path} ({len(content)} bytes)", file=sys.stderr)
                return content
        except Exception as e:
            # 调试信息：仅在环境变量启用时打印
            if os.getenv("ORANGE_BOOK_DEBUG_CACHE"):
                print(f"[缓存] 读取缓存文件失败: {cache_path}, 错误: {e}", file=sys.stderr)
            pass
    else:
        # 调试信息：仅在环境变量启用时打印
        if os.getenv("ORANGE_BOOK_DEBUG_CACHE"):
            if os.path.exists(cache_path):
                print(f"[缓存] 缓存文件已过期: {cache_path}", file=sys.stderr)
            else:
                print(f"[缓存] 缓存文件不存在: {cache_path}, 缓存目录: {CACHE_DIR}", file=sys.stderr)
    return None

def _load_from_cache_any() -> Optional[bytes]:
    """从缓存目录中加载任意ZIP文件（不需要知道URL，用于离线场景）"""
    cache_files = _find_cache_files()
    if not cache_files:
        if os.getenv("ORANGE_BOOK_DEBUG_CACHE"):
            print(f"[缓存] 缓存目录中没有ZIP文件: {CACHE_DIR}", file=sys.stderr)
        return None
    
    # 如果有多个缓存文件，使用第一个（通常只有一个）
    if len(cache_files) > 1:
        if os.getenv("ORANGE_BOOK_DEBUG_CACHE"):
            print(f"[缓存] 找到多个缓存文件，使用第一个: {cache_files[0]}", file=sys.stderr)
    
    cache_path = cache_files[0]
    if _is_cache_valid(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                content = f.read()
                # 调试信息：仅在环境变量启用时打印
                if os.getenv("ORANGE_BOOK_DEBUG_CACHE"):
                    print(f"[缓存] 从缓存加载: {cache_path} ({len(content)} bytes)", file=sys.stderr)
                return content
        except Exception as e:
            if os.getenv("ORANGE_BOOK_DEBUG_CACHE"):
                print(f"[缓存] 读取缓存文件失败: {cache_path}, 错误: {e}", file=sys.stderr)
            pass
    return None

def _acquire_lock(lock_path: str, timeout: int = 300) -> Optional[Any]:
    """获取文件锁（用于多进程同步）"""
    lock_file = None
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            lock_file = open(lock_path, 'w')
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_file
        except (IOError, OSError):
            # 锁被占用，等待后重试
            if lock_file:
                lock_file.close()
            time.sleep(0.5)
    return None

def _release_lock(lock_file: Any) -> None:
    """释放文件锁"""
    if lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
        except Exception:
            pass

# ---------- Orange Book 提供者 ----------
class OrangeBookProvider:
    """
    从FDA Orange Book ZIP文件获取真实药物数据
    
    解析 Orange Book ZIP：
      1) 先看环境变量 ORANGE_BOOK_ZIP_URL（若已指定则直接下载）
      2) 解析 ORANGE_BOOK_PAGE（数据文件页）找 "compressed (.ZIP) data file" 或 media/{id}/download
      3) 若失败，解析 ORANGE_BOOK_MAIN（主页）里的 "Orange Book Data Files" 跳转再抓
      4) 支持相对链接，允许附带 query 参数
      5) 支持缓存机制，避免重复下载
      6) 支持错误重试机制
    """
    
    def __init__(self, page_url: str = ORANGE_BOOK_PAGE, main_url: str = ORANGE_BOOK_MAIN):
        self.page_url = page_url
        self.main_url = main_url
        self.session = _session_with_retry()

    # ---- 解析策略 ----
    def _extract_media_links(self, html: str, base: str) -> List[str]:
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.find_all("a"):
            text = (a.get_text(strip=True) or "").lower()
            href = a.get("href") or ""
            if not href:
                continue
            # 文本优先：compressed (.ZIP) data file / Data Files
            is_text_hit = ("compressed" in text and "zip" in text) or ("data files" in text and "zip" in text)
            # 链接模式：绝对或相对 media/{id}/download
            is_media_abs = bool(MEDIA_ABS_RE.search(href))
            is_media_rel = bool(MEDIA_REL_RE.search(href))
            if is_text_hit or is_media_abs or is_media_rel:
                if is_media_rel:
                    href = requests.compat.urljoin(base, href)
                links.append(href)
        # 去重，保持顺序
        seen, out = set(), []
        for u in links:
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out

    def _resolve_zip_url_from_page(self, url: str) -> Optional[str]:
        resp = self.session.get(url, timeout=40)
        resp.raise_for_status()
        html = resp.text
        cands = self._extract_media_links(html, base=url)
        return cands[0] if cands else None

    def _resolve_zip_url(self) -> str:
        # 0) 环境变量兜底
        env_url = os.getenv("ORANGE_BOOK_ZIP_URL", "").strip()
        if env_url:
            return env_url

        # 1) 直接在数据文件页解析
        url = self._resolve_zip_url_from_page(self.page_url)
        if url:
            return url

        # 2) 在 Orange Book 主页找到 "Orange Book Data Files" 再解析
        resp = self.session.get(self.main_url, timeout=40)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        data_files_href = None
        for a in soup.find_all("a"):
            if "data files" in (a.get_text(strip=True) or "").lower():
                data_files_href = a.get("href") or ""
                if data_files_href:
                    data_files_href = requests.compat.urljoin(self.main_url, data_files_href)
                    break
        if data_files_href:
            url = self._resolve_zip_url_from_page(data_files_href)
            if url:
                return url

        # 3) 失败：抛出带诊断信息的异常
        snippet = (resp.text or "")[:1000].replace("\n", " ")
        raise RuntimeError("Failed to locate Orange Book ZIP link on FDA pages. "
                           f"Tried: {self.page_url} and {self.main_url}. "
                           "If you are behind a proxy/CDN variant, set env ORANGE_BOOK_ZIP_URL to the direct media link. "
                           f"Page snippet: {snippet[:400]}...")

    # ---- 下载与解析 ----
    def _download_zip(self, url: Optional[str] = None) -> bytes:
        """
        下载ZIP文件，支持缓存和重试机制
        
        Args:
            url: ZIP文件URL（如果为None，则自动解析）
        
        Returns:
            ZIP文件内容（bytes）
        """
        # 1. 优先尝试从缓存目录中查找ZIP文件（不需要联网，用于离线场景）
        cached_content = _load_from_cache_any()
        if cached_content is not None:
            return cached_content
        
        # 2. 如果提供了URL或环境变量，尝试使用URL查找缓存
        zip_url = url or os.getenv("ORANGE_BOOK_ZIP_URL", "").strip()
        if zip_url:
            cached_content = _load_from_cache(zip_url)
            if cached_content is not None:
                return cached_content
        
        # 3. 如果没有提供URL，尝试解析（需要联网）
        if not zip_url:
            try:
                zip_url = self._resolve_zip_url()
                # 解析成功后，再次尝试从缓存加载
                cached_content = _load_from_cache(zip_url)
                if cached_content is not None:
                    return cached_content
            except Exception as e:
                # 如果解析失败（可能是网络问题），检查是否有缓存文件
                cache_files = _find_cache_files()
                if cache_files:
                    # 有缓存文件但无法解析URL，尝试直接使用缓存
                    error_msg = f"Failed to resolve Orange Book ZIP URL (network may be unavailable).\n"
                    error_msg += f"  Error: {str(e)}\n"
                    error_msg += f"  Cache directory: {CACHE_DIR}\n"
                    error_msg += f"  Found cache files: {len(cache_files)}\n"
                    error_msg += f"  💡 Tip: Found cache files but cannot verify URL. Trying to use cache directly...\n"
                    # 尝试使用第一个缓存文件
                    try:
                        with open(cache_files[0], 'rb') as f:
                            content = f.read()
                            if zipfile.is_zipfile(io.BytesIO(content)):
                                if os.getenv("ORANGE_BOOK_DEBUG_CACHE"):
                                    print(f"[缓存] 使用缓存文件（无法验证URL）: {cache_files[0]}", file=sys.stderr)
                                return content
                    except Exception:
                        pass
                    raise RuntimeError(error_msg + f"  ❌ Cache files exist but cannot be read. Please check cache directory: {CACHE_DIR}")
                else:
                    # 没有缓存文件且无法联网，直接报错
                    error_msg = f"Failed to resolve Orange Book ZIP URL and no cache found.\n"
                    error_msg += f"  Error: {str(e)}\n"
                    error_msg += f"  Cache directory: {CACHE_DIR}\n"
                    error_msg += f"  ❌ No cache files found in cache directory!\n"
                    error_msg += f"  💡 Tip: Run prefetch_cache.py on a machine with network access to download the ZIP file.\n"
                    error_msg += f"  💡 Tip: Copy the cache directory to this machine.\n"
                    error_msg += f"  💡 Tip: Or set ORANGE_BOOK_ZIP_URL environment variable to skip URL resolution.\n"
                    raise RuntimeError(error_msg)
        
        # 4. 如果到这里，说明需要下载（有URL但缓存不存在）
        lock_path = _get_lock_path(zip_url)
        lock_file = _acquire_lock(lock_path)
        try:
            # 再次检查缓存（可能在等待锁期间被其他进程下载）
            cached_content = _load_from_cache(zip_url)
            if cached_content is not None:
                return cached_content
            
            # 5. 下载文件（带重试机制）
            content = None
            last_error = None
            cache_key = _get_cache_key(zip_url)
            cache_path = os.path.join(CACHE_DIR, cache_key)
            
            for attempt in range(MAX_RETRIES):
                try:
                    r = self.session.get(zip_url, timeout=60, allow_redirects=True)
                    r.raise_for_status()
                    content = r.content
                    
                    # 类型判断（有时返回 HTML 维护页）
                    ctype = r.headers.get("Content-Type", "").lower()
                    if "zip" not in ctype and not zipfile.is_zipfile(io.BytesIO(content)):
                        raise RuntimeError(f"Unexpected content while downloading Orange Book ZIP (url={zip_url}, content-type={ctype})")
                    
                    # 下载成功，保存到缓存
                    _save_to_cache(zip_url, content)
                    return content
                    
                except Exception as e:
                    last_error = e
                    if attempt < MAX_RETRIES - 1:
                        # 等待后重试
                        wait_time = RETRY_DELAY * (attempt + 1)  # 指数退避
                        time.sleep(wait_time)
            
            # 所有重试都失败了，提供详细的错误信息
            error_msg = f"Failed to download Orange Book ZIP after {MAX_RETRIES} attempts.\n"
            error_msg += f"  URL: {zip_url}\n"
            error_msg += f"  Last error: {str(last_error)}\n"
            error_msg += f"  Cache directory: {CACHE_DIR}\n"
            error_msg += f"  Expected cache file: {cache_path}\n"
            if not os.path.exists(CACHE_DIR):
                error_msg += f"  ❌ Cache directory does not exist!\n"
                error_msg += f"  💡 Tip: Set ORANGE_BOOK_CACHE_DIR environment variable to point to the cache directory.\n"
                error_msg += f"  💡 Tip: Copy the cache directory from the machine where prefetch_cache.py was run.\n"
            elif not os.path.exists(cache_path):
                error_msg += f"  ❌ Cache file does not exist!\n"
                error_msg += f"  💡 Tip: Run prefetch_cache.py on a machine with network access to download the ZIP file.\n"
                error_msg += f"  💡 Tip: Copy the cache directory to this machine.\n"
            raise RuntimeError(error_msg)
        
        finally:
            # 释放锁
            if lock_file:
                _release_lock(lock_file)

    @staticmethod
    def _read_tilde_file(zf: zipfile.ZipFile, member_name: str) -> List[List[str]]:
        with zf.open(member_name) as f:
            text = io.TextIOWrapper(f, encoding="utf-8", errors="ignore")
            reader = csv.reader(text, delimiter='~')
            return [row for row in reader]

    def resolve_by_ingredients(self, ingredients: List[str]) -> Dict[str, Any]:
        """根据药物成分获取专利、独占期和批准信息"""
        raw = self._download_zip()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = {n.lower(): n for n in zf.namelist()}
            prod_name = next((v for k, v in names.items() if "products.txt" in k), None)
            pat_name = next((v for k, v in names.items() if "patent.txt" in k or "patents.txt" in k), None)
            ex_name = next((v for k, v in names.items() if "exclusivity.txt" in k), None)
            if not (prod_name and pat_name and ex_name):
                raise RuntimeError("Orange Book zip missing expected files (Products/Patent/Exclusivity).")
            
            products = self._read_tilde_file(zf, prod_name)
            patents = self._read_tilde_file(zf, pat_name)
            exclusiv = self._read_tilde_file(zf, ex_name)

        # Products.txt 字段位序（见官方字段描述）
        P_ING, P_DF_ROUTE, P_TRADE, P_APPLICANT_ABBR, P_STRENGTH, P_NDA_TYPE, P_APPL_NO, \
        P_PROD_NO, P_TE, P_APPROVAL_DATE, P_RLD, P_RS, P_TYPE, P_APPLICANT_FULL = range(14)

        key_products: List[Tuple[str, str, List[str], str, Optional[str], str]] = []
        for row in products:
            if len(row) < 14:
                continue
            ingredient = row[P_ING].strip()
            if not ingredient:
                continue
            if _match_ob_ingredient(ingredients, ingredient):
                appl_no = row[P_APPL_NO].strip()
                prod_no = row[P_PROD_NO].strip()
                trade = row[P_TRADE].strip()
                appr_date = _normalize_date(row[P_APPROVAL_DATE].strip())
                applicant_full = row[P_APPLICANT_FULL].strip()
                key_products.append((appl_no, prod_no, [ingredient], trade, appr_date, applicant_full))

        approvals: List[Approval] = []
        for appl_no, prod_no, ai_list, trade, appr_date, applicant_full in key_products:
            approvals.append(Approval(
                region="US",
                product_name=trade,
                active_ingredient="; ".join(ai_list),
                approval_date=appr_date,
                status="Approved",
                marketing_authorisation_holder=applicant_full or None
            ))

        # Patent.txt：0:NDA Type, 1:Appl_No, 2:Prod_No, 3:Patent Number, 4:Patent Expire Date, ...
        pat_appl_idx, pat_prod_idx, pat_no_idx, pat_exp_idx = 1, 2, 3, 4
        pat_map: Dict[Tuple[str, str], List[Patent]] = {}
        for row in patents:
            if len(row) < 5:
                continue
            appl = row[pat_appl_idx].strip()
            prod = row[pat_prod_idx].strip()
            patno = row[pat_no_idx].strip()
            pexp = _normalize_date(row[pat_exp_idx].strip())
            if not (appl and prod and patno):
                continue
            pat_obj = Patent(jurisdiction="US", number=patno, expiry_date=pexp)
            pat_map.setdefault((appl, prod), []).append(pat_obj)

        # Exclusivity.txt：0:NDA Type,1:Appl_No,2:Prod_No,3:Exclusivity Code,4:Exclusivity Date
        ex_appl_idx, ex_prod_idx, ex_code_idx, ex_date_idx = 1, 2, 3, 4
        ex_map: Dict[Tuple[str, str], List[Exclusivity]] = {}
        for row in exclusiv:
            if len(row) < 5:
                continue
            appl = row[ex_appl_idx].strip()
            prod = row[ex_prod_idx].strip()
            code = row[ex_code_idx].strip()
            edate = _normalize_date(row[ex_date_idx].strip())
            if not (appl and prod and code):
                continue
            ex_obj = Exclusivity(region="US", type=code, end_date=edate)
            ex_map.setdefault((appl, prod), []).append(ex_obj)

        out_patents: List[Patent] = []
        out_excl: List[Exclusivity] = []
        for appl_no, prod_no, *_ in key_products:
            out_patents.extend(pat_map.get((appl_no, prod_no), []))
            out_excl.extend(ex_map.get((appl_no, prod_no), []))

        def uniq(seq):
            seen = set()
            res = []
            for x in seq:
                key = tuple(asdict(x).items())
                if key not in seen:
                    seen.add(key)
                    res.append(x)
            return res

        return {
            "patents": uniq(out_patents),
            "exclusivities": uniq(out_excl),
            "approvals": uniq(approvals),
            "sources": [ORANGE_BOOK_PAGE]
        }

# ---------- 聚合器 ----------
OPEN_STATUSES = {"Recruiting", "Enrolling by invitation", "Not yet recruiting", "Available"}

class TrialAggregator:
    """聚合临床试验和药物信息"""
    
    def __init__(self, trial_provider: Optional[TrialProvider] = None,
                 drug_provider: Optional[DrugIntelProvider] = None):
        self.trial_provider = trial_provider or CtGovProvider()
        self.drug_provider = drug_provider or OrangeBookProvider()

    def get_profile(self, nct_id: str) -> TrialCommercialProfile:
        """获取完整的临床试验商业档案"""
        if not self.trial_provider.can_handle(nct_id):
            raise ValueError("Unsupported NCT ID format")
        basic = self.trial_provider.fetch_basic_trial(nct_id)
        intel = self.drug_provider.resolve_by_ingredients(basic.get("ingredients", []))
        is_open = (basic.get("recruitment_status") or "") in OPEN_STATUSES
        return TrialCommercialProfile(
            nct_id=nct_id,
            ingredients=basic.get("ingredients") or [],
            sponsor=basic.get("sponsor") or "",
            is_open_for_enrollment=is_open,
            recruitment_status=basic.get("recruitment_status") or "",
            patents=intel.get("patents") or [],
            exclusivities=intel.get("exclusivities") or [],
            approvals=intel.get("approvals") or [],
            sources=[*basic.get("sources", []), *intel.get("sources", [])]
        )

