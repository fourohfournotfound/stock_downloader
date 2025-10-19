#!/usr/bin/env python3
"""
top50download_new_updated.py — Fast, robust NDL/ORATS pipeline with price-policy metadata & optional feature-view

Why this design (high-impact, leak-safe, shape-first):
- Bulk then append: programmatic exporter (`qopts.export=true`) for backfills; lean Tables API with paginate
  + server-side filters for daily deltas.  (See Nasdaq Data Link docs.)
- Shape invariants: strict MultiIndex (ticker, date), monotone per-ticker, no duplicate (ticker,date).
- Storage: Parquet with Hive-style partitioning: .../year=YYYY/month=MM/*.parquet (Arrow Datasets).
- Options/Fundamentals: ORATS underlying-level snapshots (tradedate) and Sharadar SF1 on filing date (`datekey`).
- NEW (this version): A formal **price policy** to separate *non-features* (adjusted prices) from *feature candidates*,
  with asset-class-aware defaults (equity vs crypto), sidecar metadata for downstream builders, and an optional
  **price feature-view** dataset that simply aliases feature-eligible price columns under a prefix (e.g., `feat_px_*`).
  This keeps adjusted columns available (no `feat_` prefix) while letting training code safely select only feature
  columns via `df.filter(like=prefix)` — avoiding look-ahead and accidental leakage.

Author: You.
import pandas as pd, subprocess, os
# Pick a matching trio; this matches Colab's preinstalled gcsfs 2025.3.0
#!pip -q install "fsspec==2025.3.0" "s3fs==2025.3.0" "gcsfs==2025.3.0" --upgrade

def save_list_as_csv(name, tickers, path="/content"):
    p = f"{path}/{name}.csv"
    pd.Series(tickers, name="ticker").dropna().drop_duplicates().to_csv(p, index=False)
    return p

def run_pipeline_from_csv(csv_path, dataset_root, out_file,
                          stages="all", extra_flags=("--shift-options","--shift-fundamentals")):
    cmd = ["python","top50download.py",
           "--tickers-file", csv_path,
           "--stages", stages, "--start","auto","--end","auto","--append",
           "--dataset-root", dataset_root, "--output-file", out_file, *extra_flags]
    subprocess.run(cmd, check=True)

!export AWS_EC2_METADATA_DISABLED=true

# Example: run your different universes
csv_top50   = save_list_as_csv("top50", top50)
csv_sp500   = save_list_as_csv("sp500", sp500)
csv_random  = save_list_as_csv("random", random)
csv_etf     = save_list_as_csv("etflist", etflist)
sectoretf   = save_list_as_csv("sectoretf", sectoretf)
top100etf   = save_list_as_csv("top100etf", top100etf)

# Run your pipelines
!python top50download.py \
  --tickers-list "MSFT,AAPL,NVDA,AMZN,GOOGL" \
  --stages all \
  --dataset-root "/content/data2" \
  --s3-endpoint-url "$AWS_S3_ENDPOINT_URL" \
  --parquet-compression snappy \
  --start 2010-01-01 --end auto


#--tickers-list "MSFT,AAPL,NVDA,AMZN,GOOGL"

"""

from __future__ import annotations

import os, io, re, sys, json, time, zipfile, argparse, warnings, posixpath
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlencode

# --- light bootstrap (optional in fresh envs) ---
def _ensure(pkg: str, import_name: Optional[str]=None):
    try: __import__(import_name or pkg)
    except Exception:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
        __import__(import_name or pkg)

_ensure("nasdaq-data-link", "nasdaqdatalink")
_ensure("pandas", "pandas")
_ensure("pyarrow", "pyarrow")
_ensure("requests", "requests")
_ensure("tqdm", "tqdm")
_ensure("fsspec", "fsspec")

import nasdaqdatalink
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.dataset as ds
from pyarrow.fs import LocalFileSystem, S3FileSystem, FileType, FileSelector, PyFileSystem, FSSpecHandler

warnings.filterwarnings("ignore")
pd.options.display.width = 160
pd.options.display.max_columns = 250

# --------------------
# Defaults / constants
# --------------------
PRICES_TABLE_DEFAULT  = "SHARADAR/SEP"   # equities OHLCV (filterable: ticker, date, lastupdated)
SF1_TABLE_DEFAULT     = "SHARADAR/SF1"   # fundamentals (use filing date: datekey)
OPTIONS_TABLE_DEFAULT = "ORATS/VOL"      # ORATS underlying-level metrics (premium db)

CANON_PRICE_COLS = ["open","high","low","close","closeadj","volume"]
TRACE_COLS       = ["lastupdated"]       # keep when present (revision audits)
BULK_PRICE_COLS  = ["ticker","date"] + CANON_PRICE_COLS + TRACE_COLS

# Prefer filing date (datekey) over calendardate to reduce look-ahead for fundamentals
DATE_CANDIDATES   = ("date","tradedate","tradeDate","quoteDate","datekey","calendardate")
TICKER_CANDIDATES = ("ticker","underlying","symbol","root")

# --------------------
# Price policy
# --------------------
@dataclass
class PricePolicy:
    asset_class: str                      # 'equity' or 'crypto'
    feature_allowed: List[str]            # columns eligible to be used as features by default
    feature_blocked: List[str]            # columns that must NOT be used as features (e.g., adjusted)
    target_anchors: Dict[str, str]        # which field to anchor for forward returns, e.g., {'open_to_open': 'openadj'}
    backtest_pricing_preference: List[str]# ordered list to prefer when pricing a trade/backtest

def _default_price_policy(asset_class: str) -> PricePolicy:
    ac = (asset_class or "equity").lower()
    if ac == "crypto":
        # No adjusted fields; all native OHLCV can be features and also drive backtests/targets
        allowed = ["open","high","low","close","volume"]
        blocked = []  # no adjusted fields to block
        anchors = {"open_to_open": "open", "close_to_close": "close"}
        prefer  = ["open","close","high","low"]
    else:
        # Equity/ETF default: adjusted prices available but **not features**
        allowed = ["open","high","low","close","volume"]  # non-adjusted are OK as feature bases
        blocked = ["openadj","closeadj"]                  # adjusted are *not* features
        anchors = {"open_to_open": "openadj", "close_to_close": "closeadj"}  # safe for target gen
        prefer  = ["openadj","closeadj","open","close"]
    return PricePolicy(ac, allowed, blocked, anchors, prefer)

# --------------------
# FS helpers (local, S3/Wasabi, GCS via fsspec)
# --------------------
def _collapse(p: str) -> str:
    p = p.replace("\\", "/")
    return re.sub(r"/+", "/", p)

def _fs_from_uri(uri: str, s3_endpoint_url: Optional[str] = None):
    from urllib.parse import urlparse
    u = urlparse(uri)
    scheme = (u.scheme or "").lower()

    default_endpoint = "https://s3.wasabisys.com"
    endpoint = (
        s3_endpoint_url or os.getenv("AWS_ENDPOINT_URL_S3")
        or os.getenv("AWS_S3_ENDPOINT_URL") or os.getenv("AWS_ENDPOINT_URL")
        or default_endpoint
    )
    if scheme == "s3":
        fs = S3FileSystem(
            access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            session_token=os.getenv("AWS_SESSION_TOKEN"),
            region=os.getenv("AWS_DEFAULT_REGION","us-east-1"),
            endpoint_override=endpoint
        )
        base = _collapse(f"{u.netloc}{u.path}").lstrip("/")
        return fs, base

    if scheme in ("gs","gcs","abfs","az","adl","azfs"):
        import fsspec
        fs_fsspec, _, paths = fsspec.get_fs_token_paths(uri)
        return PyFileSystem(FSSpecHandler(fs_fsspec)), _collapse(paths[0]).lstrip("/")

    # local
    fs = LocalFileSystem()
    raw = u.path if u.path else uri
    return fs, _collapse(raw)

# --------------------
# Manifest (per stage)
# --------------------
@dataclass
class StageState:   last_date: Optional[str] = None
@dataclass
class Manifest:
    prices: StageState
    options: StageState
    fundamentals: StageState
    etf_options: StageState
    def as_dict(self): return {
        "prices": self.prices.__dict__,
        "options": self.options.__dict__,
        "fundamentals": self.fundamentals.__dict__,
        "etf_options": self.etf_options.__dict__,
    }

def _manifest_path(root: str) -> str:
    return _collapse(posixpath.join(root.rstrip("/"), "_manifest.json"))

def _load_manifest(dataset_root: str, s3_endpoint: Optional[str]) -> Manifest:
    fs, base = _fs_from_uri(dataset_root, s3_endpoint)
    mpath = _manifest_path(base)
    info = fs.get_file_info(mpath)
    if info.type == FileType.NotFound:
        return Manifest(StageState(), StageState(), StageState(), StageState())
    with fs.open_input_stream(mpath) as f:
        data = json.load(io.TextIOWrapper(f, "utf-8"))
    def _get(x, k):
        d = x.get(k, {}) if isinstance(x, dict) else {}
        return StageState(d.get("last_date"))
    return Manifest(_get(data,"prices"), _get(data,"options"), _get(data,"fundamentals"), _get(data,"etf_options"))

def _save_manifest(dataset_root: str, m: Manifest, s3_endpoint: Optional[str]) -> None:
    fs, base = _fs_from_uri(dataset_root, s3_endpoint)
    mpath = _manifest_path(base)
    try: fs.create_dir(posixpath.dirname(mpath), recursive=True)
    except Exception: pass
    with fs.open_output_stream(mpath) as f:
        f.write(json.dumps(m.as_dict(), indent=2, sort_keys=True).encode("utf-8"))

# --------------------
# Small utils
# --------------------
def _ts(): return time.strftime("%H:%M:%S")
def _log(msg: str): print(f"[{_ts()}] {msg}", flush=True)

def _find_col(cols: List[str], candidates: Tuple[str, ...]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for k in candidates:
        if k.lower() in low: return low[k.lower()]
    return None

def _normalize_date_ticker(df: pd.DataFrame, prefer_date: Optional[str]=None) -> pd.DataFrame:
    """
    Normalize columns to strict ('ticker','date'), with optional preference for a specific date column
    (e.g., prefer_date='datekey' for fundamentals to reduce look-ahead).
    """
    if df is None or df.empty: return pd.DataFrame()
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()

    # prefer specific date col if present
    dcol = None
    if prefer_date and any(c.lower() == prefer_date.lower() for c in df.columns):
        dcol = [c for c in df.columns if c.lower() == prefer_date.lower()][0]

    tcol = _find_col(list(df.columns), TICKER_CANDIDATES)
    if not dcol:
        dcol = _find_col(list(df.columns), DATE_CANDIDATES)

    if not tcol or not dcol: return pd.DataFrame()
    out = df.rename(columns={tcol:"ticker", dcol:"date"}).copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.replace("-", ".", regex=False)
    out["date"]   = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_localize(None)
    return out.dropna(subset=["ticker","date"])

def _ensure_panel(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    df = df.loc[:, ~df.columns.duplicated()].copy()
    missing = {"ticker","date"} - set(df.columns)
    if missing: raise KeyError(f"Missing required columns: {sorted(missing)}")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["date"]   = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["ticker","date"]).drop_duplicates(["ticker","date"], keep="last")
    values = [c for c in df.columns if c not in ("ticker","date")]
    out = df[["ticker","date"] + values].set_index(["ticker","date"]).sort_index()
    # shape invariants
    assert not out.index.duplicated().any(), "Duplicate (ticker,date)"
    ok = out.reset_index().groupby("ticker", sort=False)["date"].apply(lambda s: s.is_monotonic_increasing).all()
    assert ok, "Per-ticker dates must be monotone increasing"
    return out

def _synthesize_openadj(df: pd.DataFrame) -> pd.DataFrame:
    """
    Provide openadj when possible so open-to-open targets are convenient & leak-safe for equities.
    For crypto (no closeadj), this will fall back to 'open' (and the policy anchors will choose 'open').
    """
    if "openadj" in df.columns: return df
    if {"open","close","closeadj"}.issubset(df.columns):
        close_safe = pd.to_numeric(df["close"], errors="coerce").replace(0, np.nan)
        df["openadj"] = pd.to_numeric(df["open"], errors="coerce") * (pd.to_numeric(df["closeadj"], errors="coerce") / close_safe)
        df["openadj"] = df["openadj"].fillna(pd.to_numeric(df["open"], errors="coerce"))
    elif "open" in df.columns:
        df["openadj"] = pd.to_numeric(df["open"], errors="coerce")
    else:
        df["openadj"] = np.nan
    return df

# --------------------
# Parquet writer (partitioned)
# --------------------
def _write_partitioned(dataset_root: str,
                       df: pd.DataFrame,
                       mode: str,
                       s3_endpoint: Optional[str],
                       verify_listing: bool,
                       compression: str = "snappy"):
    """
    Write a MultiIndex (ticker,date) DataFrame to a Parquet dataset partitioned by year/month
    using **Hive-style partitioning**: e.g., .../year=2025/month=10/part-*.parquet
    """
    if df is None or df.empty: return
    assert isinstance(df.index, pd.MultiIndex) and list(df.index.names) == ["ticker","date"], \
        "Expected MultiIndex ['ticker','date']"

    out = df.reset_index().copy()
    out["date"]  = pd.to_datetime(out["date"], utc=True).dt.tz_localize(None)
    out["year"]  = out["date"].dt.year.astype("int16")
    out["month"] = out["date"].dt.month.astype("int8")
    table = pa.Table.from_pandas(out, preserve_index=False)

    dst_fs, dst_base = _fs_from_uri(dataset_root, s3_endpoint)
    dst_base = _collapse(dst_base)
    # for local FS ensure absolute path for base
    if isinstance(dst_fs, LocalFileSystem) and not dst_base.startswith("/"):
        dst_base = "/" + dst_base

    try: dst_fs.create_dir(dst_base, recursive=True)
    except Exception: pass

    fmt = ds.ParquetFileFormat()
    try:
        file_opts = fmt.make_write_options(compression=compression)
    except Exception:
        file_opts = None

    write_kwargs = dict(
        base_dir=dst_base,
        format=fmt,
        partitioning=["year","month"],
        partitioning_flavor="hive",   # ensure year=YYYY/month=MM
        existing_data_behavior=("overwrite_or_ignore" if mode=="append" else "delete_matching"),
        filesystem=dst_fs,
        use_threads=True,
        max_open_files=64,
        max_rows_per_file=1_000_000,
        min_rows_per_group=50_000,
        max_rows_per_group=250_000,
    )
    if file_opts is not None:
        write_kwargs["file_options"] = file_opts

    _log(f"[write] → {type(dst_fs).__name__}:{dst_base} rows={len(out):,} cols={out.shape[1]}")
    t0 = time.time()
    try:
        ds.write_dataset(table, **write_kwargs)
    except TypeError:
        # older pyarrow that doesn't accept all kwargs
        for k in ("partitioning_flavor","min_rows_per_group","max_rows_per_group","max_rows_per_file","max_open_files","use_threads","file_options"):
            write_kwargs.pop(k, None)
        ds.write_dataset(table, **write_kwargs)
    _log(f"[write] done in {time.time()-t0:.1f}s")

    if verify_listing:
        try:
            infos = dst_fs.get_file_info(FileSelector(dst_base, recursive=True))
            wrote = [i.path for i in infos if i.type == FileType.File and i.path.endswith(".parquet")]
            _log(f"[verify] parquet files under '{dst_base}': {len(wrote)}")
        except Exception as e:
            _log(f"[verify] listing failed: {e}")

# --------------------
# Sidecar metadata & feature-view helpers
# --------------------
def _write_json_sidecar(dir_path: str, filename: str, obj: dict, s3_endpoint: Optional[str]):
    fs, base = _fs_from_uri(dir_path, s3_endpoint)
    try: fs.create_dir(base, recursive=True)
    except Exception: pass
    path = _collapse(posixpath.join(base, filename))
    with fs.open_output_stream(path) as f:
        f.write(json.dumps(obj, indent=2, sort_keys=True).encode("utf-8"))
    _log(f"[sidecar] wrote {type(fs).__name__}:{path}")

def _build_price_feature_view(prices_panel: pd.DataFrame, policy: PricePolicy, prefix: str) -> pd.DataFrame:
    """
    Create a "feature view" from price data by aliasing **feature-eligible** columns under `prefix`.
    - Adjusted columns (e.g., openadj/closeadj) are not included for equities.
    - For crypto, all native OHLCV are allowed.
    The output keeps the *same* (ticker,date) MultiIndex and contains only prefixed columns.
    """
    if prices_panel is None or prices_panel.empty: return pd.DataFrame(index=prices_panel.index)
    allowed = [c for c in policy.feature_allowed if c in prices_panel.columns]
    if not allowed: return pd.DataFrame(index=prices_panel.index)
    view = prices_panel[allowed].copy()
    rename_map = {c: f"{prefix}{c}" for c in allowed}
    view = view.rename(columns=rename_map)
    # Retain shape invariants on index
    view = view.sort_index()
    assert isinstance(view.index, pd.MultiIndex) and list(view.index.names) == ["ticker","date"]
    return view

# --------------------
# Exporter (bulk) — programmatic polling + download
# --------------------
API_BASE = "https://data.nasdaq.com/api/v3/datatables"

def _datatable_json_url(table_code: str, params: Dict[str, object]) -> str:
    qs = urlencode(params, doseq=True)
    return f"{API_BASE}/{table_code}.json?{qs}"

def export_poll_and_download(table_code: str,
                             filters: Dict[str, object],
                             columns: Optional[List[str]],
                             api_key: str,
                             poll_every_s: float = 2.0,
                             max_wait_s: float = 600.0,
                             out_zip_path: str = "export_tmp.zip") -> Optional[str]:
    """
    Ask the exporter for a packaged file (qopts.export=true) and poll status via .json
    until 'fresh' & link is available; then download zip with progress. (NDL doc-backed)
    """
    params = dict(filters or {})
    params["qopts.export"] = "true"
    if columns:
        cols = list(dict.fromkeys(columns + ["ticker","date"]))  # ensure both present
        params["qopts.columns"] = ",".join([c for c in cols if c])
    params["api_key"] = api_key

    t0, link, status = time.time(), None, None
    while True:
        url = _datatable_json_url(table_code, params)
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            info = r.json().get("datatable_bulk_download", {})
            file_info = info.get("file", {}) or {}
            status, link = file_info.get("status"), file_info.get("link")
            _log(f"[export] {table_code} status={status!s}")
            if status == "fresh" and link: break
        except Exception as e:
            _log(f"[export] polling error: {e}")
        if time.time() - t0 > max_wait_s:
            _log(f"[export] timeout after {int(max_wait_s)}s")
            return None
        time.sleep(poll_every_s)

    try:
        with requests.get(link, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0))
            with open(out_zip_path, "wb") as f, tqdm(
                total=total if total > 0 else None,
                unit="B", unit_scale=True, unit_divisor=1024,
                desc=f"Downloading {table_code} zip"
            ) as bar:
                for chunk in resp.iter_content(chunk_size=1<<20):
                    if chunk:
                        f.write(chunk)
                        if total > 0: bar.update(len(chunk))
        return out_zip_path
    except Exception as e:
        _log(f"[export] download error: {e}")
        return None

def _read_zip_csvs_fast(zip_path: str, include_cols: Optional[List[str]]) -> pd.DataFrame:
    if not zip_path or (not os.path.exists(zip_path)): return pd.DataFrame()
    inc = list(dict.fromkeys(include_cols or [])) if include_cols else None
    dfs = []
    with zipfile.ZipFile(zip_path, "r") as z:
        csvs = [nm for nm in z.namelist() if nm.lower().endswith(".csv")]
        _log(f"[zip] reading {len(csvs)} CSV file(s)")
        for nm in csvs:
            b = z.read(nm)
            try:
                tbl = pa_csv.read_csv(
                    pa.BufferReader(b),
                    read_options=pa_csv.ReadOptions(autogenerate_column_names=False),
                    convert_options=(pa_csv.ConvertOptions(include_columns=inc) if inc else pa_csv.ConvertOptions())
                )
                dfs.append(tbl.to_pandas())
            except Exception:
                df = pd.read_csv(io.BytesIO(b), low_memory=False)
                if inc:
                    keep = [c for c in df.columns if c in inc]
                    df = df[keep] if keep else df
                dfs.append(df)
    return pd.concat(dfs, axis=0, ignore_index=True) if dfs else pd.DataFrame()

# --------------------
# Tables (append path) — robust paginate + chunked tickers
# --------------------
def get_table_chunked(table_code: str,
                      row_filters: Dict,
                      date_bounds: Optional[Dict] = None,
                      select_cols: Optional[List[str]] = None,
                      per_page: int = 10000,
                      chunk_size: int = 80) -> pd.DataFrame:
    """
    Chunk the 'ticker' filter → smaller requests; use paginate=True; push qopts.per_page/columns.
    """
    rf = dict(row_filters or {})
    tickers = rf.get("ticker", None)
    chunks: List[Optional[List[str]]] = [None]
    if isinstance(tickers, (list, tuple, set)):
        norm = [str(t).upper().replace("-", ".") for t in tickers]
        uniq = list(dict.fromkeys(norm))
        chunks = [uniq[i:i+chunk_size] for i in range(0, len(uniq), chunk_size)]
    elif tickers is not None:
        chunks = [[str(tickers).upper().replace("-", ".")]]

    def _call(ch):
        kwargs = {}
        if ch is not None: kwargs["ticker"] = ch
        for k, v in rf.items():
            if k == "ticker": continue
            kwargs[k] = v
        if date_bounds:
            kwargs.setdefault("date", {"gte": date_bounds["gte"], "lte": date_bounds["lte"]})
        qopts = {}
        if select_cols: qopts["columns"] = list(dict.fromkeys(select_cols))
        if per_page:    qopts["per_page"] = int(per_page)
        if qopts:       kwargs["qopts"] = qopts

        part = nasdaqdatalink.get_table(table_code, paginate=True, **kwargs)  # ~1M rows with paginate
        return part if isinstance(part, pd.DataFrame) else pd.DataFrame()

    outs = []
    pbar = tqdm(total=len(chunks), desc=f"Fetching {table_code} chunks", unit="chunk")
    try:
        for ch in chunks:
            df = _call(ch); pbar.update(1); pbar.set_postfix({"rows": 0 if df is None else len(df)})
            if df is not None and not df.empty: outs.append(df)
    finally:
        pbar.close()
    return pd.concat(outs, axis=0, ignore_index=True) if outs else pd.DataFrame()

# --------------------
# Fetchers (stages)
# --------------------
def fetch_prices(tickers: List[str], start: str, end: str,
                 table: str, use_export: bool, api_key: str) -> pd.DataFrame:
    tickers = [str(t).upper().replace("-", ".") for t in (tickers or [])]
    if not tickers: return pd.DataFrame()

    df = pd.DataFrame()
    if use_export:
        zip_path = export_poll_and_download(
            table_code=table,
            filters={"ticker": ",".join(tickers), "date.gte": start, "date.lte": end},
            columns=BULK_PRICE_COLS,
            api_key=api_key,
            out_zip_path="prices_export.zip"
        )
        df = _read_zip_csvs_fast(zip_path, include_cols=BULK_PRICE_COLS) if zip_path else pd.DataFrame()
        if df.empty: _log("[prices] exporter empty/timeout; falling back to API.")
    if (not use_export) or df.empty:
        df = get_table_chunked(
            table_code=table,
            row_filters={"ticker": tickers},
            date_bounds={"gte": start, "lte": end},
            select_cols=BULK_PRICE_COLS,  # include lastupdated if present
        )
    df = _normalize_date_ticker(df)
    if df.empty: return df
    keep_vals = [c for c in CANON_PRICE_COLS + TRACE_COLS if c in df.columns]
    out = df[["ticker","date"] + keep_vals].drop_duplicates(["ticker","date"])
    out = _ensure_panel(out)
    out = _synthesize_openadj(out)  # ensures openadj exists for equities; crypto falls back to 'open'
    return out

def fetch_options(tickers: List[str], start: str, end: str,
                  table: str, use_export: bool, api_key: str) -> pd.DataFrame:
    tickers = [str(t).upper().replace("-", ".") for t in (tickers or [])]
    if not tickers: return pd.DataFrame()

    df = pd.DataFrame()
    if use_export:
        zip_path = export_poll_and_download(
            table_code=table,
            filters={"ticker": ",".join(tickers), "tradedate.gte": start, "tradedate.lte": end},
            columns=None, api_key=api_key, out_zip_path="options_export.zip"
        )
        df = _read_zip_csvs_fast(zip_path, include_cols=None) if zip_path else pd.DataFrame()
        if df.empty: _log("[options] exporter empty/timeout; falling back to API.")
    if (not use_export) or df.empty:
        df = get_table_chunked(
            table_code=table,
            row_filters={"ticker": tickers, "tradedate": {"gte": start, "lte": end}},
            date_bounds=None, select_cols=None, per_page=10000, chunk_size=60
        )
    df = _normalize_date_ticker(df)
    if df.empty: return df
    # Keep underlying-level summaries; drop ultra-wide per-contract shapes
    drop_like = ("strike","expiry","expiration","days","moneyness")
    drops = [c for c in df.columns if any(x in c.lower() for x in drop_like)]
    df = df.drop(columns=drops, errors="ignore")
    return _ensure_panel(df).add_prefix("OPT_")

def fetch_sf1(tickers: List[str], start: str, end: str,
              table: str, dimensions: List[str], use_export: bool, api_key: str) -> pd.DataFrame:
    """
    Fundamentals: Prefer filing date ('datekey') to reduce look-ahead. Keep lastupdated for audits.
    """
    tickers = [str(t).upper().replace("-", ".") for t in (tickers or [])]
    if not tickers: return pd.DataFrame()

    df = pd.DataFrame()
    if use_export:
        zip_path = export_poll_and_download(
            table_code=table,
            filters={
                "ticker": ",".join(tickers),
                "dimension": ",".join(dimensions),
                "datekey.gte": start, "datekey.lte": end
            },
            columns=None, api_key=api_key, out_zip_path="sf1_export.zip"
        )
        df = _read_zip_csvs_fast(zip_path, include_cols=None) if zip_path else pd.DataFrame()
        if df.empty: _log("[fundamentals] exporter empty/timeout; falling back to API.")
    if (not use_export) or df.empty:
        df = get_table_chunked(
            table_code=table,
            row_filters={"ticker": tickers, "dimension": dimensions, "datekey": {"gte": start, "lte": end}},
            date_bounds=None, select_cols=None, per_page=10000, chunk_size=60
        )
    # Prefer filing date if present
    df = _normalize_date_ticker(df, prefer_date="datekey")
    if df.empty: return df

    if "dimension" not in df.columns:
        value_cols = [c for c in df.columns if c not in ("ticker","date","lastupdated","reportperiod","calendardate","datekey")]
        out = df[{"ticker","date", *value_cols}].copy() if value_cols else df[["ticker","date"]].copy()
        out = out.rename(columns={c: f"FUND_{c}" for c in value_cols})
        return _ensure_panel(out)

    value_cols = [c for c in df.columns if c not in ("ticker","date","dimension","lastupdated","reportperiod","calendardate","datekey")]
    if not value_cols:
        return _ensure_panel(df[["ticker","date"]].drop_duplicates())

    piv = (df.pivot_table(index=["ticker","date"], columns="dimension", values=value_cols, aggfunc="last")
             .sort_index())
    piv.columns = [f"FUND_{val}__{dim}" for (val, dim) in piv.columns.to_flat_index()]
    return _ensure_panel(piv.reset_index())

def fetch_etf_options(etf_tickers: List[str], start: str, end: str,
                      table: str, use_export: bool, api_key: str) -> pd.DataFrame:
    etf_tickers = [str(x).upper() for x in (etf_tickers or [])]
    if not etf_tickers: return pd.DataFrame()
    df = pd.DataFrame()
    if use_export:
        zip_path = export_poll_and_download(
            table_code=table,
            filters={"ticker": ",".join(etf_tickers), "tradedate.gte": start, "tradedate.lte": end},
            columns=None, api_key=api_key, out_zip_path="etf_options_export.zip"
        )
        df = _read_zip_csvs_fast(zip_path, include_cols=None) if zip_path else pd.DataFrame()
        if df.empty: _log("[etf_options] exporter empty/timeout; falling back to API.")
    if (not use_export) or df.empty:
        df = get_table_chunked(
            table_code=table,
            row_filters={"ticker": etf_tickers, "tradedate": {"gte": start, "lte": end}},
            date_bounds=None, select_cols=None, per_page=10000, chunk_size=60
        )
    df = _normalize_date_ticker(df)
    if df.empty: return pd.DataFrame()
    # Date‑level context (EW across ETFs), broadcast later as needed
    num_cols = df.select_dtypes(include=[np.number]).columns.difference(["ticker"]).tolist()
    if not num_cols: return pd.DataFrame()
    ctx = df.groupby("date", as_index=False)[num_cols].mean(numeric_only=True)
    ctx = ctx.add_prefix("CTX_").rename(columns={"CTX_date":"date"})
    return ctx

# --------------------
# Date helpers
# --------------------
def _next_start_date(manifest_date: Optional[str], user_start: Optional[str]) -> str:
    if user_start and str(user_start).lower() != "auto": return user_start
    if manifest_date:
        dt = pd.to_datetime(manifest_date) + pd.tseries.offsets.BDay(1)
        return dt.strftime("%Y-%m-%d")
    return "2010-01-01"

def _end_date(user_end: Optional[str], prev_bday: bool) -> str:
    if user_end and user_end != "auto": return str(user_end)
    today = pd.Timestamp.today(tz="UTC").normalize()
    return (today - pd.tseries.offsets.BDay(1) if prev_bday else today).strftime("%Y-%m-%d")

# --------------------
# Runner
# --------------------
def run_pipeline(args):
    t_start = time.time()
    api_key = os.getenv("NASDAQ_DATA_LINK_API_KEY", "")
    if not api_key:
        raise RuntimeError("Set NASDAQ_DATA_LINK_API_KEY in environment.")
    nasdaqdatalink.ApiConfig.api_key = api_key

    # Tickers
    tickers: List[str] = []
    if args.tickers_file:
        s = pd.read_csv(args.tickers_file)
        tickers = (s["ticker"] if "ticker" in s.columns else s.iloc[:,0]).astype(str).tolist()
    if args.tickers_list:
        tickers += [t.strip() for t in args.tickers_list.split(",") if t.strip()]
    tickers = [t.upper().replace("-", ".") for t in tickers]
    if not tickers:
        raise ValueError("No tickers provided.")
    _log(f"[plan] tickers={len(tickers)} sample={tickers[:10]}")

    # Stages
    stages = [s.strip().lower() for s in (args.stages or "prices,options,fundamentals").split(",")]
    use_bulk = not args.append  # bulk for backfill; API for append
    if args.end == "auto" and not args.prev_bday:
        args.prev_bday = True

    # Manifest & windows
    manifest = _load_manifest(args.dataset_root, args.s3_endpoint_url) if args.append else Manifest(StageState(),StageState(),StageState(),StageState())
    end = _end_date(args.end, prev_bday=args.prev_bday)
    start = args.start if args.start and args.start != "auto" else "2010-01-01"

    # Optional repair window (re-fetch last N business days)
    if args.repair_days and args.repair_days > 0:
        end_dt = pd.to_datetime(end)
        start_dt = end_dt - pd.tseries.offsets.BDay(int(args.repair_days) - 1)
        start = start_dt.strftime("%Y-%m-%d")

    merged = pd.DataFrame()

    # Price policy (asset-class aware)
    price_policy = _default_price_policy(args.asset_class)

    # === PRICES ===
    if "prices" in stages or "all" in stages:
        start_p = _next_start_date(manifest.prices.last_date, start)
        _log(f"[prices] table={args.prices_table} method={'BULK' if use_bulk else 'API'} window={start_p}→{end} n_tickers={len(tickers)}")
        t0 = time.time()
        prices = fetch_prices(tickers, start_p, end, args.prices_table, use_bulk, api_key)
        _log(f"[prices] rows={0 if prices is None else prices.shape[0]} cols={0 if prices is None else prices.shape[1]} elapsed={time.time()-t0:.1f}s")
        if not prices.empty:
            merged = prices
            _write_partitioned(args.dataset_root + "/prices", prices, mode=("append" if args.append else "overwrite"),
                               s3_endpoint=args.s3_endpoint_url, compression=args.parquet_compression,
                               verify_listing=args.verify_write)
            manifest.prices.last_date = str(prices.index.get_level_values("date").max().date())

            # Price-policy sidecar metadata
            if args.write_price_policy:
                sidecar = {
                    "version": 1,
                    "asset_class": price_policy.asset_class,
                    "feature_allowed": price_policy.feature_allowed,
                    "feature_blocked": price_policy.feature_blocked,
                    "target_anchors": price_policy.target_anchors,
                    "backtest_pricing_preference": price_policy.backtest_pricing_preference,
                    "notes": "Adjusted price columns are present for convenience but are not feature-eligible by default (equities). "
                             "Crypto has no adjusted fields; all OHLCV are feature-eligible and drive targets/backtests."
                }
                _write_json_sidecar(args.dataset_root + "/prices", "_price_policy.json", sidecar, args.s3_endpoint_url)

            # Optional feature-view with prefixed price columns (no adjusted fields for equities)
            if args.emit_price_feature_view:
                feat_view = _build_price_feature_view(prices, price_policy, args.price_feature_prefix)
                if not feat_view.empty:
                    _write_partitioned(args.dataset_root + "/prices_feature_view", feat_view,
                                       mode=("append" if args.append else "overwrite"),
                                       s3_endpoint=args.s3_endpoint_url, compression=args.parquet_compression,
                                       verify_listing=args.verify_write)

        else:
            _log("[prices] nothing fetched.")

    # === OPTIONS ===
    if "options" in stages or "all" in stages:
        start_o = _next_start_date(manifest.options.last_date, start)
        _log(f"[options] table={args.options_table} method={'BULK' if use_bulk else 'API'} window={start_o}→{end} n_tickers={len(tickers)}")
        t0 = time.time()
        opt = fetch_options(tickers, start_o, end, args.options_table, use_bulk, api_key)
        _log(f"[options] rows={0 if opt is None else opt.shape[0]} cols={0 if opt is None else opt.shape[1]} elapsed={time.time()-t0:.1f}s")
        if not opt.empty:
            merged = opt if merged.empty else merged.join(opt, how="left")
            _write_partitioned(args.dataset_root + "/options", opt, mode=("append" if args.append else "overwrite"),
                               s3_endpoint=args.s3_endpoint_url, compression=args.parquet_compression,
                               verify_listing=args.verify_write)
            manifest.options.last_date = str(opt.index.get_level_values("date").max().date())

    # === FUNDAMENTALS (SF1) ===
    if "fundamentals" in stages or "all" in stages:
        dims = [d.strip() for d in (args.sf1_dimensions or "ARQ,ARY,ART").split(",") if d.strip()]
        start_f = _next_start_date(manifest.fundamentals.last_date, start)
        _log(f"[fundamentals] table={args.sf1_table} dims={','.join(dims)} method={'BULK' if use_bulk else 'API'} window={start_f}→{end} n_tickers={len(tickers)}")
        t0 = time.time()
        sf1 = fetch_sf1(tickers, start_f, end, args.sf1_table, dims, use_bulk, api_key)
        _log(f"[fundamentals] rows={0 if sf1 is None else sf1.shape[0]} cols={0 if sf1 is None else sf1.shape[1]} elapsed={time.time()-t0:.1f}s")
        if not sf1.empty:
            merged = sf1 if merged.empty else merged.join(sf1, how="left")
            _write_partitioned(args.dataset_root + "/fundamentals", sf1, mode=("append" if args.append else "overwrite"),
                               s3_endpoint=args.s3_endpoint_url, compression=args.parquet_compression,
                               verify_listing=args.verify_write)
            manifest.fundamentals.last_date = str(sf1.index.get_level_values("date").max().date())

    # === ETF OPTIONS (date-only context) ===
    if "etf_options" in stages or "all" in stages:
        etfs = [t.strip() for t in (args.etf_option_tickers or "").split(",") if t.strip()]
        if etfs:
            start_e = _next_start_date(manifest.etf_options.last_date, start)
            _log(f"[etf_options] table={args.options_table} method={'BULK' if use_bulk else 'API'} window={start_e}→{end} n_tickers={len(etfs)}")
            t0 = time.time()
            ctx = fetch_etf_options(etfs, start_e, end, args.options_table, use_bulk, api_key)
            _log(f"[etf_options] ctx_rows={0 if ctx is None else ctx.shape[0]} elapsed={time.time()-t0:.1f}s")
            if not ctx.empty:
                ctx = ctx.sort_values("date").drop_duplicates("date", keep="last")
                ctx["ticker"] = "__CTX__"   # make it index-compatible
                ctx_panel = _ensure_panel(ctx)
                _write_partitioned(args.dataset_root + "/etf_options", ctx_panel, mode=("append" if args.append else "overwrite"),
                                   s3_endpoint=args.s3_endpoint_url, compression=args.parquet_compression,
                                   verify_listing=args.verify_write)
                manifest.etf_options.last_date = str(ctx_panel.index.get_level_values("date").max().date())

    # Save manifest (always save)
    _save_manifest(args.dataset_root, manifest, args.s3_endpoint_url)

    _log(f"[done] total elapsed {time.time()-t_start:.1f}s")
    if not merged.empty:
        _log(f"[shape] merged example rows={merged.shape[0]:,} cols={merged.shape[1]}  "
             f"n_dates={merged.index.get_level_values('date').nunique()} n_tickers={merged.index.get_level_values('ticker').nunique()}")

def build_argparser():
    p = argparse.ArgumentParser(description="Bulk + Incremental NDL downloader (Sharadar + ORATS) with price-policy sidecars")
    p.add_argument("--dataset-root", type=str, required=True,
                   help="Destination root (e.g., '/data/ndl' or 's3://bucket/ndl')")
    p.add_argument("--tickers-file", type=str, default="",
                   help="CSV with a 'ticker' column (or first column)")
    p.add_argument("--tickers-list", type=str, default="",
                   help="Comma-separated list of tickers, e.g., 'AAPL,MSFT,GOOGL'")
    p.add_argument("--stages", type=str, default="prices,options,fundamentals",
                   help="Subset of stages: prices,options,fundamentals,etf_options,all")
    p.add_argument("--prices-table", type=str, default=PRICES_TABLE_DEFAULT)
    p.add_argument("--sf1-table", type=str, default=SF1_TABLE_DEFAULT)
    p.add_argument("--options-table", type=str, default=OPTIONS_TABLE_DEFAULT)
    p.add_argument("--sf1-dimensions", type=str, default="ARQ,ARY,ART",
                   help="Comma-separated SF1 dimensions")

    # Backfill/append controls
    p.add_argument("--append", action="store_true",
                   help="Append mode: use Tables API for daily deltas; otherwise bulk backfill")
    p.add_argument("--start", type=str, default="auto",
                   help="Start date 'YYYY-MM-DD' or 'auto' (manifest or 2010-01-01)")
    p.add_argument("--end", type=str, default="auto",
                   help="End date 'YYYY-MM-DD' or 'auto' (today or prev_bday)")
    p.add_argument("--prev-bday", action="store_true", help="When end='auto', use previous business day")
    p.add_argument("--repair-days", type=int, default=0,
                   help="Re-fetch last N business days (overrides start for quick fixes)")

    # Storage
    p.add_argument("--parquet-compression", type=str, default="snappy")
    p.add_argument("--verify-write", action="store_true",
                   help="List written parquet files for sanity")
    p.add_argument("--s3-endpoint-url", type=str, default="",
                   help="S3-compatible endpoint (e.g., Wasabi). Can also be set via AWS env vars.")
    p.add_argument("--etf-option-tickers", type=str, default="",
                   help="Comma-separated ETF list for context (e.g., 'SPY,QQQ,IWM')")

    # NEW: price policy / feature view controls
    p.add_argument("--asset-class", type=str, default="equity", choices=["equity","crypto"],
                   help="Controls default price policy: equities block adjusted fields as features; crypto allows all OHLCV")
    p.add_argument("--write-price-policy", action="store_true",
                   help="Write a _price_policy.json sidecar into dataset_root/prices")
    p.add_argument("--emit-price-feature-view", action="store_true",
                   help="Write a dataset with feature-eligible price columns aliased under --price-feature-prefix")
    p.add_argument("--price-feature-prefix", type=str, default="feat_px_",
                   help="Prefix for feature-eligible price columns in the feature-view (default: feat_px_)")

    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_pipeline(args)
