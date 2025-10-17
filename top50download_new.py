#!/usr/bin/env python3
"""
ndl_pipeline_v3.py — Fast & Observable Nasdaq Data Link pipeline for LTR (Colab-optimized)

Highlights
- Visible progress: exporter status polling (creating→fresh), chunk progress bars, per-stage timings.
- Backfill: Bulk Export (qopts.export) with polling + column projection, then fast Arrow CSV parsing.
- Append: Tables API with paginate=True, qopts.per_page=10000, filter push-down, chunked tickers.
- Safe concurrency: default 1 (respect account concurrency limits), override via --concurrency if allowed.
- Colab I/O: write to local staging first, then copy to Drive/S3. Large row groups, fewer small files.
- Shapes: strict MultiIndex (ticker,date), monotone per-ticker, no duplicates — asserts enforce shape.
- Stages: prices, options, fundamentals, etf_options — run any subset or all.
- Designed for LTR: avoids look-ahead in ingest; use your feature stage for PIT shifts if needed.

Docs behind the choices:
- Tables API: 10k rows/call; Python needs paginate=True to go up to 1,000,000 rows.  (docs)
- qopts.per_page / qopts.columns / qopts.export; exporter is asynchronous with status JSON.            (docs)
- Authenticated concurrency is typically 1 (one active, one queued).                                 (docs)
- Only “filterable” columns (e.g., SEP: ticker, date, lastupdated) can be pushed down.                (docs)
"""
from __future__ import annotations

import os, sys, io, json, time, zipfile, argparse, warnings, re, posixpath, shutil, math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlencode

# ---------- minimal bootstrap installs (Colab) ----------
def _ensure(pkg: str, import_name: Optional[str]=None):
    try:
        __import__(import_name or pkg)
    except Exception:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
        __import__(import_name or pkg)

_ensure("nasdaq-data-link", "nasdaqdatalink")
_ensure("pandas", "pandas")
_ensure("pyarrow", "pyarrow")
_ensure("tqdm", "tqdm")
_ensure("requests", "requests")
_ensure("fsspec", "fsspec")

import nasdaqdatalink
import pandas as pd
import numpy as np
import requests
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pyarrow.fs import LocalFileSystem, S3FileSystem, FileType, FileSelector, PyFileSystem, FSSpecHandler
from tqdm import tqdm

warnings.filterwarnings("ignore")
pd.options.display.width = 160
pd.options.display.max_columns = 200

# =========================
# Defaults / constants
# =========================
PRICES_TABLE_DEFAULT  = "SHARADAR/SEP"   # equities OHLCV (filterable: ticker, date, lastupdated)
SF1_TABLE_DEFAULT     = "SHARADAR/SF1"   # fundamentals
OPTIONS_TABLE_DEFAULT = "ORATS/VOL"      # ORATS underlying metrics (premium)

CANON_PRICE_COLS = ["open","high","low","close","closeadj","volume"]
BULK_PRICE_COLS  = ["ticker","date"] + CANON_PRICE_COLS

# =========================
# FileSystem helpers (local, s3, gcs via fsspec)
# =========================
def _collapse_slashes(path: str) -> str:
    # normalize but DO NOT strip the leading slash (important for local absolute paths)
    path = path.replace("\\", "/")
    path = re.sub(r"/+", "/", path)
    return path

def _fs_from_uri(uri: str, s3_endpoint_url: Optional[str] = None):
    from urllib.parse import urlparse
    p = urlparse(uri)
    scheme = (p.scheme or "").lower()

    # Use Wasabi global endpoint by default (us-east-1); override via --s3-endpoint-url
    default_endpoint = "https://s3.wasabisys.com"
    endpoint = (
        s3_endpoint_url
        or os.getenv("AWS_ENDPOINT_URL_S3")
        or os.getenv("AWS_S3_ENDPOINT_URL")
        or os.getenv("AWS_ENDPOINT_URL")
        or default_endpoint
    )

    if scheme == "s3":
        fs = S3FileSystem(
            access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            session_token=os.getenv("AWS_SESSION_TOKEN"),
            region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            endpoint_override=endpoint
        )
        # S3 keys should not start with '/'
        base = _collapse_slashes(f"{p.netloc}{p.path}").lstrip("/")
        return fs, base

    if scheme in ("gs","gcs","abfs","az","adl","azfs"):
        import fsspec
        fs_fsspec, _, paths = fsspec.get_fs_token_paths(uri)
        fs = PyFileSystem(FSSpecHandler(fs_fsspec))
        base = _collapse_slashes(paths[0]).lstrip("/")
        return fs, base

    # Local filesystem: keep absolute path (leading slash)
    fs = LocalFileSystem()
    raw = p.path if p.path else uri
    base = _collapse_slashes(raw)
    return fs, base

# =========================
# Manifest (append windows)
# =========================
@dataclass
class StageState:
    last_date: Optional[str] = None

@dataclass
class Manifest:
    prices: StageState
    options: StageState
    fundamentals: StageState
    etf_options: StageState
    def as_dict(self) -> Dict:
        return {"prices": self.prices.__dict__,
                "options": self.options.__dict__,
                "fundamentals": self.fundamentals.__dict__,
                "etf_options": self.etf_options.__dict__}

def _manifest_path(base_root: str) -> str:
    return _collapse_slashes(posixpath.join(base_root.rstrip("/"), "_manifest.json"))

def _load_manifest(dataset_root: str, s3_endpoint_url: Optional[str]) -> Manifest:
    fs, base = _fs_from_uri(dataset_root, s3_endpoint_url)
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

def _save_manifest(dataset_root: str, m: Manifest, s3_endpoint_url: Optional[str]) -> None:
    fs, base = _fs_from_uri(dataset_root, s3_endpoint_url)
    mpath = _manifest_path(base)
    try: fs.create_dir(posixpath.dirname(mpath), recursive=True)
    except Exception: pass
    with fs.open_output_stream(mpath) as f:
        f.write(json.dumps(m.as_dict(), indent=2, sort_keys=True).encode("utf-8"))

# =========================
# Progress utilities
# =========================
def _ts():
    return time.strftime("%H:%M:%S")

def _print_stage(msg):
    print(f"[{_ts()}] {msg}", flush=True)

# =========================
# HTTP Exporter (progress‑aware)
# =========================
API_BASE = "https://data.nasdaq.com/api/v3/datatables"

def _datatable_json_url(table_code: str, params: Dict[str, object]) -> str:
    # table_code like "SHARADAR/SEP"
    qs = urlencode(params, doseq=True)
    return f"{API_BASE}/{table_code}.json?{qs}"

def export_poll_and_download(table_code: str,
                             filters: Dict[str, object],
                             columns: Optional[List[str]],
                             api_key: str,
                             poll_every_s: float = 2.0,
                             max_wait_s: float = 90.0,
                             out_zip_path: str = "export_tmp.zip") -> Optional[str]:
    """
    Programmatic Bulk Export with visible progress:
      1) Request qopts.export=true + optional qopts.columns
      2) Poll JSON until status in {fresh, regenerating with link}
      3) Download the zip with a tqdm() progress bar
    Returns path to the zip, or None if timed out.
    """
    params = dict(filters or {})
    params["qopts.export"] = "true"
    if columns:
        # ensure ticker/date are included if you filter on them client-side later
        cols = list(dict.fromkeys((columns + ["ticker","date"])))
        params["qopts.columns"] = ",".join(cols)
    params["api_key"] = api_key

    t0 = time.time()
    link = None
    status = None
    while True:
        url = _datatable_json_url(table_code, params)
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            info = r.json().get("datatable_bulk_download", {})
            file_info = info.get("file", {}) or {}
            status = file_info.get("status")
            link   = file_info.get("link")
            _print_stage(f"[export] {table_code} status={status!s}")
            if status == "fresh" and link:
                break
        except Exception as e:
            _print_stage(f"[export] error polling: {e}")
        if time.time() - t0 > max_wait_s:
            _print_stage(f"[export] timeout after {int(max_wait_s)}s waiting for {table_code}")
            return None
        time.sleep(poll_every_s)

    # Download with progress bar
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
                        if total > 0:
                            bar.update(len(chunk))
        return out_zip_path
    except Exception as e:
        _print_stage(f"[export] download error: {e}")
        return None

# =========================
# Fast CSV from zip (Arrow first, pandas fallback)
# =========================
def _read_zip_csvs_fast(zip_path: str, include_cols: Optional[List[str]]) -> pd.DataFrame:
    if (zip_path is None) or (not os.path.exists(zip_path)):
        return pd.DataFrame()
    pa_csv = None
    try:
        import pyarrow.csv as pa_csv
    except Exception:
        pa_csv = None

    inc = list(dict.fromkeys(include_cols or [])) if include_cols else None
    dfs = []
    with zipfile.ZipFile(zip_path, "r") as z:
        csvs = [nm for nm in z.namelist() if nm.lower().endswith(".csv")]
        _print_stage(f"[zip] reading {len(csvs)} CSV file(s)")
        for nm in csvs:
            b = z.read(nm)
            if pa_csv is not None:
                try:
                    tbl = pa_csv.read_csv(
                        pa.BufferReader(b),
                        read_options=pa_csv.ReadOptions(autogenerate_column_names=False),
                        convert_options=(pa_csv.ConvertOptions(include_columns=inc) if inc else pa_csv.ConvertOptions())
                    )
                    dfs.append(tbl.to_pandas())
                    continue
                except Exception:
                    pass
            # pandas fallback
            df = pd.read_csv(io.BytesIO(b), low_memory=False)
            if inc:
                keep = [c for c in df.columns if c in inc]
                df = df[keep] if keep else df
            dfs.append(df)
    return pd.concat(dfs, axis=0, ignore_index=True) if dfs else pd.DataFrame()

# =========================
# Tables API (append path) — robust pagination + progress
# =========================
def get_table_chunked(table_code: str,
                      row_filters: Dict,
                      date_bounds: Optional[Dict] = None,
                      select_cols: Optional[List[str]] = None,
                      per_page: int = 10000,
                      chunk_size: int = 80,
                      concurrency: int = 1) -> pd.DataFrame:
    """
    Chunk the 'ticker' filter to reduce payload per call; use paginate=True; prefer large per_page.
    Progress bar shows chunk progress and rows fetched per chunk.
    """
    rf = dict(row_filters or {})
    tickers = rf.get("ticker", None)
    chunks = [None]
    if isinstance(tickers, (list, tuple, set)):
        tickers = list(dict.fromkeys([str(t).upper().replace("-", ".") for t in tickers]))
        chunks = [tickers[i:i+chunk_size] for i in range(0, len(tickers), chunk_size)]
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
        if select_cols:
            qopts["columns"] = list(dict.fromkeys(select_cols))
        if per_page:
            qopts["per_page"] = int(per_page)  # documented Datatables param
        if qopts:
            kwargs["qopts"] = qopts

        part = nasdaqdatalink.get_table(table_code, paginate=True, **kwargs)  # 10k/call unless paginate=True
        return part if isinstance(part, pd.DataFrame) else pd.DataFrame()

    outs = []
    # Concurrency default=1 (Nasdaq often limits to 1 active request). Keep serial unless you *know* your tier allows more.
    pbar = tqdm(total=len(chunks), desc=f"Fetching {table_code} chunks", unit="chunk")
    try:
        for ch in chunks:
            df = _call(ch)
            pbar.update(1)
            pbar.set_postfix({"rows": 0 if df is None else len(df)})
            if df is not None and not df.empty:
                outs.append(df)
    finally:
        pbar.close()
    return pd.concat(outs, axis=0, ignore_index=True) if outs else pd.DataFrame()

# =========================
# Canonicalization & shape checks
# =========================
DATE_CANDIDATES   = ("date", "tradedate", "tradeDate", "quoteDate", "datekey", "calendardate")
TICKER_CANDIDATES = ("ticker", "underlying", "symbol", "root")

def _find_col(cols: List[str], candidates: Tuple[str, ...]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for k in candidates:
        if k.lower() in lower:
            return lower[k.lower()]
    return None

def _normalize_date_ticker(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
    tcol = _find_col(list(df.columns), TICKER_CANDIDATES)
    dcol = _find_col(list(df.columns), DATE_CANDIDATES)
    if not tcol or not dcol:
        return pd.DataFrame()
    out = df.rename(columns={tcol: "ticker", dcol: "date"}).copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.replace("-", ".", regex=False)
    out["date"]   = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_localize(None)
    return out.dropna(subset=["ticker","date"])

def _ensure_panel_ti(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.loc[:, ~df.columns.duplicated()].copy()
    if not {"date","ticker"}.issubset(df.columns):
        missing = {"date","ticker"} - set(df.columns)
        raise KeyError(f"Missing required columns: {missing}")
    df["date"]   = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df.dropna(subset=["date","ticker"]).drop_duplicates(["ticker","date"], keep="last")
    value_cols = [c for c in df.columns if c not in ("date","ticker")]
    out = df[["ticker","date"] + value_cols].set_index(["ticker","date"]).sort_index()
    ok = (out.reset_index().groupby("ticker", sort=False)["date"].apply(lambda s: s.is_monotonic_increasing).all())
    assert ok, "Per-ticker dates must be monotone increasing"
    assert not out.index.duplicated().any(), "Duplicate (ticker,date) index rows"
    return out

def _synthesize_openadj(df: pd.DataFrame) -> pd.DataFrame:
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

# =========================
# Parquet writer (partitioned) — local staging then copy
# =========================
def _write_partitioned(dataset_root: str,
                       df: pd.DataFrame,
                       mode: str,
                       s3_endpoint_url: Optional[str],
                       verify_listing: bool,
                       compression: str = "snappy",
                       staging_root: Optional[str] = None):
    """
    Write a MultiIndex (ticker,date) DataFrame to a Parquet dataset partitioned by year/month.

    Behavior
    - If dataset_root is S3/Wasabi (s3://...), write DIRECTLY to S3 (skips staging).
    - Otherwise, write to a stage-specific local staging dir first, then copy to dataset_root.
    - Ensures destination directories exist before open_output_stream.

    Args
    ----
    dataset_root: str
        Final destination root (e.g., 's3://bucket/ndl_dataset/prices' or '/content/drive/.../prices').
    df: pd.DataFrame
        Must have MultiIndex ['ticker','date'].
    mode: {'append','overwrite'}
        Append uses overwrite_or_ignore; overwrite deletes matching partitions.
    s3_endpoint_url: Optional[str]
        S3-compatible endpoint (e.g., 'https://s3.wasabisys.com'); can also come from env.
    verify_listing: bool
        If True, list written files after write/copy (slower).
    compression: str
        Parquet compression ('snappy' default).
    staging_root: Optional[str]
        Local staging root; ignored when writing directly to S3.
    """
    from pyarrow.fs import LocalFileSystem, S3FileSystem, FileSelector, FileType

    log = globals().get("_print_stage", print)

    if df is None or df.empty:
        return

    # Strict LTR shape: MultiIndex(ticker,date)
    assert isinstance(df.index, pd.MultiIndex) and list(df.index.names) == ["ticker", "date"], \
        "Expected MultiIndex ['ticker','date']"

    # Prepare partition columns
    out = df.reset_index().copy()
    out["date"]  = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_localize(None)
    out["year"]  = out["date"].dt.year.astype("int16")
    out["month"] = out["date"].dt.month.astype("int8")
    table = pa.Table.from_pandas(out, preserve_index=False)

    # Determine if destination is S3 (Wasabi) and resolve stage name
    dst_fs, dst_base = _fs_from_uri(dataset_root, s3_endpoint_url)
    dst_base = _collapse_slashes(dst_base)
    stage_name = posixpath.basename(dst_base.rstrip("/")) or "stage"

    # Choose target: direct-to-S3 if S3; else use staging (if provided) then copy
    if isinstance(dst_fs, S3FileSystem):
        target_uri = dataset_root    # direct write to S3/Wasabi — fastest path
        staging_target_uri = None
    else:
        staging_target_uri = posixpath.join(staging_root, stage_name) if staging_root else None
        target_uri = staging_target_uri or dataset_root

    fs, base = _fs_from_uri(target_uri, s3_endpoint_url)
    base = _collapse_slashes(base)

    # Normalize absolute path for local FS
    if isinstance(fs, LocalFileSystem) and not base.startswith("/"):
        base = "/" + base

    # Ensure base directory exists before writing
    try:
        fs.create_dir(base, recursive=True)
    except Exception:
        pass

    # Parquet write options
    fmt = ds.ParquetFileFormat()
    try:
        file_opts = fmt.make_write_options(compression=compression)
    except Exception:
        file_opts = None

    write_kwargs = dict(
        base_dir=base,
        format=fmt,
        partitioning=["year", "month"],
        existing_data_behavior=("overwrite_or_ignore" if mode == "append" else "delete_matching"),
        filesystem=fs,
        use_threads=True,
        max_open_files=64,
        max_rows_per_file=1_000_000,
        min_rows_per_group=50_000,
        max_rows_per_group=250_000,
    )
    if file_opts is not None:
        write_kwargs["file_options"] = file_opts

    # Visible timing for the write
    log(f"[write] fs={type(fs).__name__} base='{base}' rows={len(out)} cols={out.shape[1]}")
    t0 = time.time()
    try:
        ds.write_dataset(table, **write_kwargs)
    except TypeError:
        for k in ("min_rows_per_group", "max_rows_per_group", "max_rows_per_file",
                  "max_open_files", "use_threads", "file_options"):
            write_kwargs.pop(k, None)
        ds.write_dataset(table, **write_kwargs)
    log(f"[write] done in {time.time()-t0:.2f}s")

    # Optional verify of the write target
    if verify_listing:
        try:
            infos = fs.get_file_info(FileSelector(base, recursive=True))
            wrote = [i.path for i in infos if i.type == FileType.File and i.path.endswith(".parquet")]
            log(f"[verify] parquet files under '{base}': {len(wrote)}")
        except Exception as e:
            log(f"[verify] listing failed: {e}")

    # Copy staged → destination (only when not direct-to-S3 and staging was used)
    if staging_target_uri and (staging_target_uri != dataset_root):
        src_fs, src_base = _fs_from_uri(staging_target_uri, s3_endpoint_url)
        src_base = _collapse_slashes(src_base)
        if isinstance(src_fs, LocalFileSystem) and not src_base.startswith("/"):
            src_base = "/" + src_base
        if isinstance(dst_fs, LocalFileSystem) and not dst_base.startswith("/"):
            dst_base = "/" + dst_base

        log(f"[copy] staged parquet from '{src_base}' → '{dst_base}'")

        sel = FileSelector(src_base, recursive=True)
        for info in src_fs.get_file_info(sel):
            if info.type != FileType.File or not info.path.endswith(".parquet"):
                continue
            rel = info.path[len(src_base):].lstrip("/")
            dst_path = posixpath.join(dst_base, rel)

            # Ensure destination directory exists
            try:
                dst_fs.create_dir(posixpath.dirname(dst_path), recursive=True)
            except Exception:
                pass

            with src_fs.open_input_stream(info.path) as r, \
                 dst_fs.open_output_stream(dst_path) as w:
                w.write(r.read())

        # Quick post-copy verify
        try:
            infos = dst_fs.get_file_info(FileSelector(dst_base, recursive=True))
            wrote = [i.path for i in infos if i.type == FileType.File and i.path.endswith(".parquet")]
            log(f"[verify-dest] files under destination: {len(wrote)}")
        except Exception as e:
            log(f"[verify-dest] listing failed: {e}")

# =========================
# Fetchers — prices / options / fundamentals / etf_options
# =========================
def fetch_prices(tickers: List[str], start: str, end: str,
                 table: str, use_export: bool, api_key: str,
                 concurrency: int, max_export_wait_s: float) -> pd.DataFrame:
    tickers = [str(t).upper().replace("-", ".") for t in (tickers or [])]
    if not tickers: return pd.DataFrame()

    if use_export:
        zip_path = export_poll_and_download(
            table_code=table,
            filters={"ticker": ",".join(tickers), "date.gte": start, "date.lte": end},
            columns=BULK_PRICE_COLS,
            api_key=api_key,
            max_wait_s=max_export_wait_s,
            out_zip_path="prices_export.zip"
        )
        if zip_path:
            df = _read_zip_csvs_fast(zip_path, include_cols=BULK_PRICE_COLS)
        else:
            _print_stage("[prices] exporter timed out; falling back to API pagination.")
            df = get_table_chunked(
                table_code=table,
                row_filters={"ticker": tickers},
                date_bounds={"gte": start, "lte": end},
                select_cols=BULK_PRICE_COLS,
                per_page=10000,
                chunk_size=80,
                concurrency=concurrency
            )
    else:
        df = get_table_chunked(
            table_code=table,
            row_filters={"ticker": tickers},
            date_bounds={"gte": start, "lte": end},
            select_cols=BULK_PRICE_COLS,
            per_page=10000,
            chunk_size=80,
            concurrency=concurrency
        )

    df = _normalize_date_ticker(df)
    if df.empty:
        return df
    keep_vals = [c for c in CANON_PRICE_COLS if c in df.columns]
    out = df[["ticker","date"] + keep_vals].drop_duplicates(["ticker","date"])
    out = _ensure_panel_ti(out)
    out = _synthesize_openadj(out)
    return out

def fetch_options(tickers: List[str], start: str, end: str,
                  table: str, use_export: bool, api_key: str,
                  concurrency: int, max_export_wait_s: float) -> pd.DataFrame:
    tickers = [str(t).upper().replace("-", ".") for t in (tickers or [])]
    if not tickers: return pd.DataFrame()

    if use_export:
        zip_path = export_poll_and_download(
            table_code=table,
            filters={"ticker": ",".join(tickers), "tradedate.gte": start, "tradedate.lte": end},
            columns=None,
            api_key=api_key,
            max_wait_s=max_export_wait_s,
            out_zip_path="options_export.zip"
        )
        df = _read_zip_csvs_fast(zip_path, include_cols=None) if zip_path else pd.DataFrame()
        if df.empty:
            _print_stage("[options] exporter empty/timeout; falling back to API.")
    if not use_export or df.empty:
        df = get_table_chunked(
            table_code=table,
            row_filters={"ticker": tickers, "tradedate": {"gte": start, "lte": end}},
            date_bounds=None,
            select_cols=None,
            per_page=10000,
            chunk_size=60,
            concurrency=concurrency
        )
    df = _normalize_date_ticker(df)
    if df.empty:
        return df
    # drop extremely wide option-structure columns (keeps underlying-level summaries faster)
    drop_like = ("strike","expiry","expiration","days","moneyness")
    drops = [c for c in df.columns if any(x in c.lower() for x in drop_like)]
    df = df.drop(columns=drops, errors="ignore")
    return _ensure_panel_ti(df).add_prefix("OPT_")

def fetch_sf1(tickers: list[str], start: str, end: str,
              table: str, dimensions: list[str], use_export: bool, api_key: str,
              concurrency: int, max_export_wait_s: float) -> pd.DataFrame:
    # normalize tickers once
    tickers = [str(t).upper().replace("-", ".") for t in (tickers or [])]
    if not tickers:
        return pd.DataFrame()

    if use_export:
        # FIX: removed the extra quote before .join(dimensions)
        zip_path = export_poll_and_download(
            table_code=table,
            filters={
                "ticker": ",".join(tickers),
                "dimension": ",".join(dimensions),   # <-- fixed here
                "datekey.gte": start,
                "datekey.lte": end
            },
            columns=None,
            api_key=api_key,
            max_wait_s=max_export_wait_s,
            out_zip_path="sf1_export.zip"
        )
        df = _read_zip_csvs_fast(zip_path, include_cols=None) if zip_path else pd.DataFrame()
        if df.empty:
            _print_stage("[fundamentals] exporter empty/timeout; falling back to API.")
    if (not use_export) or df.empty:
        df = get_table_chunked(
            table_code=table,
            row_filters={"ticker": tickers, "dimension": dimensions, "datekey": {"gte": start, "lte": end}},
            date_bounds=None,
            select_cols=None,
            per_page=10000,
            chunk_size=60,
            concurrency=concurrency
        )

    df = _normalize_date_ticker(df)
    if df.empty:
        return df

    if "dimension" not in df.columns:
        value_cols = [c for c in df.columns if c not in ("ticker","date","lastupdated","reportperiod","calendardate","datekey")]
        out = df[["ticker","date"] + value_cols].rename(columns={c: f"FUND_{c}" for c in value_cols})
        return _ensure_panel_ti(out)

    value_cols = [c for c in df.columns if c not in ("ticker","date","dimension","lastupdated","reportperiod","calendardate","datekey")]
    if not value_cols:
        return _ensure_panel_ti(df[["ticker","date"]].drop_duplicates())

    piv = (df.pivot_table(index=["ticker","date"], columns="dimension", values=value_cols, aggfunc="last")
             .sort_index())
    piv.columns = [f"FUND_{val}__{dim}" for (val, dim) in piv.columns.to_flat_index()]
    return _ensure_panel_ti(piv.reset_index())

def fetch_etf_options(etf_tickers: List[str], start: str, end: str,
                      table: str, use_export: bool, api_key: str,
                      concurrency: int, max_export_wait_s: float) -> pd.DataFrame:
    t = [str(x).upper() for x in (etf_tickers or [])]
    if not t: return pd.DataFrame()
    if use_export:
        zip_path = export_poll_and_download(
            table_code=table,
            filters={"ticker": ",".join(t), "tradedate.gte": start, "tradedate.lte": end},
            columns=None,
            api_key=api_key,
            max_wait_s=max_export_wait_s,
            out_zip_path="etf_options_export.zip"
        )
        df = _read_zip_csvs_fast(zip_path, include_cols=None) if zip_path else pd.DataFrame()
        if df.empty:
            _print_stage("[etf_options] exporter empty/timeout; falling back to API.")
    if (not use_export) or df.empty:
        df = get_table_chunked(
            table_code=table,
            row_filters={"ticker": t, "tradedate": {"gte": start, "lte": end}},
            date_bounds=None,
            select_cols=None,
            per_page=10000,
            chunk_size=60,
            concurrency=concurrency
        )
    df = _normalize_date_ticker(df)
    if df.empty:
        return df
    num_cols = df.select_dtypes(include=[np.number]).columns.difference(["ticker"]).tolist()
    if not num_cols: return pd.DataFrame()
    ctx = df.groupby("date", as_index=False)[num_cols].mean(numeric_only=True)
    ctx = ctx.add_prefix("CTX_").rename(columns={"CTX_date":"date"})
    return ctx  # date-only; broadcast later

# =========================
# Dates & manifest helpers
# =========================
def _next_start_date(manifest_date: Optional[str], user_start: Optional[str]) -> str:
    if user_start and str(user_start).lower() != "auto":
        return user_start
    if manifest_date:
        dt = pd.to_datetime(manifest_date) + pd.tseries.offsets.BDay(1)
        return dt.strftime("%Y-%m-%d")
    return "2010-01-01"

def _end_date(user_end: Optional[str], prev_bday: bool) -> str:
    if user_end and user_end != "auto":
        return str(user_end)
    today = pd.Timestamp.today(tz="UTC").normalize()
    return (today - pd.tseries.offsets.BDay(1) if prev_bday else today).strftime("%Y-%m-%d")

# =========================
# Runner
# =========================
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
    _print_stage(f"[plan] tickers={len(tickers)} sample={tickers[:8]}")

    # Stages
    stages = [s.strip().lower() for s in args.stages.split(",")]
    use_bulk = not args.append
    if args.end == "auto" and not args.prev_bday:
        args.prev_bday = True

    # Manifest & windows
    manifest = _load_manifest(args.dataset_root, args.s3_endpoint_url) if args.append else Manifest(StageState(),StageState(),StageState(),StageState())
    end = _end_date(args.end, prev_bday=args.prev_bday)

    # Staging (Colab tip: write local, then copy to Drive/S3)
    staging_root = args.staging_root.strip() or None
    if staging_root:
        os.makedirs(staging_root, exist_ok=True)

    merged = pd.DataFrame()

    # === PRICES ===
    if "prices" in stages or "all" in stages:
        start_p = _next_start_date(manifest.prices.last_date, args.start)
        _print_stage(f"[prices] table={args.prices_table} method={'BULK' if use_bulk else 'API'} window={start_p}→{end} n_tickers={len(tickers)}")
        t0 = time.time()
        prices = fetch_prices(tickers, start_p, end, args.prices_table, use_bulk, api_key, args.concurrency, args.max_export_wait_s)
        _print_stage(f"[prices] rows={0 if prices is None else prices.shape[0]} cols={0 if prices is None else prices.shape[1]} elapsed={time.time()-t0:.1f}s")
        if not prices.empty:
            merged = prices
            _write_partitioned(args.dataset_root + "/prices", prices, mode=("append" if args.append else "overwrite"),
                               s3_endpoint_url=args.s3_endpoint_url, staging_root=staging_root,
                               compression=args.parquet_compression, verify_listing=args.verify_write)
            manifest.prices.last_date = str(prices.index.get_level_values("date").max().date())
        else:
            _print_stage("[prices] nothing fetched.")

    # === OPTIONS ===
    if "options" in stages or "all" in stages:
        start_o = _next_start_date(manifest.options.last_date, args.start)
        _print_stage(f"[options] table={args.options_table} method={'BULK' if use_bulk else 'API'} window={start_o}→{end} n_tickers={len(tickers)}")
        t0 = time.time()
        opt = fetch_options(tickers, start_o, end, args.options_table, use_bulk, api_key, args.concurrency, args.max_export_wait_s)
        _print_stage(f"[options] rows={0 if opt is None else opt.shape[0]} cols={0 if opt is None else opt.shape[1]} elapsed={time.time()-t0:.1f}s")
        if not opt.empty:
            merged = opt if merged.empty else merged.join(opt, how="left")
            _write_partitioned(args.dataset_root + "/options", opt, mode=("append" if args.append else "overwrite"),
                               s3_endpoint_url=args.s3_endpoint_url, staging_root=staging_root,
                               compression=args.parquet_compression, verify_listing=args.verify_write)
            manifest.options.last_date = str(opt.index.get_level_values("date").max().date())

    # === FUNDAMENTALS (SF1) ===
    if "fundamentals" in stages or "all" in stages:
        dims = [d.strip() for d in (args.sf1_dimensions or "ARQ,ARY,ART").split(",") if d.strip()]
        start_f = _next_start_date(manifest.fundamentals.last_date, args.start)
        _print_stage(f"[fundamentals] table={args.sf1_table} dims={','.join(dims)} method={'BULK' if use_bulk else 'API'} window={start_f}→{end} n_tickers={len(tickers)}")
        t0 = time.time()
        sf1 = fetch_sf1(tickers, start_f, end, args.sf1_table, dims, use_bulk, api_key, args.concurrency, args.max_export_wait_s)
        _print_stage(f"[fundamentals] rows={0 if sf1 is None else sf1.shape[0]} cols={0 if sf1 is None else sf1.shape[1]} elapsed={time.time()-t0:.1f}s")
        if not sf1.empty:
            merged = sf1 if merged.empty else merged.join(sf1, how="left")
            _write_partitioned(args.dataset_root + "/fundamentals", sf1, mode=("append" if args.append else "overwrite"),
                               s3_endpoint_url=args.s3_endpoint_url, staging_root=staging_root,
                               compression=args.parquet_compression, verify_listing=args.verify_write)
            manifest.fundamentals.last_date = str(sf1.index.get_level_values("date").max().date())

    # === ETF OPTIONS (broadcast by date) ===
    if "etf_options" in stages or "all" in stages:
        etfs = [t.strip() for t in (args.etf_option_tickers or "").split(",") if t.strip()]
        if etfs:
            start_e = _next_start_date(manifest.etf_options.last_date, args.start)
            _print_stage(f"[etf_options] table={args.options_table} method={'BULK' if use_bulk else 'API'} window={start_e}→{end} n_tickers={len(etfs)}")
            t0 = time.time()
            ctx = fetch_etf_options(etfs, start_e, end, args.options_table, use_bulk, api_key, args.concurrency, args.max_export_wait_s)
            _print_stage(f"[etf_options] ctx_rows={0 if ctx is None else ctx.shape[0]} elapsed={time.time()-t0:.1f}s")
            if not merged.empty and not ctx.empty:
                left = merged.reset_index()
                merged = _ensure_panel_ti(left.merge(ctx, on="date", how="left").rename(columns=lambda c: c if c=="date" or c=="ticker" else c))
            if not merged.empty:
                _write_partitioned(args.dataset_root + "/etf_options_ctx",
                                   _ensure_panel_ti(merged.reset_index()),
                                   mode=("append" if args.append else "overwrite"),
                                   s3_endpoint_url=args.s3_endpoint_url, staging_root=staging_root,
                                   compression=args.parquet_compression, verify_listing=False)
                manifest.etf_options.last_date = str(merged.index.get_level_values("date").max().date())

    # Save merged snapshot (optional) + manifest
    if not merged.empty:
        for col in [*CANON_PRICE_COLS, "openadj"]:
            if col not in merged.columns:
                merged[col] = np.nan if col != "openadj" else _synthesize_openadj(merged)["openadj"]
        merged = merged[~merged.index.duplicated(keep="last")]
        if args.output_file:
            fs, base = _fs_from_uri(args.output_file, args.s3_endpoint_url)
            pq.write_table(pa.Table.from_pandas(merged.reset_index().sort_values(["ticker","date"])),
                           base, filesystem=fs, compression=args.parquet_compression)
        _save_manifest(args.dataset_root, manifest, args.s3_endpoint_url)
        nrows, ncols = merged.shape
        ntickers = merged.index.get_level_values("ticker").nunique()
        ndates   = merged.index.get_level_values("date").nunique()
        dmin     = str(merged.index.get_level_values("date").min().date())
        dmax     = str(merged.index.get_level_values("date").max().date())
        _print_stage(f"[done] {dmin}..{dmax} shape={nrows:,} x {ncols:,} (#tickers={ntickers}, #dates={ndates}) total_elapsed={time.time()-t_start:.1f}s")
    else:
        _print_stage("[done] No rows fetched.")

# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Nasdaq Data Link — fast/observable LTR ingest (Colab-optimized).")
    # Inputs
    p.add_argument("--tickers-file", type=str, default="", help="CSV with 'ticker' column (or first column).")
    p.add_argument("--tickers-list", type=str, default="", help="Comma-separated tickers.")
    p.add_argument("--etf-option-tickers", type=str, default="", help="Comma-separated ETFs for ctx broadcast (e.g., SPY,QQQ).")
    # Stages
    p.add_argument("--stages", type=str, default="prices", help="Comma-separated: prices,options,fundamentals,etf_options or 'all'.")
    # Dates
    p.add_argument("--start", type=str, default="auto", help="'YYYY-MM-DD' or 'auto' (after manifest).")
    p.add_argument("--end",   type=str, default="auto", help="'YYYY-MM-DD' or 'auto' (today or prev-bday).")
    p.add_argument("--prev-bday", action="store_true", help="End on previous business day (safer for daily runs).")
    p.add_argument("--append", action="store_true", help="Append mode: uses Tables API for deltas since manifest.")
    # Output
    p.add_argument("--dataset-root", type=str, required=True, help="Parquet dataset root (local or cloud URI).")
    p.add_argument("--output-file",  type=str, default="", help="Optional merged snapshot parquet path (local or cloud URI).")
    p.add_argument("--staging-root", type=str, default="", help="Local staging dir for fast writes (e.g., /content/_stage).")
    p.add_argument("--parquet-compression", type=str, default="snappy", choices=["snappy","zstd","gzip","lz4","none"])
    # Tables
    p.add_argument("--prices-table",  type=str, default=PRICES_TABLE_DEFAULT)
    p.add_argument("--options-table", type=str, default=OPTIONS_TABLE_DEFAULT)
    p.add_argument("--sf1-table",     type=str, default=SF1_TABLE_DEFAULT)
    p.add_argument("--sf1-dimensions", type=str, default="ARQ,ARY,ART")
    # Performance
    p.add_argument("--concurrency", type=int, default=1, help="Threads for chunked Tables API on append. Default 1 (Nasdaq concurrency).")
    p.add_argument("--max-export-wait-s", type=float, default=90.0, help="Max seconds to wait for exporter before falling back.")
    p.add_argument("--s3-endpoint-url", type=str, default="", help="Custom S3-compatible endpoint (e.g., Wasabi/R2/MinIO).")
    p.add_argument("--verify-write", action="store_true", help="List files after Parquet write (slower).")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
