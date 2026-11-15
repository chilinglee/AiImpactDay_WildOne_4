"""
野灣捐款資料 Cohort / 留存 / 支持時刻 分析管線（對應 AI Impact Day 官方資料 野灣_示範資料_v1.xlsx）

用法：
    python yewan_cohort.py --data <xlsx 檔 或 CSV 資料夾> --output-dir <輸出資料夾> [--as-of 2026-10-31]

    給 xlsx 時，會先轉成 CSV 放在 <輸出資料夾>/raw/（對帳測試也讀這份）。

輸出：
    00_資料品質檢查.csv          每項檢查的嚴重度、筆數、比例、說明
    01_捐款明細_整合.csv          ②③ 兩張捐款表 + 聯絡人屬性 + Cohort 欄位
    02_捐款人主檔.csv             每位聯絡人一列（首捐、捐款型態、是否新捐款人、是否年輕支持者…）
    03_定期方案主檔.csv           每個方案一列：由 ② 的定期捐款編號 + ④ 明細「還原」（④ 一人只有一列）
    04_新捐款人明細.csv           期間新捐款人的後續行為
    t1_* / t2_* / t3_*.csv        任務一/二/三
    summary.json                  關鍵數字 + 每個發現的「可信度判定」（給 AI 撰寫敘述用）
    report.html                   互動報告

設計原則：
    1. 先驗證、再分析：每個發現都附上樣本數與統計檢定，未達顯著就標示「不宜下結論」。
    2. 定義寫死在程式裡、也寫在 references/metrics.md：同一個數字永遠同一個算法。
    3. 只依賴 pandas / numpy。
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
FILES = {
    "contacts": "Contacts.csv",
    "recurring": "Recurring_Donations.csv",
    "one_time": "One_Time_Donations.csv",
    "plans": "Recurring_Details.csv",
    "events": "Event_Timeline.csv",
}

# 官方欄位名稱 → 程式內部名稱。欄位名稱有變時，在這裡加別名即可。
# 「A / B」這種合併欄位（例如「執行期數 / 約定總期數」）會先自動拆成 A、B 兩欄，再套用別名。
COLUMN_ALIASES = {
    "捐款建立日期": "捐款日期", "總金額": "金額", "定期定額狀態": "狀態", "最新 建立日期": "最新建立日期",
    "活動或事件": "事件名稱", "類型": "事件類型", "年齡級距": "年齡",
    "SystemID": "系統編號", "系統編碼": "系統編號", "Date": "捐款日期", "Amount": "金額",
}
ID_COLS = ["系統編號", "捐款編號", "定期捐款編號"]

AGE_MISSING = {"未填寫", ""}
YOUNG_AGE_BANDS = {"5-9歲", "10-14歲", "15-19歲", "20-24歲", "20歲以下"}   # 任務一：25 歲以下
ACTIVE_STATUSES = {"進行中", "推定進行中（明細未列）"}
STOPPED_STATUSES = {"暫停中", "已失敗", "已取消", "待處理", "推定停止（明細未列）"}
COMPLETED_STATUS = "已完成"
CHURN_TYPE = {"待處理": "被動：第一次扣款就失敗", "已失敗": "被動：扣款失敗",
              "暫停中": "暫停（原因不明）", "已取消": "主動取消", "推定停止（明細未列）": "推定停止（明細未列）"}
SILENT_DAYS = 45            # 進行中但超過幾天沒扣款 = 靜默流失
INACTIVITY_BUCKETS = [(0, 7, "0-7天"), (8, 30, "8-30天"), (31, 60, "31-60天"),
                      (61, 90, "61-90天"), (91, 100_000, "90天以上")]
EVENT_PRE_DAYS, EVENT_POST_DAYS = 14, 7
BIG_GIFT = 30_000           # 官方說明頁：③ 有 9 筆 3 萬元以上
SPIKE_P = 0.001            # 單日高峰的門檻（很嚴，避免把雜訊當高峰）
ALPHA = 0.05
AMBIGUOUS_TOWNS = {"東區", "西區", "南區", "北區", "中區", "中正區", "中山區", "信義區", "大安區",
                   "仁愛區", "安樂區", "七堵區", "暖暖區", "大同區", "東勢區", "東勢鄉"}


# ---------------------------------------------------------------------------
# 小工具：統計檢定（不依賴 scipy）
# ---------------------------------------------------------------------------
def _gammainc_upper_reg(a: float, x: float) -> float:
    """Q(a, x) 正則化上不完全 gamma，用於卡方 p 值。"""
    if x <= 0:
        return 1.0
    if x < a + 1:  # series for P
        term = total = 1.0 / a
        n = a
        for _ in range(500):
            n += 1
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-12:
                break
        p = total * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return max(0.0, 1.0 - p)
    b = x + 1 - a
    c = 1 / 1e-300
    d = 1 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-12:
            break
    return min(1.0, h * math.exp(-x + a * math.log(x) - math.lgamma(a)))


def chi2_independence(table: pd.DataFrame) -> tuple[float, int, float]:
    """回傳 (卡方值, 自由度, p 值)。table 為次數交叉表。"""
    obs = table.to_numpy(dtype=float)
    obs = obs[obs.sum(axis=1) > 0][:, obs.sum(axis=0) > 0]
    if obs.shape[0] < 2 or obs.shape[1] < 2:
        return 0.0, 0, 1.0
    exp = obs.sum(axis=1, keepdims=True) * obs.sum(axis=0, keepdims=True) / obs.sum()
    stat = float(((obs - exp) ** 2 / exp).sum())
    dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    return stat, dof, _gammainc_upper_reg(dof / 2, stat / 2)


def poisson_upper_p(observed: int, expected: float) -> float:
    """P(X >= observed | Poisson(expected))：活動後的人數是否「多到不像巧合」。"""
    if expected <= 0:
        return 0.0 if observed > 0 else 1.0
    cdf = sum(math.exp(k * math.log(expected) - expected - math.lgamma(k + 1))
              for k in range(observed))
    return max(0.0, 1.0 - cdf)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def verdict(p: float, n_min: int, min_n: int = 30) -> str:
    if n_min < min_n:
        return "樣本太少：不宜下結論"
    return "差異顯著" if p < ALPHA else "無顯著差異：不宜下結論"


def _mode(s: pd.Series):
    m = s.dropna().mode()
    return m.iat[0] if len(m) else np.nan


# ---------------------------------------------------------------------------
# 1. 載入
# ---------------------------------------------------------------------------
@dataclass
class Data:
    contacts: pd.DataFrame
    recurring: pd.DataFrame
    one_time: pd.DataFrame
    plans: pd.DataFrame          # ④ 原始明細（一人一列）
    events: pd.DataFrame
    as_of: pd.Timestamp = None
    period_start: pd.Timestamp = None
    contacts_raw_n: int = 0
    notes: list = field(default_factory=list)


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df = df.apply(lambda s: s.str.strip())
    df = df.where(~df.isin(["", "nan", "NaT", "None"]), np.nan)
    df.columns = [c.strip() for c in df.columns]
    for col in list(df.columns):            # 拆開「A / B」合併欄位
        names = [x.strip() for x in col.split("/")]
        if len(names) == 2 and all(names):
            parts = df[col].str.split("/", n=1, expand=True).reindex(columns=[0, 1])
            df[names[0]] = parts[0].str.strip()
            df[names[1]] = parts[1].str.strip()
            df = df.drop(columns=col)
    df = df.rename(columns=lambda c: COLUMN_ALIASES.get(c, c))
    for col in ID_COLS:
        if col in df:
            df[col] = df[col].map(lambda v: re.sub(r"\.0$", "", v) if isinstance(v, str) else v)
    return df


def _num(df: pd.DataFrame, cols):
    for col in cols:
        if col in df:
            df[col + "_raw"] = df[col]
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _dt(df: pd.DataFrame, cols):
    for col in cols:
        if col in df:
            df[col + "_raw"] = df[col]
            df[col] = pd.to_datetime(df[col], errors="coerce")


def _parse_events(e: pd.DataFrame, period_start, as_of) -> pd.DataFrame:
    """活動時間軸：『每月第一週』展開成每月 1 日；備註裡的『至 M/D』當作活動結束日。"""
    rows = []
    for _, ev in e.iterrows():
        raw = str(ev.get("日期", "")).strip()
        d = pd.to_datetime(raw, errors="coerce")
        if pd.notna(d):
            starts = [d]
        elif "每月" in raw:
            starts = list(pd.date_range(period_start, as_of, freq="MS"))
        else:
            continue
        note = str(ev.get("備註", "") or "")
        for s in starts:
            end = s + pd.Timedelta(days=EVENT_POST_DAYS - 1)
            m = re.search(r"至\s*(\d{1,2})\s*/\s*(\d{1,2})", note)
            if m and len(starts) == 1:
                cand = pd.Timestamp(year=s.year, month=int(m.group(1)), day=int(m.group(2)))
                end = cand if cand >= s else cand + pd.DateOffset(years=1)
            name = str(ev.get("事件名稱", ""))
            rows.append({"日期": s.normalize(), "結束日期": end.normalize(),
                         "事件名稱": name + (f"（{s:%Y-%m}）" if len(starts) > 1 else ""),
                         "事件類型": ev.get("事件類型", ""), "管道": ev.get("管道", ""), "備註": ev.get("備註", ""),
                         "日期原文": raw})
    out = pd.DataFrame(rows).sort_values("日期").reset_index(drop=True)
    out.insert(0, "事件編號", [f"EV{i + 1:02d}" for i in range(len(out))])
    return out


def prepare_data(src: Path, out_dir: Path) -> Path:
    """src 是 xlsx → 轉成 CSV 放到 out_dir/raw；是資料夾 → 直接用。"""
    if src.is_file() and src.suffix.lower() in (".xlsx", ".xlsm"):
        import importlib.util
        spec = importlib.util.spec_from_file_location("fetch_sheet", Path(__file__).with_name("fetch_sheet.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        raw = out_dir / "raw"
        mod.xlsx_to_csvs(src, raw)
        return raw
    return src


def load(data_dir: Path, as_of: str | None = None) -> Data:
    raw = {k: _read(data_dir / v) for k, v in FILES.items()}
    c, r, o, p, e = (raw[k] for k in ["contacts", "recurring", "one_time", "plans", "events"])
    _dt(c, ["聯絡人建立日期"])
    _dt(r, ["捐款日期"])
    _dt(o, ["捐款日期"])
    _dt(p, ["開始日期", "結束日期", "失敗或取消日期", "最新建立日期", "最近一筆收到日期", "最近失敗日期"])
    _num(r, ["金額"])
    _num(o, ["金額"])
    p["不限期"] = p["約定總期數"].eq("不限")
    p["約定總期數"] = p["約定總期數"].where(~p["不限期"], "0")
    p["剩餘期數"] = p["剩餘期數"].where(p["剩餘期數"].ne("不限"))
    _num(p, ["約定金額", "執行期數", "約定總期數", "剩餘期數", "成功次數", "請款次數", "收到金額總計"])
    # ② 交易編號 = 「方案第一筆捐款編號_第幾期」
    tid = r["交易編號"].str.extract(r"^(\d+)_(\d+)$")
    r["交易編號前段"] = tid[0]
    r["期數"] = pd.to_numeric(tid[1], errors="coerce")
    # 年齡「未填寫」視為缺漏
    for df in (c, r, o):
        df["年齡已填"] = df["年齡"].notna() & ~df["年齡"].isin(AGE_MISSING)
    d = Data(c, r, o, p, e)
    d.contacts_raw_n = len(c)
    all_dates = pd.concat([r["捐款日期"], o["捐款日期"]]).dropna()
    d.period_start = all_dates.min().normalize()
    d.as_of = pd.Timestamp(as_of) if as_of else all_dates.max().normalize()
    d.events = _parse_events(e, d.period_start, d.as_of)
    # 聯絡人系統編號重複：保留最早建立的一列（品質檢查會另外報告）
    d.contacts = c.sort_values("聯絡人建立日期").drop_duplicates("系統編號", keep="first").reset_index(drop=True)
    return d


# ---------------------------------------------------------------------------
# 2. 資料品質檢查
# ---------------------------------------------------------------------------
def validate(d: Data, plans: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []

    def add(table, check, bad, total, severity, note, examples=None):
        bad = int(bad)
        rows.append({
            "資料表": table, "檢查項目": check, "異常筆數": bad, "總筆數": int(total),
            "異常比例": round(bad / total, 4) if total else 0.0, "嚴重度": severity if bad else "OK",
            "對分析的影響": note, "範例": ", ".join(map(str, (examples or [])[:5])) if bad else "",
        })

    c, r, o, p = d.contacts, d.recurring, d.one_time, d.plans
    # --- 主鍵 / 外鍵
    craw_dup = d.contacts_raw_n - len(c)
    add("①聯絡人", "系統編號 重複", craw_dup, d.contacts_raw_n, "WARN",
        "已保留最早建立的一列；欄位說明已註明有一筆重複")
    for name, df in [("②定期紀錄", r), ("③單筆紀錄", o)]:
        dup = df["捐款編號"].duplicated(keep=False)
        add(name, "捐款編號 重複", dup.sum(), len(df), "ERROR", "主鍵重複會讓金額重複計算",
            df.loc[dup, "捐款編號"].unique().tolist())
    both = set(r["捐款編號"]) & set(o["捐款編號"])
    add("②③", "捐款編號 同時出現在定期與單筆", len(both), len(r) + len(o), "ERROR",
        "兩表共用同一組編號，不應重疊", sorted(both))
    dup = p["系統編號"].duplicated(keep=False)
    add("④定期明細", "系統編號 重複（應為一人一列）", dup.sum(), len(p), "WARN", "", p.loc[dup, "系統編號"].tolist())
    dup = p["定期捐款編號"].duplicated(keep=False)
    add("④定期明細", "定期捐款編號 重複", dup.sum(), len(p), "ERROR", "", p.loc[dup, "定期捐款編號"].tolist())
    ids = set(c["系統編號"])
    for name, df in [("②定期紀錄", r), ("③單筆紀錄", o), ("④定期明細", p)]:
        bad = ~df["系統編號"].isin(ids)
        add(name, "系統編號 對不到聯絡人", bad.sum(), len(df), "ERROR",
            "捐款會缺年齡/性別/地區", df.loc[bad, "系統編號"].tolist())
    multi = r.groupby("定期捐款編號")["系統編號"].nunique()
    add("②定期紀錄", "同一定期捐款編號 對到多位捐款人", (multi > 1).sum(), len(multi), "ERROR",
        "方案歸屬錯誤", multi[multi > 1].index.tolist())
    tdup = r["交易編號"].duplicated(keep=False)
    add("②定期紀錄", "交易編號 重複（不同方案共用同一前段）", tdup.sum(), len(r), "INFO",
        "交易編號不是唯一鍵：主鍵用 捐款編號；方案用 定期捐款編號；期數取交易編號後段",
        r.loc[tdup, "交易編號"].unique().tolist())

    # --- 格式
    for name, df, col in [("②定期紀錄", r, "捐款日期"), ("③單筆紀錄", o, "捐款日期"),
                          ("①聯絡人", c, "聯絡人建立日期"), ("④定期明細", p, "開始日期")]:
        bad = df[col].isna() & df[col + "_raw"].notna()
        add(name, f"{col} 無法解析", bad.sum(), len(df), "ERROR", "該列會被排除",
            df.loc[bad, col + "_raw"].tolist())
    bad = r["期數"].isna()
    add("②定期紀錄", "交易編號 無法解析出期數（應為 X_期數）", bad.sum(), len(r), "WARN",
        "這些扣款無法判斷第幾期", r.loc[bad, "交易編號"].tolist())
    for name, df, col in [("②定期紀錄", r, "金額"), ("③單筆紀錄", o, "金額"), ("④定期明細", p, "約定金額")]:
        bad = df[col].isna() | (df[col] <= 0)
        add(name, f"{col} 非正數或空白", bad.sum(), len(df), "ERROR", "金額統計會失真",
            df.loc[bad, col + "_raw"].tolist())
        q1, q3 = df[col].quantile([.25, .75])
        hi = df[col] > q3 + 3 * (q3 - q1)
        add(name, f"{col} 極端值（> Q3 + 3×IQR）", hi.sum(), len(df), "INFO",
            "預設保留；總額會被少數大額拉高，報告同時看中位數", sorted(df.loc[hi, col].astype(int).tolist())[::-1])
    big = o["金額"] >= BIG_GIFT
    add("③單筆紀錄", f"單筆 ≥ {BIG_GIFT:,} 元（官方說明頁標註的大額）", big.sum(), len(o), "INFO",
        "會大幅拉高平均；任務一請個別看，不要混進一般單筆捐款人的輪廓",
        sorted(o.loc[big, "金額"].astype(int).tolist())[::-1])
    for name, df in [("②定期紀錄", r), ("③單筆紀錄", o)]:
        if "付款類型" in df:
            add(name, "付款類型 整欄空白", int(df["付款類型"].isna().all()), 1, "INFO", "真實匯出也是空的，忽略")

    # --- 缺漏（任務一）
    add("①聯絡人", "年齡 未填寫", (~c["年齡已填"]).sum(), len(c), "INFO",
        "25 歲以下比例一律以『已填年齡者』為分母，並同時報告填寫率")
    add("①聯絡人", "性別 空白", c["性別"].isna().sum(), len(c), "INFO", "多為沒有捐款紀錄的聯絡人")
    amb = c["鄉鎮市區"].isin(AMBIGUOUS_TOWNS)
    add("①聯絡人", "鄉鎮市區 為跨縣市同名（無法判斷縣市）", amb.sum(), len(c), "INFO",
        "沒有縣市欄；②③ 的住家-縣市約 96% 空白。地區分析只能到『可辨識的鄉鎮』", c.loc[amb, "鄉鎮市區"].unique().tolist())

    # --- 邏輯一致性
    first = pd.concat([r[["系統編號", "捐款日期"]], o[["系統編號", "捐款日期"]]]) \
        .groupby("系統編號")["捐款日期"].min()
    cj = c.set_index("系統編號").join(first.rename("_first"), how="inner")
    bad = cj["聯絡人建立日期"] > cj["_first"]
    add("①聯絡人", "聯絡人建立日期 晚於 首次捐款日期", bad.sum(), len(cj), "WARN",
        "這些人的新/舊判斷不可靠", cj.index[bad].tolist())
    s = r.sort_values(["定期捐款編號", "捐款日期"])
    mono = s.groupby("定期捐款編號")["期數"].apply(lambda v: v.is_monotonic_increasing)
    add("②定期紀錄", "期數 未隨日期遞增", (~mono).sum(), len(mono), "WARN",
        "『第幾期停止』會不準", mono.index[~mono].tolist())
    gs = r.groupby("定期捐款編號")["期數"]
    gap = (gs.max() - gs.min() + 1) != gs.size()
    add("②定期紀錄", "期數 有跳號（期間內漏掉某幾期）", gap.sum(), len(gap), "INFO", "", gap.index[gap].tolist())
    per_m = r.groupby(["定期捐款編號", r["捐款日期"].dt.to_period("M")]).size()
    add("②定期紀錄", "同一方案同月扣款超過 1 次", (per_m > 1).sum(), len(per_m), "INFO",
        "多為扣款日跨月邊界；留存以『當月是否有扣款』計", [f"{a}@{b}" for a, b in per_m[per_m > 1].index[:5]])
    pm = r.merge(p[["定期捐款編號", "約定金額"]], on="定期捐款編號")
    bad = pm["金額"] != pm["約定金額"]
    add("②定期紀錄", "扣款金額 ≠ ④ 約定金額", bad.sum(), len(pm), "INFO", "可能是調整金額",
        pm.loc[bad, "捐款編號"].tolist())

    # ④ 一人一列 → 方案還原
    in_r = set(r["定期捐款編號"])
    unlisted = in_r - set(p["定期捐款編號"])
    add("④定期明細", "② 有扣款、但 ④ 沒列出的方案（一人多方案只顯示一個）", len(unlisted), len(in_r), "WARN",
        "這些方案的狀態、失敗紀錄、約定期數看不到 → 由 ② 還原，狀態標為『推定』", sorted(unlisted)[:5])
    stale = p[~p["定期捐款編號"].isin(in_r) & p["系統編號"].isin(set(r["系統編號"]))]
    add("④定期明細", "④ 顯示的是舊方案，但此人期間內仍有其他方案在扣款", len(stale), len(p), "WARN",
        "④ 的『已失敗／待處理／已完成』多半是換卡或續約前的舊方案 → 不可直接列入挽回名單",
        (stale["系統編號"] + ":" + stale["狀態"]).tolist())
    smax = r.groupby("定期捐款編號")["期數"].max()
    pj = p.set_index("定期捐款編號").join(smax.rename("_smax"), how="inner")
    diff = pj["執行期數"] - pj["_smax"]
    add("④定期明細", "執行期數 ≠ ② 最大期數", (diff != 0).sum(), len(pj), "INFO",
        "④ 在 11/1 匯出，可能多出 ② 沒有的 11/1 那一期（差 1 屬正常）",
        [f"{i}:+{int(v)}" for i, v in diff[diff != 0].items()])
    big = diff.abs() > 1
    add("④定期明細", "執行期數 與 ② 最大期數 差超過 1", big.sum(), len(pj), "WARN", "期數口徑不一致",
        diff[big].index.tolist())
    pi = p.set_index("定期捐款編號")
    bad = pi["執行期數"] != pi["成功次數"]
    add("④定期明細", "執行期數 ≠ 成功次數", bad.sum(), len(pi), "INFO", "", pi.index[bad].tolist())
    bad = (pi["請款次數"] > pi["成功次數"]) != pi["最近失敗日期"].notna()
    add("④定期明細", "有失敗次數但無最近失敗日期（或相反）", bad.sum(), len(pi), "WARN",
        "失敗挽回名單可能漏人", pi.index[bad].tolist())
    done = pi["狀態"].eq(COMPLETED_STATUS) & (pi["約定總期數"] > 0)
    bad = done & (pi["執行期數"] < pi["約定總期數"])
    add("④定期明細", "狀態=已完成 但 執行期數 < 約定總期數", bad.sum(), done.sum(), "WARN",
        "已完成不等於扣滿", pi.index[bad].tolist())
    fin = pi["約定總期數"] > 0
    bad = fin & pi["剩餘期數"].notna() & (pi["約定總期數"] - pi["執行期數"] != pi["剩餘期數"])
    add("④定期明細", "剩餘期數 ≠ 約定總期數 − 執行期數", bad.sum(), fin.sum(), "INFO", "", pi.index[bad].tolist())
    if plans is not None:
        n_new = int(plans["期間新方案"].sum())
        rows.append({"資料表": "方案主檔", "檢查項目": "期間內新開的定期方案數", "異常筆數": n_new,
                     "總筆數": len(plans), "異常比例": round(n_new / len(plans), 4), "嚴重度": "INFO",
                     "對分析的影響": "含換卡／續約重開的方案；新定期捐款人另以『此人最早方案在期間內開始』判斷",
                     "範例": ""})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. 整合主檔
# ---------------------------------------------------------------------------
def build_donations(d: Data) -> pd.DataFrame:
    r = d.recurring.assign(捐款類型="定期定額")
    o = d.one_time.assign(捐款類型="單筆", 定期捐款編號=np.nan, 期數=np.nan)
    cols = ["捐款編號", "交易編號", "系統編號", "捐款日期", "金額", "付款工具", "捐款類型", "定期捐款編號", "期數"]
    tx = pd.concat([r[cols], o[cols]], ignore_index=True).dropna(subset=["捐款日期", "金額"])
    tx = tx.merge(d.contacts[["系統編號", "年齡", "性別", "鄉鎮市區"]], on="系統編號", how="left")
    tx["月份"] = tx["捐款日期"].dt.to_period("M").astype(str)
    first = tx.groupby("系統編號")["捐款日期"].transform("min")
    tx["Cohort月份"] = first.dt.to_period("M").astype(str)
    tx["距首捐月數"] = (tx["捐款日期"].dt.year - first.dt.year) * 12 + tx["捐款日期"].dt.month - first.dt.month
    return tx.sort_values(["系統編號", "捐款日期", "捐款編號"]).reset_index(drop=True)


def charge_gap_p90(tx: pd.DataFrame) -> int:
    """同一方案相鄰兩次扣款的間隔天數 P90（下限 SILENT_DAYS）：判斷『沒扣款＝停止』前至少要等這麼久。"""
    r = tx[tx["捐款類型"] == "定期定額"].sort_values(["定期捐款編號", "捐款日期"])
    gaps = r.groupby("定期捐款編號")["捐款日期"].diff().dt.days.dropna()
    return int(max(SILENT_DAYS, math.ceil(gaps.quantile(0.9)))) if len(gaps) else SILENT_DAYS


def _term_group(term: float, unlimited: bool) -> str:
    if unlimited:
        return "不限期"
    if pd.isna(term):
        return "未知（明細未列）"
    for hi, lab in [(6, "≤6期"), (12, "7-12期"), (24, "13-24期"), (36, "25-36期")]:
        if term <= hi:
            return lab
    return "37期以上"


def build_plans(d: Data, tx: pd.DataFrame, gap_days: int) -> pd.DataFrame:
    """方案主檔：② 的每個定期捐款編號 ∪ ④ 的每一列。④ 有列的用 ④ 狀態；沒列的由 ② 推定。"""
    r = tx[tx["捐款類型"] == "定期定額"]
    g = r.groupby("定期捐款編號")
    agg = pd.DataFrame({
        "系統編號_②": g["系統編號"].first(), "期間扣款筆數": g.size(),
        "期間首次扣款日": g["捐款日期"].min(), "期間最近扣款日": g["捐款日期"].max(),
        "期間最小期數": g["期數"].min(), "期間最大期數": g["期數"].max(),
        "期間扣款金額": g["金額"].sum(), "扣款金額": g["金額"].agg(_mode), "付款工具_②": g["付款工具"].agg(_mode),
    })
    det = d.plans.drop(columns=[c for c in d.plans.columns if c.endswith("_raw")]).set_index("定期捐款編號")
    p = pd.DataFrame(index=agg.index.union(det.index)).join(det).join(agg)
    p.index.name = "定期捐款編號"
    p["明細有列"] = p.index.isin(det.index)
    p["系統編號"] = p["系統編號"].fillna(p["系統編號_②"])
    p["約定金額"] = p["約定金額"].fillna(p["扣款金額"])
    p["不限期"] = p["不限期"].eq(True)
    p["執行期數"] = p["執行期數"].fillna(p["期間最大期數"])
    p["金流機制"] = p["金流機制"].fillna(p["付款工具_②"].replace({"LINE Pay": "LINE / Apple Pay"}))
    p["最近成功扣款日"] = p[["最近一筆收到日期", "期間最近扣款日"]].max(axis=1)
    est = p["期間首次扣款日"] - pd.to_timedelta((p["期間最小期數"] - 1) * 30.44, unit="D")
    p["開始日期為推估"] = p["開始日期"].isna() & est.notna()
    p["開始日期"] = p["開始日期"].fillna(est.dt.normalize())
    p["沉默天數"] = (d.as_of - p["最近成功扣款日"].dt.normalize()).dt.days.clip(lower=0)  # 以日曆天計
    p["沉默級距"] = pd.cut(p["沉默天數"], bins=[-1] + [b[1] for b in INACTIVITY_BUCKETS],
                       labels=[b[2] for b in INACTIVITY_BUCKETS])
    unl = ~p["明細有列"]
    p.loc[unl, "狀態"] = np.where(p.loc[unl, "沉默天數"] <= gap_days, "推定進行中（明細未列）", "推定停止（明細未列）")
    p["期間新方案"] = p["期間最小期數"].eq(1) | (p["開始日期"] >= d.period_start)
    p["約定期數分組"] = [_term_group(t, u) for t, u in zip(p["約定總期數"], p["不限期"])]
    p["有扣款失敗"] = (p["請款次數"] > p["成功次數"]).fillna(False)
    p["失敗後已恢復"] = p["有扣款失敗"] & (p["最近成功扣款日"] > p["最近失敗日期"])
    p["停止"] = p["狀態"].isin(STOPPED_STATUSES)
    stop_date = pd.Series(pd.NaT, index=p.index)
    stop_date = stop_date.fillna(p["失敗或取消日期"].where(p["狀態"].isin(["已失敗", "已取消"])))
    stop_date = stop_date.fillna(p["最新建立日期"].where(p["狀態"].eq("待處理")))
    stop_date = stop_date.fillna(p["最近成功扣款日"].where(p["狀態"].isin(["暫停中", "推定停止（明細未列）"])))
    stop_date = stop_date.fillna(p["結束日期"].where(p["狀態"].eq(COMPLETED_STATUS)))
    p["停止或結束日期"] = stop_date
    p["停止距今天數"] = (d.as_of - stop_date.dt.normalize()).dt.days.clip(lower=0)
    p["是否在扣款中"] = p["狀態"].isin(ACTIVE_STATUSES)

    # 同一人的其他方案：換卡＝新方案（官網規則），所以停止的方案若此人另有方案在扣，不算流失
    p = p.reset_index()
    act = p[p["是否在扣款中"]][["系統編號", "定期捐款編號", "開始日期", "約定金額"]]
    pair = p[["定期捐款編號", "系統編號", "開始日期", "約定金額"]].merge(act, on="系統編號", suffixes=("", "_他"))
    pair = pair[pair["定期捐款編號"] != pair["定期捐款編號_他"]]
    p["此人仍有其他方案在扣款"] = p["定期捐款編號"].isin(pair["定期捐款編號"])
    later = pair[pair["開始日期_他"] >= pair["開始日期"]]
    p["之後有加碼方案"] = p["定期捐款編號"].isin(later.loc[later["約定金額_他"] > later["約定金額"], "定期捐款編號"])
    p["方案數(此人)"] = p.groupby("系統編號")["定期捐款編號"].transform("size")
    p["換方案(非流失)"] = (p["停止"] | p["狀態"].eq(COMPLETED_STATUS)) & p["此人仍有其他方案在扣款"]
    # ④ 的多方案可能是續約／換卡，也可能是「同時兩個方案」停掉一個 → 人還在，但每月金額變少
    act_sum = p[p["是否在扣款中"]].groupby("系統編號")["約定金額"].sum()
    other_sum = p["系統編號"].map(act_sum).fillna(0) - np.where(p["是否在扣款中"], p["約定金額"], 0)
    p["換方案後月額變化"] = np.where(p["換方案(非流失)"], other_sum - p["約定金額"], np.nan)
    p["真流失"] = p["停止"] & ~p["此人仍有其他方案在扣款"]
    p["流失類型"] = np.where(p["真流失"], p["狀態"].map(CHURN_TYPE), "")
    p["被動流失"] = p["真流失"] & p["狀態"].isin(["待處理", "已失敗"])
    p["靜默流失"] = p["狀態"].eq("進行中") & (p["沉默天數"] > SILENT_DAYS)
    first_plan = p.groupby("系統編號")["開始日期"].transform("min")
    p["新定期捐款人的方案"] = first_plan >= d.period_start
    p = p.merge(d.contacts[["系統編號", "年齡", "性別", "鄉鎮市區"]], on="系統編號", how="left")
    drop = ["系統編號_②", "扣款金額", "付款工具_②"]
    return p.drop(columns=drop)


def build_donors(d: Data, tx: pd.DataFrame, plans: pd.DataFrame) -> pd.DataFrame:
    c = d.contacts[["系統編號", "年齡", "年齡已填", "性別", "鄉鎮市區", "聯絡人建立日期"]].copy()
    g = tx.groupby("系統編號")
    first_row = tx.drop_duplicates("系統編號", keep="first").set_index("系統編號")
    n_rec = g["捐款類型"].apply(lambda s: (s == "定期定額").sum())
    n_one = g["捐款類型"].apply(lambda s: (s == "單筆").sum())
    agg = pd.DataFrame({
        "首捐日期": g["捐款日期"].min(), "最近捐款日期": g["捐款日期"].max(),
        "捐款筆數": g.size(), "總捐款金額": g["金額"].sum(),
        "定期扣款筆數": n_rec, "單筆筆數": n_one,
        "首捐類型": first_row["捐款類型"], "首捐付款工具": first_row["付款工具"],
    })
    pp = plans.groupby("系統編號").agg(定期方案數=("定期捐款編號", "size"),
                                       最早方案開始日期=("開始日期", "min"),
                                       定期扣款中=("是否在扣款中", "any"))
    out = c.merge(agg, left_on="系統編號", right_index=True, how="left") \
           .merge(pp, left_on="系統編號", right_index=True, how="left")
    out["期間捐款人"] = out["首捐日期"].notna()
    out["定期方案數"] = out["定期方案數"].fillna(0).astype(int)
    out["定期扣款中"] = out["定期扣款中"].eq(True)
    out["捐款型態"] = np.select(
        [out["定期扣款筆數"].fillna(0).gt(0) & out["單筆筆數"].fillna(0).gt(0),
         out["定期扣款筆數"].fillna(0).gt(0), out["單筆筆數"].fillna(0).gt(0)],
        ["定期＋單筆", "只有定期", "只有單筆"], "期間無捐款")
    out["年輕支持者(<25)"] = out["年齡"].isin(YOUNG_AGE_BANDS)
    out["距今天數"] = (d.as_of - out["最近捐款日期"]).dt.days
    # 期間新捐款人：期間內有捐款，且聯絡人是在期間內才建立的（官網多為捐款當下建立）
    out["期間新捐款人"] = out["期間捐款人"] & (out["聯絡人建立日期"] >= d.period_start)
    out["舊聯絡人期間捐款"] = out["期間捐款人"] & ~out["期間新捐款人"]
    return out


# ---------------------------------------------------------------------------
# 4. 任務一：看懂誰在支持（輕量版，主要給分群溝通用）
# ---------------------------------------------------------------------------
def t1_profile(donors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = {"全部聯絡人": donors, "期間捐款人": donors[donors["期間捐款人"]]}
    for k in ["只有單筆", "只有定期", "定期＋單筆"]:
        groups[k] = donors[donors["捐款型態"] == k]
    for name, g in groups.items():
        filled = g["年齡已填"]
        young = g["年輕支持者(<25)"] & filled
        lo, hi = wilson_ci(int(young.sum()), int(filled.sum()))
        rows.append({"群組": name, "人數": len(g), "年齡填寫率": round(filled.mean(), 4) if len(g) else np.nan,
                     "25歲以下(占已填)": round(young.sum() / filled.sum(), 4) if filled.sum() else np.nan,
                     "95%下界": round(lo, 4), "95%上界": round(hi, 4),
                     "生理女占比(占已填)": round((g["性別"] == "生理女").sum() / g["性別"].notna().sum(), 4)
                     if g["性別"].notna().sum() else np.nan,
                     "期間捐款中位數": g["總捐款金額"].median()})
    df = pd.DataFrame(rows)
    sub = donors[donors["期間捐款人"] & donors["年齡已填"]]
    sub = sub[sub["捐款型態"] != "期間無捐款"]
    stat, dof, pval = chi2_independence(pd.crosstab(sub["捐款型態"], sub["年輕支持者(<25)"]))
    df.attrs["test"] = {"檢定": "捐款型態（單筆/定期/兩者）與 是否 25 歲以下 是否相關（只看已填年齡）",
                        "p值": round(pval, 4), "判定": verdict(pval, int(sub["年輕支持者(<25)"].sum()))}
    return df


# ---------------------------------------------------------------------------
# 5. 任務二：留住定期定額捐款人
# ---------------------------------------------------------------------------
def hazard_base(plans: pd.DataFrame) -> pd.DataFrame:
    """第幾期停止的觀察對象：期間內有扣款的方案（從期間第一期開始觀察＝延遲進入）＋期間新開的方案（從第 0 期開始）。"""
    b = plans[(plans["期間扣款筆數"].fillna(0) > 0) | plans["期間新方案"]].copy()
    b["進入觀察期數"] = np.where(b["期間新方案"], 0, b["期間最小期數"].fillna(0))
    return b


def t2_stop_period(plans: pd.DataFrame) -> pd.DataFrame:
    """Kaplan-Meier（延遲進入）：第 k 期『真流失』的條件機率。第 0 期＝第一次扣款就沒成功。"""
    b = hazard_base(plans)
    rows, surv = [], 1.0
    for k in range(0, int(b["執行期數"].max()) + 1):
        at_risk = ((b["進入觀察期數"] <= k) & (b["執行期數"] >= k)).sum()
        at_k = b["執行期數"] == k
        stopped = (at_k & b["真流失"]).sum()
        h = stopped / at_risk if at_risk else 0.0
        surv *= 1 - h
        lo, hi = wilson_ci(int(stopped), int(at_risk))
        rows.append({"期數": k, "仍在觀察中的方案數": int(at_risk), "於此期流失": int(stopped),
                     "其中被動(扣款失敗)": int((at_k & b["被動流失"]).sum()),
                     "於此期已完成": int((at_k & b["狀態"].eq(COMPLETED_STATUS)).sum()),
                     "於此期換方案": int((at_k & b["換方案(非流失)"]).sum()),
                     "停止率(hazard)": round(h, 4), "停止率95%下界": round(lo, 4), "停止率95%上界": round(hi, 4),
                     "累積存活率": round(surv, 4)})
    df = pd.DataFrame(rows)
    df = df[df["仍在觀察中的方案數"] > 0].reset_index(drop=True)
    early = df[df["期數"] <= 24]
    tab = pd.DataFrame({"流失": early["於此期流失"], "未流失": early["仍在觀察中的方案數"] - early["於此期流失"]})
    stat, dof, pval = chi2_independence(tab)
    df.attrs["test"] = {"檢定": "第 0–24 期的停止率是否相同（卡方；若不同＝有特定危險期數）",
                        "p值": round(pval, 4), "判定": verdict(pval, int(tab["流失"].sum()))}
    return df


def t2_term_vs_actual(plans: pd.DataFrame) -> pd.DataFrame:
    b = hazard_base(plans)
    b = b[b["明細有列"]]
    g = b.groupby("約定期數分組")
    df = pd.crosstab(b["約定期數分組"], b["狀態"])
    df["方案數"] = g.size()
    df["平均執行期數"] = g["執行期數"].mean().round(2)
    df["真流失率"] = g["真流失"].mean().round(4)
    full = (b["約定總期數"] > 0) & (b["執行期數"] >= b["約定總期數"])
    early = b["狀態"].eq(COMPLETED_STATUS) & (b["約定總期數"] > 0) & (b["執行期數"] < b["約定總期數"])
    df["真正扣滿約定期數"] = full.groupby(b["約定期數分組"]).sum()
    df["已完成但未扣滿"] = early.groupby(b["約定期數分組"]).sum()
    done = b["狀態"].eq(COMPLETED_STATUS)
    df["已完成→另有方案在扣"] = (done & b["此人仍有其他方案在扣款"]).groupby(b["約定期數分組"]).sum()
    df["已完成→沒有續約"] = (done & ~b["此人仍有其他方案在扣款"]).groupby(b["約定期數分組"]).sum()
    df["之後有加碼方案"] = g["之後有加碼方案"].sum()
    stat, dof, pval = chi2_independence(pd.crosstab(b["約定期數分組"], b["真流失"]))
    df.attrs["test"] = {"檢定": "約定期數（不限/6/12/…）與 真流失 是否相關（卡方）", "p值": round(pval, 4),
                        "判定": verdict(pval, int(b["真流失"].sum()))}
    return df.reset_index()


def t2_failure_recovery(plans: pd.DataFrame) -> pd.DataFrame:
    p = plans[plans["明細有列"]]
    f = p[p["有扣款失敗"]]
    g = f.groupby("金流機制")
    df = pd.DataFrame({"有失敗方案數": g.size(), "已恢復(同方案)": g["失敗後已恢復"].sum(),
                       "未恢復但另開方案在扣": (g["此人仍有其他方案在扣款"].sum())})
    df["恢復率(同方案)"] = (df["已恢復(同方案)"] / df["有失敗方案數"]).round(4)
    df["失敗發生率"] = p.groupby("金流機制")["有扣款失敗"].mean().round(4)
    stat, dof, pval = chi2_independence(pd.crosstab(f["金流機制"], f["失敗後已恢復"]))
    df.attrs["test"] = {"檢定": "金流機制 與 失敗後是否恢復（卡方）", "p值": round(pval, 4),
                        "判定": verdict(pval, int(g.size().min()) if len(g) else 0)}
    return df.reset_index()


def t2_churn_types(plans: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """期間內停下來的方案：流失類型 × 停止時間，並把『換方案』分開，避免誤當流失。"""
    b = hazard_base(plans)
    b = b[b["停止"] | b["狀態"].eq(COMPLETED_STATUS)]
    switch = np.where(b["換方案後月額變化"] < 0, "換方案但每月金額變少（此人仍在扣款）", "換方案／續約（此人仍在扣款，月額持平或增加）")
    b = b.assign(分類=np.where(b["換方案(非流失)"], switch,
                              np.where(b["真流失"], b["流失類型"], "已完成且沒有續約")))
    df = b.groupby("分類").agg(方案數=("定期捐款編號", "size"), 中位數執行期數=("執行期數", "median"),
                              中位數約定金額=("約定金額", "median"),
                              停止未滿30天=("停止距今天數", lambda s: int((s <= 30).sum())),
                              停止31至90天=("停止距今天數", lambda s: int(s.between(31, 90).sum())),
                              停止超過90天=("停止距今天數", lambda s: int((s > 90).sum())))
    return df.sort_values("方案數", ascending=False).reset_index()


def t2_inactivity(plans: pd.DataFrame) -> pd.DataFrame:
    p = plans[plans["期間扣款筆數"].fillna(0) > 0]
    df = pd.crosstab(p["沉默級距"], p["狀態"], margins=True, margins_name="合計")
    return df.reset_index().rename(columns={"沉默級距": "距最近一次扣款"})


def t2_monthly_churn(tx: pd.DataFrame, as_of: pd.Timestamp, min_wait: int) -> pd.DataFrame:
    """以『人』計：當月有任何定期扣款＝活躍；最後一次定期扣款在 m 月、之後都沒有＝ m 月流失（換卡不會被誤算）。"""
    r = tx[tx["捐款類型"] == "定期定額"]
    first_m = r.groupby("系統編號")["捐款日期"].min().dt.to_period("M")
    last_m = r.groupby("系統編號")["捐款日期"].max().dt.to_period("M")
    rows = []
    charged = r.groupby(r["捐款日期"].dt.to_period("M"))["系統編號"].nunique()
    for m in pd.period_range(first_m.min(), as_of.to_period("M")):
        active = ((first_m <= m) & (last_m >= m)).sum()
        churned = (last_m == m).sum()
        new = (first_m == m).sum()
        enough = (as_of - m.end_time).days >= min_wait
        # 當月成功扣款人數＝官方資料說明頁的「每月成功扣款人數」；在籍人數另含當月剛好漏扣、之後又扣到的人
        rows.append({"月份": str(m), "當月成功扣款人數": int(charged.get(m, 0)), "在籍人數(首末扣款之間)": int(active),
                     "當月首次出現": int(new),
                     "之後不再扣款": int(churned) if enough else None,
                     "月流失率": round(churned / active, 4) if (active and enough) else None,
                     "備註": "" if enough else f"距截止不足 {min_wait} 天（扣款間隔P90），尚無法判定"})
    return pd.DataFrame(rows)


def cohort_retention(tx: pd.DataFrame, donors: pd.DataFrame, kind: str) -> pd.DataFrame:
    """kind='recurring'：期間內新開始定期的人，以首次扣款月份分群，看之後每月是否仍有定期扣款
       kind='new'      ：期間新捐款人，以首捐月份分群，看之後每月是否有任何捐款"""
    if kind == "recurring":
        ids = set(donors.loc[donors["最早方案開始日期"] >= tx["捐款日期"].min().normalize(), "系統編號"])
        t = tx[(tx["捐款類型"] == "定期定額") & tx["系統編號"].isin(ids)].copy()
    else:
        ids = set(donors.loc[donors["期間新捐款人"], "系統編號"])
        t = tx[tx["系統編號"].isin(ids)].copy()
    t["c"] = t.groupby("系統編號")["捐款日期"].transform("min").dt.to_period("M")
    t["m"] = t["捐款日期"].dt.to_period("M")
    t["age"] = (t["m"] - t["c"]).apply(lambda x: x.n)
    size = t.groupby("c")["系統編號"].nunique()
    act = t.groupby(["c", "age"])["系統編號"].nunique().unstack(fill_value=0)
    rate = act.div(size, axis=0).round(4)
    last = t["m"].max()
    for cm in rate.index:
        rate.loc[cm, rate.columns > (last - cm).n] = np.nan   # 觀察不到 ≠ 0
    rate.insert(0, "Cohort人數", size)
    rate.index = rate.index.astype(str)
    rate.columns = ["Cohort人數"] + [f"M{a}" for a in rate.columns[1:]]
    return rate.reset_index().rename(columns={"c": "Cohort月份"})


SEGMENT_COLS = ["年齡", "性別", "約定金額", "約定期數分組", "金流機制"]


def segment_compare(df: pd.DataFrame, flag: str, cols=SEGMENT_COLS, label="") -> pd.DataFrame:
    out = []
    cols = [c for c in cols if c in df]
    for col in cols:
        g = df.groupby(col)[flag]
        stat, dof, pval = chi2_independence(pd.crosstab(df[col], df[flag]))
        p_adj = min(1.0, pval * len(cols))  # Bonferroni
        v = verdict(p_adj, int(g.size().min()))
        for val, s in g:
            lo, hi = wilson_ci(int(s.sum()), len(s))
            out.append({"指標": label or flag, "分群維度": col, "群組": str(val), "人數/方案數": len(s),
                        "比率": round(s.mean(), 4), "95%下界": round(lo, 4), "95%上界": round(hi, 4),
                        "整體比率": round(df[flag].mean(), 4), "維度p值": round(pval, 4),
                        "校正後p值": round(p_adj, 4), "判定": v})
    return pd.DataFrame(out)


def t2_reminder_list(plans: pd.DataFrame) -> pd.DataFrame:
    """優先提醒名單（一人一列）：規則透明、每一分都有理由。此人另有方案在扣款的停止方案，不列入。"""
    p = plans.reset_index(drop=True).copy()
    score = pd.Series(0.0, index=p.index)
    reasons = [[] for _ in range(len(p))]
    action = pd.Series("", index=p.index)
    other = p["此人仍有其他方案在扣款"]

    def rule(mask, pts, reason, act=None):
        nonlocal score
        mask = mask.fillna(False).to_numpy(dtype=bool)
        score += np.where(mask, pts, 0)
        for i in np.flatnonzero(mask):
            reasons[i].append(reason)
        if act:
            empty = (action == "").to_numpy() & mask
            action[empty] = act

    rule(p["狀態"].isin(["已失敗", "待處理"]) & ~other & (p["停止距今天數"] <= 365), 40,
         "扣款失敗後沒有任何方案在扣（被動流失，可挽回）", "付款失敗挽回：說明如何用新卡重新設定")
    rule(p["狀態"].eq("進行中") & p["有扣款失敗"] & ~p["失敗後已恢復"], 35,
         "進行中但最近一次扣款失敗、尚未恢復", "付款失敗挽回：說明如何用新卡重新設定")
    rule(p["狀態"].eq("暫停中") & ~other & (p["停止距今天數"] <= 365), 30, "方案暫停中", "恢復邀請：分享近期救援成果")
    rule(p["靜默流失"], 25, f"進行中但超過 {SILENT_DAYS} 天無扣款", "靜默關懷：確認是否仍願意支持")
    rule(p["狀態"].eq("推定停止（明細未列）") & ~other & (p["停止距今天數"] <= 365), 20,
         "明細未列、② 已久未扣款", "先到後台確認方案狀態")
    rule(p["是否在扣款中"] & p["剩餘期數"].between(0, 2), 15, "約定期數剩 0–2 期", "續約邀請：成果回顧＋續約連結")
    rule(p["狀態"].eq(COMPLETED_STATUS) & ~other & (p["停止距今天數"] <= 365), 20,
         "期滿結束且沒有續約", "續約邀請：成果回顧＋續約連結")
    rule(p["狀態"].eq("已取消") & ~other & (p["停止距今天數"] <= 365), 5, "主動取消", "尊重取消：只寄年度成果，不催")
    rule(p["停止"] & ~other & (p["停止距今天數"] <= 30), 20, "停止未滿 30 天（最容易挽回）")
    rule(p["停止"] & ~other & p["停止距今天數"].between(31, 90), 10, "停止 31–90 天")
    score += (p["約定金額"] / 100).clip(upper=15).fillna(0)
    rule(p["執行期數"] >= 6, 5, "已穩定扣款 6 期以上")
    p["優先分數"] = score.round(1)
    p["理由"] = ["；".join(x) for x in reasons]
    p["建議行動"] = action
    p = p[p["建議行動"] != ""]
    p = p.sort_values("優先分數", ascending=False).drop_duplicates("系統編號", keep="first")
    cols = ["系統編號", "定期捐款編號", "優先分數", "建議行動", "理由", "狀態", "約定金額", "執行期數",
            "約定總期數", "剩餘期數", "停止或結束日期", "停止距今天數", "沉默天數", "最近失敗日期", "金流機制",
            "方案數(此人)", "明細有列", "年齡", "性別", "鄉鎮市區"]
    return p[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. 任務三：找出讓人開始支持的時刻
# ---------------------------------------------------------------------------
def _daily(dates: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    return dates.dt.normalize().value_counts().reindex(days, fill_value=0)


def t3_event_lift(d: Data, donors: pd.DataFrame, plans: pd.DataFrame) -> pd.DataFrame:
    """每個活動窗（活動日～結束日，預設 7 天）的新捐款人／新定期方案，是否顯著多於活動前 14 天的日均。"""
    days = pd.date_range(d.period_start, d.as_of)
    new_plan_date = plans["期間首次扣款日"].where(plans["期間最小期數"].eq(1), plans["開始日期"])
    series = {"新捐款人": _daily(donors.loc[donors["期間新捐款人"], "首捐日期"], days),
              "新定期方案": _daily(new_plan_date[plans["期間新方案"]], days)}
    rows = []
    ev = d.events
    for _, e in ev.iterrows():
        t0, t1 = e["日期"], min(e["結束日期"], d.as_of)
        for outcome, daily in series.items():
            pre = daily[t0 - pd.Timedelta(days=EVENT_PRE_DAYS): t0 - pd.Timedelta(days=1)]
            post = daily[t0:t1]
            pre_rate = pre.mean() if len(pre) else np.nan
            exp = pre_rate * len(post) if len(pre) else np.nan
            obs = int(post.sum())
            pval = poisson_upper_p(obs, exp) if len(pre) >= 7 else np.nan
            rows.append({"事件編號": e["事件編號"], "日期": t0.date().isoformat(), "結束日期": t1.date().isoformat(),
                         "事件類型": e["事件類型"], "事件名稱": e["事件名稱"], "指標": outcome,
                         "活動前日均": round(pre_rate, 2), "活動窗天數": len(post), "活動窗人數": obs,
                         "預期人數": round(exp, 1), "增幅倍數": round(obs / exp, 2) if exp else np.nan,
                         "p值": round(pval, 4) if pval == pval else np.nan})
    df = pd.DataFrame(rows)
    n_tests = df.groupby("指標")["p值"].transform(lambda s: s.notna().sum())
    df["校正後p值"] = (df["p值"] * n_tests).clip(upper=1).round(4)
    df["判定"] = np.where(df["p值"].isna(), "基準期不足：不宜下結論",
                        np.where(df["校正後p值"] < ALPHA, "顯著增加",
                                 np.where(df["p值"] < ALPHA, "未校正才顯著：不宜下結論", "無顯著增加：不宜歸因於此活動")))
    return df


def t3_unexplained_spikes(d: Data, donors: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    """單日高峰（新捐款人或單筆筆數遠高於前 28 天中位數），並標出時間軸上有沒有對應活動。"""
    days = pd.date_range(d.period_start, d.as_of)
    series = {"新捐款人": _daily(donors.loc[donors["期間新捐款人"], "首捐日期"], days),
              "單筆筆數": _daily(tx.loc[tx["捐款類型"] == "單筆", "捐款日期"], days)}
    rows = []
    for name, s in series.items():
        base = s.shift(1).rolling(28, min_periods=7).median()
        for day, obs in s.items():
            b = base.get(day)
            if pd.isna(b):
                continue
            exp = max(float(b), 0.5)
            p = poisson_upper_p(int(obs), exp)
            if p < SPIKE_P and obs >= 2 * exp and obs >= 10:
                near = d.events[(d.events["日期"] - pd.Timedelta(days=1) <= day) &
                                (d.events["結束日期"] + pd.Timedelta(days=2) >= day)]
                rows.append({"日期": day.date().isoformat(), "星期": "一二三四五六日"[day.dayofweek], "指標": name,
                             "當日數": int(obs), "前28天中位數": round(float(b), 1), "倍數": round(obs / exp, 1),
                             "p值": round(p, 6),
                             "時間軸對應活動": "；".join(near["事件名稱"]) if len(near) else "（找不到）"})
    return pd.DataFrame(rows, columns=["日期", "星期", "指標", "當日數", "前28天中位數", "倍數", "p值", "時間軸對應活動"])


def t3_new_donor_followup(d: Data, donors: pd.DataFrame, tx: pd.DataFrame):
    """單筆首捐的新捐款人：之後是否再捐？是否轉定期？依首捐時的活動、付款工具、年齡分組。"""
    new = donors[donors["期間新捐款人"]].copy()
    t = tx[tx["系統編號"].isin(new["系統編號"])]
    t = t.join(new.set_index("系統編號")["首捐日期"].rename("_f"), on="系統編號")
    later = t[t["捐款日期"] > t["_f"]]
    for w in (30, 90):
        ids = later.loc[later["捐款日期"] <= later["_f"] + pd.Timedelta(days=w), "系統編號"].unique()
        new[f"{w}天內再捐"] = new["系統編號"].isin(ids)
        new[f"可觀察滿{w}天"] = new["首捐日期"] <= d.as_of - pd.Timedelta(days=w)
    new["之後轉定期"] = new["首捐類型"].eq("單筆") & new["系統編號"].isin(
        later.loc[later["捐款類型"] == "定期定額", "系統編號"])
    lab = pd.Series("無對應活動", index=new.index)
    for _, e in d.events.sort_values("日期", ascending=False).iterrows():
        hit = new["首捐日期"].dt.normalize().between(e["日期"], e["結束日期"])
        lab[hit & lab.eq("無對應活動")] = e["事件類型"]
    new["首捐時的活動類型"] = lab
    new["年齡(含未填)"] = new["年齡"].where(new["年齡已填"], "未填寫")
    one = new[new["首捐類型"] == "單筆"]
    out = []
    for w in (30, 90):
        sub = one[one[f"可觀察滿{w}天"]]
        out.append(segment_compare(sub, f"{w}天內再捐", cols=["首捐時的活動類型", "首捐付款工具"],
                                   label=f"{w}天內再捐率（單筆首捐）"))
    out.append(segment_compare(one, "之後轉定期", cols=["首捐時的活動類型", "首捐付款工具"],
                               label="轉為定期率（單筆首捐）"))
    return pd.concat(out, ignore_index=True), new


def t3_weekly_new(donors: pd.DataFrame) -> pd.DataFrame:
    new = donors[donors["期間新捐款人"]]
    w = new["首捐日期"].dt.to_period("W-SUN").dt.start_time
    by = pd.crosstab(w, new["首捐類型"])
    by["新捐款人數"] = by.sum(axis=1)
    return by.rename_axis("週起始日").reset_index()


# ---------------------------------------------------------------------------
# 7. 匯總 + 報告
# ---------------------------------------------------------------------------
def overview(d: Data, donors: pd.DataFrame, plans: pd.DataFrame, tx: pd.DataFrame) -> dict:
    dn = donors[donors["期間捐款人"]]
    filled_c, filled_d = donors["年齡已填"], dn["年齡已填"]
    young = lambda g: round(float((g["年輕支持者(<25)"] & g["年齡已填"]).sum() / g["年齡已填"].sum()), 4)
    hb = hazard_base(plans)
    return {
        "資料期間": f"{d.period_start.date()} ~ {d.as_of.date()}",
        "聯絡人數": int(len(donors)), "期間捐款人數": int(len(dn)),
        "期間捐款率": round(len(dn) / len(donors), 4),
        "捐款總額": int(tx["金額"].sum()), "定期扣款總額": int(tx.loc[tx["捐款類型"] == "定期定額", "金額"].sum()),
        "單筆總額": int(tx.loc[tx["捐款類型"] == "單筆", "金額"].sum()),
        "單筆金額中位數": float(tx.loc[tx["捐款類型"] == "單筆", "金額"].median()),
        "交易筆數": int(len(tx)),
        "定期方案數(還原後)": int(len(plans)), "其中④有列": int(plans["明細有列"].sum()),
        "定期捐款人數": int(plans["系統編號"].nunique()),
        "一人多方案的人數": int((plans.groupby("系統編號").size() > 1).sum()),
        "目前扣款中人數": int(donors["定期扣款中"].sum()),
        "期間觀察方案數": int(len(hb)),
        "期間真流失方案數": int(hb["真流失"].sum()),
        "其中被動流失(扣款失敗)": int(hb["被動流失"].sum()),
        "換方案非流失數": int(hb["換方案(非流失)"].sum()),
        "④顯示失敗或待處理但此人仍在扣款": int((plans["明細有列"] & plans["狀態"].isin(["已失敗", "待處理"])
                                   & plans["此人仍有其他方案在扣款"]).sum()),
        "方案狀態分布": plans["狀態"].value_counts().to_dict(),
        "扣款失敗方案數": int((plans["明細有列"] & plans["有扣款失敗"]).sum()),
        "失敗後已恢復方案數": int((plans["明細有列"] & plans["失敗後已恢復"]).sum()),
        "靜默流失方案數": int(plans["靜默流失"].sum()),
        "期間新捐款人數": int(donors["期間新捐款人"].sum()),
        "期間新開定期方案數": int(plans["期間新方案"].sum()),
        "年齡填寫率(聯絡人)": round(float(filled_c.mean()), 4), "年齡填寫率(捐款人)": round(float(filled_d.mean()), 4),
        "25歲以下占聯絡人(已填)": young(donors), "25歲以下占捐款人(已填)": young(dn),
    }


def run(data: Path, out_dir: Path, as_of: str | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = prepare_data(data, out_dir)
    d = load(data_dir, as_of)
    tx = build_donations(d)
    gap90 = charge_gap_p90(tx)
    plans = build_plans(d, tx, gap90)
    donors = build_donors(d, tx, plans)
    checks = validate(d, plans)
    if (checks["嚴重度"] == "ERROR").any():
        print("⚠ 有 ERROR 等級的資料問題，結果可能失真，請先看 00_資料品質檢查.csv")

    res = {}
    res["t1_支持者輪廓"] = t1_profile(donors)
    res["t2_停止期數"] = t2_stop_period(plans)
    res["t2_流失類型"] = t2_churn_types(plans, d.as_of)
    res["t2_設定期數vs實際"] = t2_term_vs_actual(plans)
    res["t2_扣款失敗與恢復"] = t2_failure_recovery(plans)
    res["t2_沉默天數分布"] = t2_inactivity(plans)
    res["t2_月流失率"] = t2_monthly_churn(tx, d.as_of, gap90)
    res["t2_定期Cohort留存"] = cohort_retention(tx, donors, "recurring")
    hb = hazard_base(plans)
    res["t2_分群比較_流失"] = segment_compare(hb[hb["明細有列"]], "真流失", label="真流失率")
    res["t2_優先提醒名單"] = t2_reminder_list(plans)
    res["t3_活動前後"] = t3_event_lift(d, donors, plans)
    res["t3_未解釋高峰"] = t3_unexplained_spikes(d, donors, tx)
    res["t3_新捐款人後續行為"], new_detail = t3_new_donor_followup(d, donors, tx)
    res["t3_每週新捐款人"] = t3_weekly_new(donors)
    res["t3_新捐款人Cohort留存"] = cohort_retention(tx, donors, "new")

    def save(df, name):
        df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    save(checks, "00_資料品質檢查")
    save(tx, "01_捐款明細_整合")
    save(donors, "02_捐款人主檔")
    save(plans, "03_定期方案主檔")
    save(new_detail, "04_新捐款人明細")
    save(d.events, "05_活動時間軸_整理")
    for k, v in res.items():
        save(v, k)

    tests = {k: v.attrs["test"] for k, v in res.items() if "test" in v.attrs}
    seg_verdicts = {}
    for k in ["t2_分群比較_流失", "t3_新捐款人後續行為"]:
        for (ind, dim), s in res[k].groupby(["指標", "分群維度"]):
            seg_verdicts[f"{ind}｜{dim}"] = {"p值": float(s["維度p值"].iat[0]),
                                             "校正後p值": float(s["校正後p值"].iat[0]), "判定": s["判定"].iat[0]}
    ev = res["t3_活動前後"]
    rl = res["t2_優先提醒名單"]
    summary = {
        "總覽": overview(d, donors, plans, tx),
        "資料品質": checks[checks["嚴重度"] != "OK"][["資料表", "檢查項目", "異常筆數", "異常比例", "嚴重度", "對分析的影響"]]
        .to_dict("records"),
        "檢定結果": tests,
        "分群檢定": seg_verdicts,
        "活動判定": ev[["事件編號", "日期", "事件名稱", "指標", "增幅倍數", "p值", "校正後p值", "判定"]].to_dict("records"),
        "未解釋高峰": res["t3_未解釋高峰"].to_dict("records"),
        "優先提醒名單": {"人數": int(len(rl)), "依建議行動": rl["建議行動"].value_counts().to_dict()},
        "參數": {"停止狀態": sorted(STOPPED_STATUSES), "靜默流失天數": SILENT_DAYS,
               "活動前基準天數": EVENT_PRE_DAYS, "活動後預設觀察天數": EVENT_POST_DAYS,
               "顯著水準": ALPHA, "扣款間隔P90天數": gap90, "高峰門檻p": SPIKE_P,
               "多重比較校正": "Bonferroni（同一指標內的分群維度數／活動數）"},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                          encoding="utf-8")
    import importlib.util
    spec = importlib.util.spec_from_file_location("report", Path(__file__).with_name("report.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.build_report(out_dir, summary, checks, res)
    return summary


def main():
    ap = argparse.ArgumentParser(description="野灣捐款資料 Cohort / 留存 / 支持時刻 分析")
    ap.add_argument("--data", "--data-dir", dest="data", type=Path, default=Path("."),
                    help="官方 xlsx 檔，或含 5 個 CSV 的資料夾")
    ap.add_argument("--output-dir", type=Path, default=Path("yewan_output"))
    ap.add_argument("--as-of", default=None, help="觀察截止日（預設為資料中最後一筆捐款日）")
    a = ap.parse_args()
    s = run(a.data, a.output_dir, a.as_of)
    o = s["總覽"]
    print(f"資料期間 {o['資料期間']}｜捐款人 {o['期間捐款人數']:,}／聯絡人 {o['聯絡人數']:,}｜"
          f"定期方案 {o['定期方案數(還原後)']:,}（④ 有列 {o['其中④有列']:,}）")
    print(f"優先提醒名單 {s['優先提醒名單']['人數']:,} 人；輸出：{a.output_dir.resolve()}（先開 report.html）")


if __name__ == "__main__":
    main()
