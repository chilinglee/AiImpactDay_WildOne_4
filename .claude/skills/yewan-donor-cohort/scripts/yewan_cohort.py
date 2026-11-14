"""
野灣捐款資料 Cohort / 留存 / 支持時刻 分析管線

用法：
    python yewan_cohort.py --data-dir <CSV 資料夾> --output-dir <輸出資料夾> [--as-of 2026-10-31]

輸出：
    00_資料品質檢查.csv          每項檢查的嚴重度、筆數、比例、說明
    01_捐款明細_整合.csv          兩張捐款表 + 聯絡人屬性 + Cohort 欄位
    02_捐款人主檔.csv             每位聯絡人一列（RFM、首捐類型、是否年輕支持者…）
    03_定期方案主檔.csv           每個定期方案一列（停扣期數、失敗/恢復、沉默天數…）
    t2_*.csv                      任務二：留住定期定額捐款人
    t3_*.csv                      任務三：找出讓人開始支持的時刻
    summary.json                  關鍵數字 + 每個發現的「可信度判定」（給 AI 撰寫敘述用）
    report.html                   互動報告（可排序 / 篩選表格、圖表、可信度標記）

設計原則：
    1. 先驗證、再分析：每個發現都附上樣本數與統計檢定，未達顯著就標示「不宜下結論」。
    2. 定義寫死在程式裡、也寫在 references/metrics.md：同一個數字永遠同一個算法。
    3. 只依賴 pandas / numpy（Colab、Claude 程式執行環境都有）。
"""
from __future__ import annotations

import argparse
import json
import math
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

# 若當天拿到的欄位名稱不同，在這裡加別名即可（別名 -> 標準名）
COLUMN_ALIASES = {
    "SystemID": "系統編號", "系統編碼": "系統編號",
    "Date": "捐款日期", "Amount": "金額",
}

YOUNG_AGE_BANDS = {"20歲以下", "20-24歲"}          # 任務一：25 歲以下
STOPPED_STATUSES = {"暫停中", "已失敗"}             # 視為「停止」的事件
COMPLETED_STATUS = "已完成"
SILENT_DAYS = 45                                   # 進行中但超過幾天沒扣款 = 靜默流失
INACTIVITY_BUCKETS = [(0, 7, "0-7天"), (8, 30, "8-30天"), (31, 60, "31-60天"),
                      (61, 90, "61-90天"), (91, 10_000, "90天以上")]
EVENT_PRE_DAYS, EVENT_POST_DAYS = 14, 7
WARMUP_DAYS = 30   # 資料期間開頭：期間前已在捐的人會被誤判為「新」，基準期排除這段
EVENT_CHANNEL_MAP = {"電子報": {"電子報"}, "市集": {"市集"}, "社群貼文": {"社群"},
                     "媒體報導": {"搜尋", "官網"}}
ALPHA = 0.05


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
    # continued fraction for Q
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
    """P(X >= observed | Poisson(expected))：活動後新捐款人是否「多到不像巧合」。"""
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


# ---------------------------------------------------------------------------
# 1. 載入
# ---------------------------------------------------------------------------
@dataclass
class Data:
    contacts: pd.DataFrame
    recurring: pd.DataFrame
    one_time: pd.DataFrame
    plans: pd.DataFrame
    events: pd.DataFrame
    as_of: pd.Timestamp = None
    period_start: pd.Timestamp = None
    notes: list = field(default_factory=list)


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df.columns = [COLUMN_ALIASES.get(c.strip(), c.strip()) for c in df.columns]
    return df.apply(lambda s: s.str.strip() if s.dtype == object else s)


def load(data_dir: Path, as_of: str | None = None) -> Data:
    raw = {k: _read(data_dir / v) for k, v in FILES.items()}
    c, r, o, p, e = (raw[k] for k in ["contacts", "recurring", "one_time", "plans", "events"])
    for df, cols in [(c, ["建立日期", "首次捐款日期"]), (r, ["捐款日期"]), (o, ["捐款日期"]),
                     (p, ["加入日期", "最近失敗日期"]), (e, ["日期"])]:
        for col in cols:
            df[col + "_raw"] = df[col]
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for df, cols in [(r, ["金額", "扣款期數"]), (o, ["金額"]),
                     (p, ["約定金額", "執行期數", "約定總期數", "成功次數", "請款次數"])]:
        for col in cols:
            df[col + "_raw"] = df[col]
            df[col] = pd.to_numeric(df[col], errors="coerce")
    d = Data(c, r, o, p, e)
    all_dates = pd.concat([r["捐款日期"], o["捐款日期"]]).dropna()
    d.period_start = all_dates.min().normalize()
    d.as_of = pd.Timestamp(as_of) if as_of else all_dates.max().normalize()
    return d


# ---------------------------------------------------------------------------
# 2. 資料品質檢查（Jill：可靠性；Leo：髒資料定義）
# ---------------------------------------------------------------------------
def validate(d: Data) -> pd.DataFrame:
    rows = []

    def add(table, check, bad, total, severity, note, examples=None):
        rows.append({
            "資料表": table, "檢查項目": check, "異常筆數": int(bad), "總筆數": int(total),
            "異常比例": round(bad / total, 4) if total else 0.0, "嚴重度": severity if bad else "OK",
            "對分析的影響": note, "範例": ", ".join(map(str, (examples or [])[:5])) if bad else "",
        })

    c, r, o, p = d.contacts, d.recurring, d.one_time, d.plans
    # --- 主鍵 / 外鍵
    for name, df, key in [("聯絡人", c, "系統編號"), ("定期扣款", r, "交易編號"),
                          ("單筆捐款", o, "交易編號"), ("定期明細", p, "定期捐款編號")]:
        dup = df[key].duplicated(keep=False)
        add(name, f"{key} 重複", dup.sum(), len(df), "ERROR", "主鍵重複會讓 join 後筆數膨脹",
            df.loc[dup, key].unique().tolist())
    ids = set(c["系統編號"])
    for name, df in [("定期扣款", r), ("單筆捐款", o), ("定期明細", p)]:
        bad = ~df["系統編號"].isin(ids)
        add(name, "系統編號 對不到聯絡人", bad.sum(), len(df), "ERROR",
            "捐款會缺年齡/性別/地區，分群分析會漏人", df.loc[bad, "系統編號"].tolist())
    bad = ~r["定期捐款編號"].isin(set(p["定期捐款編號"]))
    add("定期扣款", "定期捐款編號 對不到定期明細", bad.sum(), len(r), "ERROR",
        "扣款無法對應方案狀態", r.loc[bad, "定期捐款編號"].tolist())
    m = r.merge(p[["定期捐款編號", "系統編號"]], on="定期捐款編號", how="inner", suffixes=("", "_明細"))
    bad = m["系統編號"] != m["系統編號_明細"]
    add("定期扣款", "扣款的系統編號 ≠ 方案的系統編號", bad.sum(), len(m), "ERROR",
        "同一筆扣款歸屬兩個人", m.loc[bad, "交易編號"].tolist())

    # --- 格式
    for name, df, col in [("定期扣款", r, "捐款日期"), ("單筆捐款", o, "捐款日期"),
                          ("聯絡人", c, "建立日期"), ("定期明細", p, "加入日期")]:
        bad = df[col].isna() & df[col + "_raw"].notna()
        add(name, f"{col} 無法解析", bad.sum(), len(df), "ERROR", "該列會被排除",
            df.loc[bad, col + "_raw"].tolist())
    for name, df, col in [("定期扣款", r, "金額"), ("單筆捐款", o, "金額"), ("定期明細", p, "約定金額")]:
        bad = df[col].isna() | (df[col] <= 0)
        add(name, f"{col} 非正數或空白", bad.sum(), len(df), "ERROR", "金額統計會失真",
            df.loc[bad, col + "_raw"].tolist())
        q1, q3 = df[col].quantile([.25, .75])
        hi = df[col] > q3 + 3 * (q3 - q1)
        add(name, f"{col} 極端值（> Q3 + 3×IQR）", hi.sum(), len(df), "INFO",
            "預設保留；若要排除請在 SKILL 流程中先與 NGO 確認", df.loc[hi, col].tolist())

    # --- 邏輯一致性（模擬資料常見問題）
    first = pd.concat([r[["系統編號", "捐款日期"]], o[["系統編號", "捐款日期"]]]) \
        .groupby("系統編號")["捐款日期"].min()
    cj = c.set_index("系統編號").join(first.rename("_first"))
    bad = cj["建立日期"] > cj["_first"]
    add("聯絡人", "建立日期 晚於 首次捐款日期", bad.sum(), cj["_first"].notna().sum(), "WARN",
        "建立日期不能當作『開始支持的時間』；改用實際首捐日期", cj.index[bad].tolist())
    bad = cj["首次捐款日期"].notna() & (cj["首次捐款日期"] != cj["_first"])
    add("聯絡人", "首次捐款日期 ≠ 交易紀錄中最早一筆", bad.sum(), len(cj), "WARN",
        "以交易紀錄為準", cj.index[bad].tolist())
    flag = cj["期間捐款人"].eq("是")
    bad = flag != cj["_first"].notna()
    add("聯絡人", "期間捐款人 旗標與交易紀錄不符", bad.sum(), len(cj), "WARN",
        "以交易紀錄為準", cj.index[bad].tolist())

    pm = r.merge(p[["定期捐款編號", "約定金額"]], on="定期捐款編號")
    bad = pm["金額"] != pm["約定金額"]
    add("定期扣款", "扣款金額 ≠ 約定金額", bad.sum(), len(pm), "INFO",
        "可能是調整金額（加碼/降額）", pm.loc[bad, "交易編號"].tolist())
    r_month = r.assign(_m=r["捐款日期"].dt.to_period("M"))
    per_m = r_month.groupby(["定期捐款編號", "_m"]).size()
    add("定期扣款", "同一方案同月扣款超過 1 次", (per_m > 1).sum(), len(per_m), "WARN",
        "月扣款理應每月一次；留存率應以『當月是否有扣款』計，不以筆數計",
        [f"{a}@{b}" for a, b in per_m[per_m > 1].index[:5]])
    mono = r.sort_values(["定期捐款編號", "捐款日期"]).groupby("定期捐款編號")["扣款期數"] \
        .apply(lambda s: s.is_monotonic_increasing)
    add("定期扣款", "扣款期數 未隨日期遞增", (~mono).sum(), len(mono), "WARN",
        "扣款期數 不可靠 → 『第幾期停止』改用 定期明細.執行期數", mono.index[~mono].tolist())
    cnt = r.groupby("定期捐款編號").size()
    pj = p.set_index("定期捐款編號").join(cnt.rename("_n"))
    bad = pj["_n"].fillna(0) != pj["成功次數"]
    add("定期明細", "成功次數 ≠ 扣款紀錄筆數", bad.sum(), len(pj), "WARN",
        "兩表口徑不同；方案層級指標以定期明細為準，時間序列以扣款紀錄為準", pj.index[bad].tolist())
    bad = pj["執行期數"] != pj["成功次數"]
    add("定期明細", "執行期數 ≠ 成功次數", bad.sum(), len(pj), "INFO", "", pj.index[bad].tolist())
    bad = (pj["請款次數"] > pj["成功次數"]) != pj["最近失敗日期"].notna()
    add("定期明細", "有失敗次數但無失敗日期（或相反）", bad.sum(), len(pj), "WARN",
        "失敗挽回名單可能漏人", pj.index[bad].tolist())
    done = pj["狀態"].eq(COMPLETED_STATUS) & (pj["約定總期數"] > 0)
    bad = done & (pj["執行期數"] < pj["約定總期數"])
    add("定期明細", "狀態=已完成 但 執行期數 < 約定總期數", bad.sum(), done.sum(), "WARN",
        "『已完成』可能包含提前結束；不要直接當成功案例", pj.index[bad].tolist())
    months_since_join = ((d.as_of - pj["加入日期"]).dt.days / 30.44).round()
    bad = pj["執行期數"] > months_since_join + 1
    add("定期明細", "執行期數 > 加入至今月數", bad.sum(), len(pj), "WARN",
        "期數與加入日期矛盾", pj.index[bad].tolist())
    period_months = round((d.as_of - d.period_start).days / 30.44)
    old = pj["加入日期"] < d.period_start - pd.Timedelta(days=31)
    capped = old & (pj["執行期數"].max() <= period_months)
    add("定期明細", f"執行期數上限 ≤ 觀察期月數（{period_months}）但方案更早加入", capped.sum(), old.sum(), "WARN",
        "執行期數疑似只計觀察期內 → 『第幾期停止』的 hazard 在後段會被機械性放大，請看獨立性檢定",
        pj.index[capped].tolist())
    new_in_period = (pj["加入日期"] >= d.period_start).sum()
    rows.append({"資料表": "定期明細", "檢查項目": "期間內新加入的定期方案數", "異常筆數": int(new_in_period),
                 "總筆數": len(pj), "異常比例": round(new_in_period / len(pj), 4),
                 "嚴重度": "WARN" if new_in_period == 0 else "INFO",
                 "對分析的影響": "若為 0：無法用加入日期觀察『單筆轉定期』，任務三需改用代理指標並註明缺資料",
                 "範例": ""})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. 整合主檔（Join）
# ---------------------------------------------------------------------------
def build_donations(d: Data) -> pd.DataFrame:
    r = d.recurring.assign(捐款類型="定期定額", 捐款管道="")
    o = d.one_time.assign(捐款類型="單筆", 定期捐款編號="", 扣款期數=np.nan)
    cols = ["交易編號", "系統編號", "捐款日期", "金額", "付款工具", "捐款類型", "定期捐款編號", "扣款期數", "捐款管道"]
    tx = pd.concat([r[cols], o[cols]], ignore_index=True).dropna(subset=["捐款日期", "金額"])
    tx = tx.merge(d.contacts[["系統編號", "年齡級距", "性別", "鄉鎮市區"]], on="系統編號", how="left")
    tx["月份"] = tx["捐款日期"].dt.to_period("M").astype(str)
    first = tx.groupby("系統編號")["捐款日期"].transform("min")
    tx["Cohort月份"] = first.dt.to_period("M").astype(str)
    tx["距首捐月數"] = (tx["捐款日期"].dt.year - first.dt.year) * 12 + tx["捐款日期"].dt.month - first.dt.month
    return tx.sort_values(["系統編號", "捐款日期", "交易編號"]).reset_index(drop=True)


def build_donors(d: Data, tx: pd.DataFrame) -> pd.DataFrame:
    c = d.contacts[["系統編號", "年齡級距", "性別", "鄉鎮市區", "建立日期"]].copy()
    g = tx.groupby("系統編號")
    first_row = tx.drop_duplicates("系統編號", keep="first").set_index("系統編號")
    agg = pd.DataFrame({
        "首捐日期": g["捐款日期"].min(), "最近捐款日期": g["捐款日期"].max(),
        "捐款筆數": g.size(), "總捐款金額": g["金額"].sum(),
        "定期扣款筆數": g["捐款類型"].apply(lambda s: (s == "定期定額").sum()),
        "單筆筆數": g["捐款類型"].apply(lambda s: (s == "單筆").sum()),
        "首捐類型": first_row["捐款類型"], "首捐管道": first_row["捐款管道"],
    })
    plans_per = d.plans.groupby("系統編號").agg(定期方案數=("定期捐款編號", "size"),
                                                最早方案加入日期=("加入日期", "min"))
    out = c.merge(agg, left_on="系統編號", right_index=True, how="left") \
           .merge(plans_per, left_on="系統編號", right_index=True, how="left")
    out["期間捐款人"] = out["首捐日期"].notna()
    out["定期方案數"] = out["定期方案數"].fillna(0).astype(int)
    out["年輕支持者(<25)"] = out["年齡級距"].isin(YOUNG_AGE_BANDS)
    out["距今天數"] = (d.as_of - out["最近捐款日期"]).dt.days
    # 期間新捐款人：期間首捐為單筆，且在期間開始前沒有任何定期方案
    pre_plan = out["最早方案加入日期"].notna() & (out["最早方案加入日期"] < d.period_start)
    out["期間新捐款人"] = out["期間捐款人"] & out["首捐類型"].eq("單筆") & ~pre_plan
    out["暖機期首捐"] = out["首捐日期"] < d.period_start + pd.Timedelta(days=WARMUP_DAYS)
    return out


def build_plans(d: Data, tx: pd.DataFrame) -> pd.DataFrame:
    rtx = tx[tx["捐款類型"] == "定期定額"]
    g = rtx.groupby("定期捐款編號")
    agg = pd.DataFrame({
        "扣款紀錄筆數": g.size(), "首次扣款日": g["捐款日期"].min(), "最近扣款日": g["捐款日期"].max(),
        "扣款月數": g["月份"].nunique(), "主要付款工具": g["付款工具"].agg(lambda s: s.mode().iat[0]),
    })
    p = d.plans.drop(columns=[c for c in d.plans.columns if c.endswith("_raw")]) \
        .merge(agg, left_on="定期捐款編號", right_index=True, how="left") \
        .merge(d.contacts[["系統編號", "年齡級距", "性別", "鄉鎮市區"]], on="系統編號", how="left")
    p["沉默天數"] = (d.as_of - p["最近扣款日"]).dt.days
    p["沉默級距"] = pd.cut(p["沉默天數"], bins=[-1] + [b[1] for b in INACTIVITY_BUCKETS],
                       labels=[b[2] for b in INACTIVITY_BUCKETS])
    p["有扣款失敗"] = p["請款次數"] > p["成功次數"]
    p["失敗後已恢復"] = p["有扣款失敗"] & (p["最近扣款日"] > p["最近失敗日期"])
    p["失敗距今天數"] = (d.as_of - p["最近失敗日期"]).dt.days
    p["停止"] = p["狀態"].isin(STOPPED_STATUSES)
    p["靜默流失"] = p["狀態"].eq("進行中") & (p["沉默天數"] > SILENT_DAYS)
    p["有設定期限"] = p["約定總期數"] > 0
    p["剩餘期數"] = np.where(p["有設定期限"], p["約定總期數"] - p["執行期數"], np.nan)
    # 續約/加碼：同一人有另一個「加入較晚」的方案
    later = p[["系統編號", "加入日期", "約定金額"]]
    pair = p[["定期捐款編號", "系統編號", "加入日期", "約定金額"]].merge(later, on="系統編號", suffixes=("", "_後"))
    pair = pair[pair["加入日期_後"] > pair["加入日期"]]
    p["之後有再開方案"] = p["定期捐款編號"].isin(pair["定期捐款編號"])
    p["之後有加碼方案"] = p["定期捐款編號"].isin(pair.loc[pair["約定金額_後"] > pair["約定金額"], "定期捐款編號"])
    return p


# ---------------------------------------------------------------------------
# 4. 任務二：留住定期定額捐款人
# ---------------------------------------------------------------------------
def t2_stop_period(plans: pd.DataFrame) -> pd.DataFrame:
    """以執行期數做 Kaplan-Meier：在第 k 期『停止』的條件機率（hazard）。"""
    rows, surv = [], 1.0
    for k in range(1, int(plans["執行期數"].max()) + 1):
        at_risk = (plans["執行期數"] >= k).sum()
        stopped = ((plans["執行期數"] == k) & plans["停止"]).sum()
        completed = ((plans["執行期數"] == k) & plans["狀態"].eq(COMPLETED_STATUS)).sum()
        h = stopped / at_risk if at_risk else 0.0
        surv *= 1 - h
        lo, hi = wilson_ci(int(stopped), int(at_risk))
        rows.append({"期數": k, "仍在觀察中的方案數": int(at_risk), "於此期停止": int(stopped),
                     "於此期已完成": int(completed), "停止率(hazard)": round(h, 4),
                     "停止率95%下界": round(lo, 4), "停止率95%上界": round(hi, 4),
                     "累積存活率": round(surv, 4)})
    df = pd.DataFrame(rows)
    # 檢定：各期停止率是否一致（若一致 → 沒有特定的『危險期數』）
    tab = pd.DataFrame({"停止": df["於此期停止"], "未停止": df["仍在觀察中的方案數"] - df["於此期停止"]})
    stat, dof, pval = chi2_independence(tab)
    df.attrs["test_hazard"] = {"檢定": "各期 hazard 是否相同（卡方；受觀察窗上限影響）", "p值": round(pval, 4),
                               "判定": verdict(pval, int(tab["停止"].sum()), 30)}
    # 主要判定：停止與否 是否和 執行期數 相關？若無關，hazard 上升只是「人越來越少」的機械效果
    stat2, dof2, p2 = chi2_independence(pd.crosstab(plans["執行期數"], plans["停止"]))
    df.attrs["test"] = {"檢定": "停止與否 是否與 執行期數 相關（卡方獨立性）", "p值": round(p2, 4),
                        "判定": verdict(p2, int(plans["停止"].sum()))}
    return df


def t2_term_vs_actual(plans: pd.DataFrame) -> pd.DataFrame:
    g = plans.groupby("約定總期數")
    df = pd.crosstab(plans["約定總期數"], plans["狀態"])
    df["方案數"] = g.size()
    df["平均執行期數"] = g["執行期數"].mean().round(2)
    df["停止率"] = g["停止"].mean().round(4)
    full = (plans["約定總期數"] > 0) & (plans["執行期數"] >= plans["約定總期數"])
    early = plans["狀態"].eq(COMPLETED_STATUS) & (plans["執行期數"] < plans["約定總期數"])
    df["真正扣滿約定期數"] = full.groupby(plans["約定總期數"]).sum()
    df["已完成但未扣滿"] = early.groupby(plans["約定總期數"]).sum()
    df["之後有再開方案"] = g["之後有再開方案"].sum()
    df["之後有加碼方案"] = g["之後有加碼方案"].sum()
    df.index = df.index.map(lambda v: "未設定期限" if v == 0 else f"{int(v)}期")
    stat, dof, pval = chi2_independence(pd.crosstab(plans["約定總期數"], plans["停止"]))
    df.attrs["test"] = {"檢定": "有無設定期限/期限長短 與 停止 是否相關（卡方）", "p值": round(pval, 4),
                        "判定": verdict(pval, int(plans["停止"].sum()))}
    return df.reset_index().rename(columns={"約定總期數": "設定期數"})


def t2_failure_recovery(plans: pd.DataFrame) -> pd.DataFrame:
    f = plans[plans["有扣款失敗"]]
    g = f.groupby("主要付款工具")
    df = pd.DataFrame({"有失敗方案數": g.size(), "已恢復": g["失敗後已恢復"].sum()})
    df["恢復率"] = (df["已恢復"] / df["有失敗方案數"]).round(4)
    all_g = plans.groupby("主要付款工具")
    df["失敗發生率"] = (all_g["有扣款失敗"].mean()).round(4)
    stat, dof, pval = chi2_independence(pd.crosstab(f["主要付款工具"], f["失敗後已恢復"]))
    df.attrs["test"] = {"檢定": "付款工具 與 失敗後是否恢復（卡方）", "p值": round(pval, 4),
                        "判定": verdict(pval, int(len(f)))}
    return df.reset_index()


def t2_inactivity(plans: pd.DataFrame) -> pd.DataFrame:
    df = pd.crosstab(plans["沉默級距"], plans["狀態"], margins=True, margins_name="合計")
    return df.reset_index().rename(columns={"沉默級距": "距最近一次扣款"})


def charge_gap_p90(tx: pd.DataFrame) -> int:
    """同一方案相鄰兩次扣款的間隔天數 P90：判斷『沒扣款＝流失』前，至少要等這麼久。"""
    r = tx[tx["捐款類型"] == "定期定額"].sort_values(["定期捐款編號", "捐款日期"])
    gaps = r.groupby("定期捐款編號")["捐款日期"].diff().dt.days.dropna()
    return int(max(SILENT_DAYS, gaps.quantile(0.9))) if len(gaps) else SILENT_DAYS


def t2_monthly_churn(plans: pd.DataFrame, as_of: pd.Timestamp, min_wait: int = SILENT_DAYS) -> pd.DataFrame:
    """月流失：最後一次扣款落在 m 月、之後到觀察截止都沒有再扣 = m 月流失。
    距截止日不足 min_wait 天的月份無法判定（扣款間隔本來就可能這麼長），標示為『觀察不足』。"""
    last_m = plans["最近扣款日"].dt.to_period("M")
    first_m = plans["首次扣款日"].dt.to_period("M")
    months = pd.period_range(first_m.min(), as_of.to_period("M"))
    rows = []
    for m in months:
        active = ((first_m <= m) & (last_m >= m)).sum()
        churned = (last_m == m).sum()
        enough = (as_of - m.end_time).days >= min_wait
        rows.append({"月份": str(m), "期初活躍方案": int(active),
                     "本月後未再扣款": int(churned) if enough else None,
                     "月流失率": round(churned / active, 4) if (active and enough) else None,
                     "備註": "" if enough else f"距截止不足 {min_wait} 天（扣款間隔P90），尚無法判定"})
    return pd.DataFrame(rows)


def cohort_retention(tx: pd.DataFrame, donors: pd.DataFrame, kind: str) -> pd.DataFrame:
    """kind = 'recurring'：以首次扣款月份分群，看之後每月是否仍有扣款
       kind = 'new'      ：期間新捐款人，以首捐月份分群，看之後每月是否有任何捐款"""
    if kind == "recurring":
        t = tx[tx["捐款類型"] == "定期定額"].copy()
        t["c"] = t.groupby("系統編號")["捐款日期"].transform("min").dt.to_period("M")
    else:
        ids = set(donors.loc[donors["期間新捐款人"], "系統編號"])
        t = tx[tx["系統編號"].isin(ids)].copy()
        t["c"] = t.groupby("系統編號")["捐款日期"].transform("min").dt.to_period("M")
    t["m"] = t["捐款日期"].dt.to_period("M")
    t["age"] = (t["m"] - t["c"]).apply(lambda x: x.n)
    size = t.groupby("c")["系統編號"].nunique()
    act = t.groupby(["c", "age"])["系統編號"].nunique().unstack(fill_value=0)
    rate = act.div(size, axis=0).round(4)
    # 觀察不到的格子設為 NaN（不是 0）
    last = t["m"].max()
    for cm in rate.index:
        max_age = (last - cm).n
        rate.loc[cm, rate.columns > max_age] = np.nan
    rate.insert(0, "Cohort人數", size)
    rate.index = rate.index.astype(str)
    rate.columns = ["Cohort人數"] + [f"M{a}" for a in rate.columns[1:]]
    return rate.reset_index().rename(columns={"c": "Cohort月份"})


SEGMENT_COLS = ["年齡級距", "性別", "鄉鎮市區", "約定金額", "約定總期數", "主要付款工具"]


def segment_compare(df: pd.DataFrame, flag: str, cols=SEGMENT_COLS, label="") -> pd.DataFrame:
    out = []
    cols = [c for c in cols if c in df]
    for col in cols:
        g = df.groupby(col)[flag]
        stat, dof, pval = chi2_independence(pd.crosstab(df[col], df[flag]))
        p_adj = min(1.0, pval * len(cols))  # Bonferroni：一次比很多維度，總會有一個「碰巧顯著」
        v = verdict(p_adj, int(g.size().min()))
        for val, s in g:
            lo, hi = wilson_ci(int(s.sum()), len(s))
            out.append({"指標": label or flag, "分群維度": col, "群組": str(val), "人數/方案數": len(s),
                        "比率": round(s.mean(), 4), "95%下界": round(lo, 4), "95%上界": round(hi, 4),
                        "整體比率": round(df[flag].mean(), 4), "維度p值": round(pval, 4),
                        "校正後p值": round(p_adj, 4), "判定": v})
    return pd.DataFrame(out)


def t2_reminder_list(plans: pd.DataFrame) -> pd.DataFrame:
    """優先提醒名單：規則透明、每一分都有理由，方便人工檢查。"""
    p = plans.copy()
    score = pd.Series(0.0, index=p.index)
    reasons = [[] for _ in range(len(p))]
    action = pd.Series("", index=p.index)

    def rule(mask, pts, reason, act):
        nonlocal score
        mask = mask.fillna(False).to_numpy()
        score += np.where(mask, pts, 0)
        for i in np.flatnonzero(mask):
            reasons[i].append(reason)
        empty = (action == "").to_numpy() & mask
        action[empty] = act

    rule(p["有扣款失敗"] & ~p["失敗後已恢復"] & ~p["狀態"].eq(COMPLETED_STATUS), 40,
         "扣款失敗後尚未成功扣款", "付款失敗挽回：提醒更新付款資訊")
    rule(p["狀態"].eq("暫停中"), 25, "方案暫停中", "恢復邀請：分享近期救援成果")
    rule(p["狀態"].eq("待處理"), 20, "方案待處理", "確認付款設定")
    rule(p["靜默流失"], 25, f"進行中但超過 {SILENT_DAYS} 天無扣款", "靜默關懷：確認是否仍願意支持")
    rule(p["有設定期限"] & p["剩餘期數"].between(0, 2) & ~p["狀態"].eq(COMPLETED_STATUS) & ~p["停止"], 15,
         "約定期數即將到期", "續約邀請")
    rule(p["狀態"].eq(COMPLETED_STATUS) & ~p["之後有再開方案"], 20, "已完成且尚未續約", "續約/加碼邀請")
    rule(p["狀態"].eq("已失敗"), 5, "已失敗（挽回難度高）", "低優先挽回")
    value = (p["約定金額"] / 100).clip(upper=15)
    score += value
    tenure = p["執行期數"] >= 6
    score += np.where(tenure, 5, 0)
    for i in np.flatnonzero(tenure.to_numpy()):
        reasons[i].append("已穩定扣款 6 期以上")
    p["優先分數"] = score.round(1)
    p["理由"] = ["；".join(x) for x in reasons]
    p["建議行動"] = action
    p = p[p["建議行動"] != ""]
    cols = ["定期捐款編號", "系統編號", "優先分數", "建議行動", "理由", "狀態", "約定金額", "執行期數",
            "約定總期數", "沉默天數", "最近失敗日期", "主要付款工具", "年齡級距", "性別", "鄉鎮市區"]
    return p.sort_values("優先分數", ascending=False)[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. 任務三：找出讓人開始支持的時刻
# ---------------------------------------------------------------------------
def t3_event_lift(d: Data, donors: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    new = donors[donors["期間新捐款人"]]
    days = pd.date_range(d.period_start, d.as_of)
    daily = new["首捐日期"].dt.normalize().value_counts().reindex(days, fill_value=0)
    warm_end = d.period_start + pd.Timedelta(days=WARMUP_DAYS)
    first_tx = tx[tx["系統編號"].isin(new["系統編號"])].drop_duplicates("系統編號")
    base_ch = first_tx[first_tx["捐款日期"] >= warm_end]["捐款管道"].value_counts(normalize=True)
    rows = []
    for _, ev in d.events.sort_values("日期").iterrows():
        t = ev["日期"]
        pre = daily[max(t - pd.Timedelta(days=EVENT_PRE_DAYS), warm_end): t - pd.Timedelta(days=1)]
        post = daily[t: t + pd.Timedelta(days=EVENT_POST_DAYS - 1)]
        pre_rate = pre.mean() if len(pre) else np.nan
        exp = pre_rate * len(post) if len(pre) else np.nan
        obs = int(post.sum())
        pval = poisson_upper_p(obs, exp) if len(pre) >= 7 else np.nan
        win = first_tx[(first_tx["捐款日期"] >= t) & (first_tx["捐款日期"] < t + pd.Timedelta(days=EVENT_POST_DAYS))]
        chans = EVENT_CHANNEL_MAP.get(ev["事件類型"], set())
        match_share = win["捐款管道"].isin(chans).mean() if len(win) else np.nan
        base_share = sum(base_ch.get(ch, 0) for ch in chans)
        if np.isnan(pval):
            v = "基準期不足（暖機期內）：不宜下結論"
        else:
            v = "新捐款人顯著增加" if pval < ALPHA else "無顯著增加：不宜歸因於此活動"
        rows.append({"事件編號": ev["事件編號"], "日期": t.date().isoformat(), "事件類型": ev["事件類型"],
                     "事件名稱": ev["事件名稱"], "活動前日均新捐款人": round(pre_rate, 2),
                     f"活動後{EVENT_POST_DAYS}天新捐款人": obs, "預期人數": round(exp, 1),
                     "增幅倍數": round(obs / exp, 2) if exp else np.nan, "p值": round(pval, 4),
                     "對應管道占比(活動窗)": round(match_share, 3), "對應管道占比(基準)": round(base_share, 3),
                     "判定": v})
    return pd.DataFrame(rows)


def t3_new_donor_followup(d: Data, donors: pd.DataFrame, tx: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """新捐款人之後是否再捐？是否轉定期？依首捐管道 & 是否在活動窗內首捐分組。"""
    new = donors[donors["期間新捐款人"]].copy()
    t = tx[tx["系統編號"].isin(new["系統編號"])]
    first = new.set_index("系統編號")["首捐日期"]
    t = t.join(first.rename("_f"), on="系統編號")
    later = t[t["捐款日期"] > t["_f"]]
    for w in (30, 90):
        ids = later.loc[later["捐款日期"] <= later["_f"] + pd.Timedelta(days=w), "系統編號"].unique()
        new[f"{w}天內再捐"] = new["系統編號"].isin(ids)
        new[f"可觀察滿{w}天"] = new["首捐日期"] <= d.as_of - pd.Timedelta(days=w)
    new["之後有定期扣款"] = new["系統編號"].isin(later.loc[later["捐款類型"] == "定期定額", "系統編號"])
    win = pd.Series(False, index=new.index)
    for t0 in events["日期"]:
        win |= new["首捐日期"].between(t0, t0 + pd.Timedelta(days=EVENT_POST_DAYS - 1))
    new["活動窗內首捐"] = np.where(win, "活動後7天內", "其他時間")
    out = []
    for col in ["首捐管道", "活動窗內首捐"]:
        for w in (30, 90):
            sub = new[new[f"可觀察滿{w}天"]]
            seg = segment_compare(sub, f"{w}天內再捐", cols=[col], label=f"{w}天內再捐率")
            out.append(seg)
    conv = segment_compare(new, "之後有定期扣款", cols=["首捐管道"], label="轉為定期扣款率")
    out.append(conv)
    return pd.concat(out, ignore_index=True), new


def t3_weekly_new(donors: pd.DataFrame) -> pd.DataFrame:
    new = donors[donors["期間新捐款人"]]
    w = new["首捐日期"].dt.to_period("W-SUN").dt.start_time
    return w.value_counts().sort_index().rename_axis("週起始日").reset_index(name="新捐款人數")


# ---------------------------------------------------------------------------
# 6. 匯總 + 報告
# ---------------------------------------------------------------------------
def overview(d: Data, donors: pd.DataFrame, plans: pd.DataFrame, tx: pd.DataFrame) -> dict:
    dn = donors[donors["期間捐款人"]]
    young_c = donors["年輕支持者(<25)"].mean()
    young_d = dn["年輕支持者(<25)"].mean()
    stat, dof, pval = chi2_independence(pd.crosstab(donors["年輕支持者(<25)"], donors["期間捐款人"]))
    return {
        "資料期間": f"{d.period_start.date()} ~ {d.as_of.date()}",
        "聯絡人數": int(len(donors)), "期間捐款人數": int(len(dn)),
        "期間捐款率": round(len(dn) / len(donors), 4),
        "捐款總額": int(tx["金額"].sum()), "交易筆數": int(len(tx)),
        "定期方案數": int(len(plans)), "定期捐款人數": int(plans["系統編號"].nunique()),
        "期間新捐款人數": int(donors["期間新捐款人"].sum()),
        "期間新加入定期方案數": int((plans["加入日期"] >= d.period_start).sum()),
        "方案狀態分布": plans["狀態"].value_counts().to_dict(),
        "扣款失敗方案數": int(plans["有扣款失敗"].sum()),
        "失敗後已恢復方案數": int(plans["失敗後已恢復"].sum()),
        "靜默流失方案數": int(plans["靜默流失"].sum()),
        "25歲以下占聯絡人": round(young_c, 4), "25歲以下占捐款人": round(young_d, 4),
        "25歲以下捐款率差異判定": verdict(pval, int(donors["年輕支持者(<25)"].sum())),
    }


def run(data_dir: Path, out_dir: Path, as_of: str | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    d = load(data_dir, as_of)
    checks = validate(d)
    if (checks["嚴重度"] == "ERROR").any():
        print("⚠ 有 ERROR 等級的資料問題，結果可能失真，請先看 00_資料品質檢查.csv")
    tx = build_donations(d)
    donors = build_donors(d, tx)
    plans = build_plans(d, tx)

    res = {}
    res["t2_停止期數"] = t2_stop_period(plans)
    res["t2_設定期數vs實際"] = t2_term_vs_actual(plans)
    res["t2_扣款失敗與恢復"] = t2_failure_recovery(plans)
    res["t2_沉默天數分布"] = t2_inactivity(plans)
    gap90 = charge_gap_p90(tx)
    res["t2_月流失率"] = t2_monthly_churn(plans, d.as_of, gap90)
    res["t2_定期Cohort留存"] = cohort_retention(tx, donors, "recurring")
    res["t2_分群比較_停止"] = segment_compare(plans, "停止", label="停止率")
    res["t2_分群比較_靜默流失"] = segment_compare(plans, "靜默流失", label="靜默流失率")
    res["t2_優先提醒名單"] = t2_reminder_list(plans)
    res["t3_活動前後新捐款人"] = t3_event_lift(d, donors, tx)
    res["t3_新捐款人後續行為"], new_detail = t3_new_donor_followup(d, donors, tx, d.events)
    res["t3_每週新捐款人"] = t3_weekly_new(donors)
    res["t3_新捐款人Cohort留存"] = cohort_retention(tx, donors, "new")

    def save(df, name):
        df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    save(checks, "00_資料品質檢查")
    save(tx, "01_捐款明細_整合")
    save(donors, "02_捐款人主檔")
    save(plans, "03_定期方案主檔")
    save(new_detail, "04_新捐款人明細")
    for k, v in res.items():
        save(v, k)

    tests = {k: v.attrs["test"] for k, v in res.items() if "test" in v.attrs}
    seg_verdicts = {}
    for k in ["t2_分群比較_停止", "t2_分群比較_靜默流失", "t3_新捐款人後續行為"]:
        df = res[k]
        for (ind, dim), s in df.groupby(["指標", "分群維度"]):
            seg_verdicts[f"{ind}｜{dim}"] = {"p值": float(s["維度p值"].iat[0]),
                                             "校正後p值": float(s["校正後p值"].iat[0]), "判定": s["判定"].iat[0]}
    summary = {
        "總覽": overview(d, donors, plans, tx),
        "資料品質": checks[checks["嚴重度"] != "OK"][["資料表", "檢查項目", "異常筆數", "異常比例", "嚴重度", "對分析的影響"]]
        .to_dict("records"),
        "檢定結果": tests,
        "分群檢定": seg_verdicts,
        "活動判定": res["t3_活動前後新捐款人"][["事件編號", "事件名稱", "增幅倍數", "p值", "判定"]].to_dict("records"),
        "優先提醒名單": {"總數": int(len(res["t2_優先提醒名單"])),
                     "依建議行動": res["t2_優先提醒名單"]["建議行動"].value_counts().to_dict()},
        "參數": {"停止狀態": sorted(STOPPED_STATUSES), "靜默流失天數": SILENT_DAYS,
               "活動前基準天數": EVENT_PRE_DAYS, "活動後觀察天數": EVENT_POST_DAYS,
               "暖機期天數": WARMUP_DAYS, "顯著水準": ALPHA, "扣款間隔P90天數": gap90,
               "多重比較校正": "Bonferroni（同一指標內的分群維度數）"},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                          encoding="utf-8")
    try:
        from report import build_report
        build_report(out_dir, summary, checks, res)
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location("report", Path(__file__).with_name("report.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.build_report(out_dir, summary, checks, res)
    return summary


def main():
    ap = argparse.ArgumentParser(description="野灣捐款資料 Cohort / 留存 / 支持時刻 分析")
    ap.add_argument("--data-dir", type=Path, default=Path("."))
    ap.add_argument("--output-dir", type=Path, default=Path("yewan_output"))
    ap.add_argument("--as-of", default=None, help="觀察截止日（預設為資料中最後一筆捐款日）")
    a = ap.parse_args()
    s = run(a.data_dir, a.output_dir, a.as_of)
    o = s["總覽"]
    print(f"資料期間 {o['資料期間']}｜捐款人 {o['期間捐款人數']:,}／聯絡人 {o['聯絡人數']:,}｜定期方案 {o['定期方案數']:,}")
    print(f"輸出：{a.output_dir.resolve()}（先開 report.html）")


if __name__ == "__main__":
    main()
