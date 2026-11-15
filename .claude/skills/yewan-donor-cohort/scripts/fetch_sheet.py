"""把官方資料（xlsx）轉成 yewan_cohort.py 使用的 5 個 CSV。

用法（三選一）：
    python fetch_sheet.py --xlsx 野灣_示範資料_v1.xlsx --out-dir data      # 本機 xlsx
    python fetch_sheet.py --drive-id <Drive 檔案 ID> --out-dir data       # Drive 上的 xlsx（需「知道連結者可檢視」）
    python fetch_sheet.py --sheet-id <Google Sheet ID> --out-dir data     # 原生 Google 試算表

⚠ 只適用於虛構/去識別化資料；真實捐款人資料不要公開分享。

官方 xlsx 的分頁（以開頭圓圈數字辨識，分頁名稱改字也不影響）：
    ①聯絡人                → Contacts.csv
    ②一年內定期定額紀錄    → Recurring_Donations.csv
    ③一年內單筆捐款紀錄    → One_Time_Donations.csv
    ④定期定額明細          → Recurring_Details.csv
    ⑤活動時間軸（虛構）    → Event_Timeline.csv   （表頭不在第一列，會自動找「日期」那一列）
「讀我」「⑥任務卡」「⑦欄位說明」不需要轉換。
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.parse
import urllib.request
from pathlib import Path

TABS = {  # 輸出檔名: (分頁開頭符號, 表頭一定會出現的欄位)
    "Contacts": ("①", "系統編號"),
    "Recurring_Donations": ("②", "系統編號"),
    "One_Time_Donations": ("③", "系統編號"),
    "Recurring_Details": ("④", "系統編號"),
    "Event_Timeline": ("⑤", "日期"),
}


def _find_header(raw, key: str):
    """回傳把 key 所在列當表頭後的 DataFrame。"""
    for i in range(min(15, len(raw))):
        vals = [str(v).strip() for v in raw.iloc[i].tolist()]
        if key in vals:
            df = raw.iloc[i + 1:].copy()
            df.columns = vals
            df = df.loc[:, [c not in ("", "nan", "None") for c in df.columns]]
            return df.dropna(how="all")
    raise ValueError(f"找不到含「{key}」的表頭列")


def xlsx_to_csvs(xlsx, out_dir: Path) -> dict:
    """xlsx（路徑或 bytes）→ out_dir 下 5 個 CSV。回傳 {檔名: 列數}。"""
    import pandas as pd
    src = io.BytesIO(xlsx) if isinstance(xlsx, (bytes, bytearray)) else xlsx
    sheets = pd.read_excel(src, sheet_name=None, header=None, dtype=object)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for fname, (prefix, key) in TABS.items():
        names = [n for n in sheets if n.strip().startswith(prefix)]
        if not names:
            raise ValueError(f"xlsx 中找不到以「{prefix}」開頭的分頁（現有：{list(sheets)}）")
        df = _find_header(sheets[names[0]], key)
        df.to_csv(out_dir / f"{fname}.csv", index=False, encoding="utf-8-sig")
        counts[fname] = len(df)
    return counts


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=180) as r:
        body = r.read()
    if body.lstrip()[:1] == b"<":
        raise RuntimeError("回傳的是 HTML：檔案未公開，或 ID 錯誤")
    return body


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--xlsx", type=Path)
    g.add_argument("--drive-id")
    g.add_argument("--sheet-id")
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    a = ap.parse_args()
    if a.xlsx:
        counts = xlsx_to_csvs(a.xlsx, a.out_dir)
    elif a.drive_id:
        body = _download(f"https://drive.google.com/uc?export=download&id={a.drive_id}")
        counts = xlsx_to_csvs(body, a.out_dir)
    else:
        body = _download(f"https://docs.google.com/spreadsheets/d/{a.sheet_id}/export?format=xlsx")
        counts = xlsx_to_csvs(body, a.out_dir)
    for k, n in counts.items():
        print(f"{k}.csv  {n:,} 列")
    return 0


if __name__ == "__main__":
    sys.exit(main())
