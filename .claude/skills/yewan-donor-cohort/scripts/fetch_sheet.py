"""從 Google Sheet 下載各分頁成 CSV，供 yewan_cohort.py 使用。

用法：
    python fetch_sheet.py --sheet-id <ID> --out-dir data

前提：Sheet 已設成「知道連結的任何人都可以檢視」。
⚠ 只適用於虛構/去識別化資料；真實捐款人資料不要公開分享。

分頁名稱預設和 CSV 檔名相同（Contacts、Recurring_Donations…）。
分頁名稱不同時，用 --tab 檔名=分頁名 覆寫，例如 --tab Contacts=聯絡人。
"""
import argparse
import sys
import urllib.parse
import urllib.request
from pathlib import Path

TABS = ["Contacts", "Recurring_Donations", "One_Time_Donations",
        "Recurring_Details", "Event_Timeline"]


def fetch(sheet_id: str, tab: str) -> bytes:
    url = (f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?"
           + urllib.parse.urlencode({"tqx": "out:csv", "sheet": tab}))
    with urllib.request.urlopen(url, timeout=120) as r:
        body = r.read()
    if body.lstrip().startswith(b"<"):
        raise RuntimeError(f"分頁 {tab} 回傳 HTML（Sheet 未公開或分頁名稱錯誤）")
    return body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet-id", required=True)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--tab", action="append", default=[],
                    help="檔名=分頁名，可重複")
    a = ap.parse_args()
    mapping = {t: t for t in TABS}
    for kv in a.tab:
        k, v = kv.split("=", 1)
        mapping[k] = v
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for fname, tab in mapping.items():
        body = fetch(a.sheet_id, tab)
        (out / f"{fname}.csv").write_bytes(body)
        rows = body.count(b"\n")
        print(f"{fname}.csv  ←  分頁「{tab}」  {rows:,} 列")
    return 0


if __name__ == "__main__":
    sys.exit(main())
