"""
驗證 Skill 產出的數字對不對（可靠性檢查）。

兩種測試：
  A. 定義測試：用一份官方格式的手工小資料（答案可以心算），確認每個指標的定義正確，
     特別是「④ 一人一列」「換卡＝新方案」這些陷阱。
  B. 對帳測試：用「只靠 Python 標準函式庫」的另一套寫法，獨立重算真實資料的關鍵數字，
     和 pandas 管線的輸出比對。兩套程式算出一樣的數字，才算可信。

用法：
    python test_yewan_cohort.py                                  # 只跑 A
    python test_yewan_cohort.py --data 野灣_示範資料_v1.xlsx      # A + B（也可給 CSV 資料夾）
"""
from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd  # noqa: E402
import yewan_cohort as yc  # noqa: E402

DATA: Path | None = None


def write(path: Path, header: list[str], rows: list[list]):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


TX_HEADER = ["系統編號", "性別", "年齡", "住家-縣市", "鄉鎮市區", "捐款編號", "捐款建立日期", "總金額", "交易編號",
             "金流機制", "付款工具", "付款類型", "定期捐款編號"]


def make_fixture(d: Path):
    """5 位聯絡人（含 1 筆重複）、4 個方案（其中 1 個 ④ 沒列）、3 筆單筆、2 個活動。

    S1 (1)：舊聯絡人，期間新開方案 P1(101)，2/5 扣款失敗、3/5 恢復
    S2 (2)：舊方案 P2(102) 在 ④ 顯示已失敗；但 1/25 另開加碼方案 P3(103)（④ 沒列）→ 換卡，不是流失
    S3 (3)：期間新聯絡人，單筆首捐 3/2，4/1 再捐
    S4 (4)：沒有捐款，聯絡人重複兩列
    S5 (5)：期間新開方案 P5(105)，第一次扣款就失敗（待處理），沒有其他方案 → 真流失
    """
    write(d / "Contacts.csv", ["系統編號", "年齡", "性別", "鄉鎮市區", "聯絡人建立日期"], [
        ["1", "20-24歲", "生理女", "A區", "2020-01-01 10:00:00"],
        ["2", "40-44歲", "生理男", "A區", "2020-01-01 10:00:00"],
        ["3", "15-19歲", "生理女", "東區", "2026-03-02 08:00:00"],
        ["4", "未填寫", "", "B區", "2020-01-01 10:00:00"],
        ["4", "未填寫", "", "B區", "2020-01-01 10:00:05"],
        ["5", "30-34歲", "生理女", "B區", "2026-01-05 08:00:00"],
    ])
    rec = [
        ["2", "生理男", "40-44歲", "", "A區", "8001", "2025-11-10 08:30:00", "1000", "50_24", "1", "信用卡", "", "102"],
        ["2", "生理男", "40-44歲", "", "A區", "8002", "2025-12-10 08:30:00", "1000", "50_25", "1", "信用卡", "", "102"],
        ["1", "生理女", "20-24歲", "", "A區", "8003", "2026-01-05 08:30:00", "500", "70_1", "1", "信用卡", "", "101"],
        ["1", "生理女", "20-24歲", "", "A區", "8004", "2026-02-12 08:30:00", "500", "70_2", "1", "信用卡", "", "101"],
        ["1", "生理女", "20-24歲", "", "A區", "8005", "2026-03-05 08:30:00", "500", "70_3", "1", "信用卡", "", "101"],
        ["2", "生理男", "40-44歲", "", "A區", "8006", "2026-01-25 08:30:00", "1500", "80_1", "1", "信用卡", "", "103"],
        ["2", "生理男", "40-44歲", "", "A區", "8007", "2026-02-25 08:30:00", "1500", "80_2", "1", "信用卡", "", "103"],
        ["2", "生理男", "40-44歲", "", "A區", "8008", "2026-03-25 08:30:00", "1500", "80_3", "1", "信用卡", "", "103"],
        ["2", "生理男", "40-44歲", "", "A區", "8009", "2026-04-25 08:30:00", "1500", "80_4", "1", "信用卡", "", "103"],
    ]
    write(d / "Recurring_Donations.csv", TX_HEADER, rec)
    write(d / "One_Time_Donations.csv", TX_HEADER, [
        ["3", "生理女", "15-19歲", "", "東區", "9001", "2026-03-02 09:00:00", "300", "9001", "7", "Apple Pay 幕前支付", "", ""],
        ["3", "生理女", "15-19歲", "", "東區", "9002", "2026-04-01 09:00:00", "800", "9002", "7", "Apple Pay 幕前支付", "", ""],
        ["1", "生理女", "20-24歲", "", "A區", "9003", "2026-03-20 09:00:00", "2000", "9003", "1", "信用卡", "", ""],
    ])
    write(d / "Recurring_Details.csv",
          ["系統編號", "約定金額", "金流機制", "剩餘期數", "執行期數 / 約定總期數", "開始日期", "結束日期", "失敗或取消日期",
           "定期定額狀態", "成功次數/ 請款次數", "收到金額總計", "最新 建立日期", "最近一筆收到日期", "最近失敗日期", "定期捐款編號"], [
        ["1", "500", "信用卡", "9", "3 / 12", "2026-01-05", "", "", "進行中", "3 / 4", "1500",
         "2026-03-05 08:30:00", "2026-03-05 08:30:00", "2026-02-05 08:30:00", "101"],
        ["2", "1000", "信用卡", "不限", "25 / 不限", "2024-01-10", "", "2026-01-10 08:30:00", "已失敗", "25 / 26", "25000",
         "2026-01-10 08:30:00", "2025-12-10 08:30:00", "2026-01-10 08:30:00", "102"],
        ["5", "300", "LINE / Apple Pay", "不限", "0 / 不限", "2026-01-05", "", "", "待處理", "0 / 1", "",
         "2026-01-05 08:30:00", "", "2026-01-05 08:30:00", "105"],
    ])
    write(d / "Event_Timeline.csv", ["日期", "活動或事件", "類型", "管道", "備註"], [
        ["每月第一週", "每月電子報", "電子報", "Email", "全年固定"],
        ["2026-03-01", "測試市集", "線下活動", "市集攤位", "活動至 3/10"],
    ])


class DefinitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        make_fixture(d)
        cls.data = yc.load(d, as_of="2026-04-30")
        cls.tx = yc.build_donations(cls.data)
        cls.gap = yc.charge_gap_p90(cls.tx)
        cls.plans_df = yc.build_plans(cls.data, cls.tx, cls.gap)
        cls.plans = cls.plans_df.set_index("定期捐款編號")
        cls.donors_df = yc.build_donors(cls.data, cls.tx, cls.plans_df)
        cls.donors = cls.donors_df.set_index("系統編號")
        cls.checks = yc.validate(cls.data, cls.plans_df).set_index("檢查項目")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_combined_columns_split(self):
        p = self.data.plans.set_index("定期捐款編號")
        self.assertEqual(p.loc["101", "執行期數"], 3)
        self.assertEqual(p.loc["101", "約定總期數"], 12)
        self.assertEqual(p.loc["101", "請款次數"], 4)
        self.assertTrue(p.loc["102", "不限期"])
        self.assertEqual(p.loc["102", "約定總期數"], 0)

    def test_join_row_count_and_dedupe(self):
        self.assertEqual(len(self.tx), 12)                 # 9 定期 + 3 單筆，join 不可膨脹
        self.assertEqual(len(self.data.contacts), 5)       # 重複的聯絡人只留一列
        self.assertEqual(self.checks.loc["系統編號 重複", "異常筆數"].max(), 1)

    def test_period_from_transaction_id(self):
        t = self.tx.set_index("捐款編號")
        self.assertEqual(t.loc["8004", "期數"], 2)
        self.assertEqual(t.loc["8002", "期數"], 25)

    def test_plan_master_recovers_unlisted_plan(self):
        self.assertEqual(len(self.plans), 4)
        self.assertFalse(self.plans.loc["103", "明細有列"])
        self.assertEqual(self.plans.loc["103", "狀態"], "推定進行中（明細未列）")
        self.assertEqual(self.checks.loc["② 有扣款、但 ④ 沒列出的方案（一人多方案只顯示一個）", "異常筆數"], 1)

    def test_card_switch_is_not_churn(self):
        self.assertTrue(self.plans.loc["102", "此人仍有其他方案在扣款"])
        self.assertTrue(self.plans.loc["102", "換方案(非流失)"])
        self.assertFalse(self.plans.loc["102", "真流失"])
        self.assertTrue(self.plans.loc["102", "之後有加碼方案"])
        self.assertEqual(self.plans.loc["102", "換方案後月額變化"], 500)   # 1000 → 1500

    def test_first_charge_failure_is_true_churn(self):
        self.assertTrue(self.plans.loc["105", "真流失"])
        self.assertTrue(self.plans.loc["105", "被動流失"])
        self.assertEqual(self.plans.loc["105", "流失類型"], "被動：第一次扣款就失敗")
        self.assertTrue(self.plans.loc["105", "期間新方案"])

    def test_failure_recovery(self):
        self.assertTrue(self.plans.loc["101", "失敗後已恢復"])      # 失敗 2/5，3/5 有扣款
        self.assertFalse(self.plans.loc["102", "失敗後已恢復"])

    def test_silence_days(self):
        self.assertEqual(self.plans.loc["101", "沉默天數"], (date(2026, 4, 30) - date(2026, 3, 5)).days)

    def test_new_donor_definition(self):
        self.assertTrue(self.donors.loc["3", "期間新捐款人"])       # 期間內建立聯絡人＋捐款
        self.assertFalse(self.donors.loc["1", "期間新捐款人"])      # 舊聯絡人
        self.assertFalse(self.donors.loc["5", "期間捐款人"])        # 只有失敗的方案，沒有成功捐款
        self.assertEqual(self.donors.loc["3", "首捐類型"], "單筆")

    def test_young_share_uses_filled_age(self):
        t1 = yc.t1_profile(self.donors_df).set_index("群組")
        self.assertAlmostEqual(t1.loc["期間捐款人", "25歲以下(占已填)"], round(2 / 3, 4))
        self.assertAlmostEqual(t1.loc["全部聯絡人", "年齡填寫率"], 0.8)    # 4/5 有填

    def test_stop_hazard_delayed_entry(self):
        sp = yc.t2_stop_period(self.plans_df).set_index("期數")
        self.assertEqual(sp.loc[0, "仍在觀察中的方案數"], 3)     # P1、P3、P5 從第 0 期開始；P2 從第 24 期才進入
        self.assertEqual(sp.loc[0, "於此期流失"], 1)             # P5
        self.assertEqual(sp.loc[25, "於此期換方案"], 1)           # P2

    def test_reminder_list(self):
        rl = yc.t2_reminder_list(self.plans_df).set_index("系統編號")
        self.assertNotIn("2", rl.index)                          # 換卡的人不打擾
        self.assertIn("付款失敗挽回", rl.loc["5", "建議行動"])
        self.assertEqual(len(rl), rl.index.nunique())

    def test_failure_outcome(self):
        fo = yc.t2_failure_outcome(self.plans_df, self.data.period_start).set_index("金流機制")
        self.assertEqual(fo.loc["合計", "同方案恢復"], 1)      # P1：2/5 失敗、3/5 恢復
        self.assertEqual(fo.loc["合計", "另開方案在扣"], 1)    # P2：失敗後另開 P3
        self.assertEqual(fo.loc["合計", "尚未恢復"], 1)        # P5：首期失敗

    def test_progress_ratio_and_preventive(self):
        pr = yc.t2_progress_ratio(self.plans_df)
        allr = pr[pr["約定期數分組"] == "全部"]
        self.assertEqual(int(allr["流失數"].sum()), 0)            # 有約定期數的 P1 沒流失
        self.assertEqual(int(allr["經過的期數(人期)"].sum()), 4)   # P1 第 0–3 期
        sp = yc.t2_stop_period(self.plans_df)
        pv = yc.t2_preventive_list(self.plans_df, sp, self.data.as_of)
        self.assertTrue(set(pv["系統編號"]) <= {"1", "2"})        # 只會是仍在扣款的人
        self.assertEqual(len(pv), pv["系統編號"].nunique())

    def test_term_cohort(self):
        old = yc.TERM_COHORT_MIN_AT_RISK
        yc.TERM_COHORT_MIN_AT_RISK = 1
        try:
            tc = yc.t2_term_cohort(self.plans_df, max_k=26).set_index("設定期數")
        finally:
            yc.TERM_COHORT_MIN_AT_RISK = old
        self.assertEqual(tc.loc["12期", "方案數"], 1)               # P1
        self.assertEqual(tc.loc["12期", "第3期"], 1.0)               # P1 扣到第 3 期仍在扣
        self.assertTrue(pd.isna(tc.loc["12期", "第4期"]))            # 之後觀察不到 ≠ 0
        self.assertEqual(tc.loc["不限期", "方案數"], 2)              # P2（換卡）、P5（首期失敗）
        self.assertEqual(tc.loc["不限期", "第0期"], 0.0)             # 第 0 期只有 P5 在觀察，且流失
        self.assertTrue(pd.isna(tc.loc["不限期", "期滿後續約率"]))

    def test_events_expanded(self):
        ev = self.data.events
        self.assertEqual((ev["事件類型"] == "電子報").sum(), 5)   # 2025-12 ~ 2026-04 每月 1 日
        m = ev[ev["事件名稱"] == "測試市集"].iloc[0]
        self.assertEqual(m["結束日期"].date(), date(2026, 3, 10))

    def test_cohort_unobserved_is_nan_not_zero(self):
        cr = yc.cohort_retention(self.tx, self.donors_df, "new").set_index("Cohort月份")
        self.assertEqual(cr.loc["2026-03", "M1"], 1.0)
        self.assertNotIn("M2", cr.columns)

    def test_stats_helpers(self):
        self.assertAlmostEqual(yc._gammainc_upper_reg(0.5, 3.841 / 2), 0.05, places=3)
        self.assertAlmostEqual(yc.poisson_upper_p(16, 10), 0.0487, places=3)


# ---------------------------------------------------------------------------
# B. 對帳：只用標準函式庫讀 CSV，獨立重算
# ---------------------------------------------------------------------------
def _dt(s: str):
    s = (s or "").strip()
    if not s:
        return None
    return datetime.fromisoformat(s[:19]) if len(s) > 10 else datetime.fromisoformat(s)


def _pair(s: str):
    a, b = [x.strip() for x in s.split("/", 1)]
    return a, b


def _quantile(xs, q):
    xs = sorted(xs)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


class ReconciliationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if DATA is None:
            raise unittest.SkipTest("未提供 --data")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        cls.summary = yc.run(DATA, cls.out)
        raw = yc.prepare_data(DATA, cls.out)

        def rd(name):
            with (raw / name).open(encoding="utf-8-sig", newline="") as f:
                return list(csv.DictReader(f))
        cls.c, cls.r, cls.o, cls.p = (rd(yc.FILES[k]) for k in ["contacts", "recurring", "one_time", "plans"])
        cls.as_of = max(_dt(x["捐款建立日期"]) for x in cls.r + cls.o).replace(hour=0, minute=0, second=0)
        cls.start = min(_dt(x["捐款建立日期"]) for x in cls.r + cls.o).replace(hour=0, minute=0, second=0)
        # 方案的最近一次扣款、獨立計算扣款間隔 P90
        by_plan = defaultdict(list)
        for x in cls.r:
            by_plan[x["定期捐款編號"]].append(_dt(x["捐款建立日期"]))
        gaps = []
        for ds in by_plan.values():
            ds.sort()
            gaps += [(b - a).days for a, b in zip(ds, ds[1:])]
        cls.gap = max(yc.SILENT_DAYS, math.ceil(_quantile(gaps, 0.9)))
        cls.last = {k: max(v) for k, v in by_plan.items()}
        listed = {x["定期捐款編號"] for x in cls.p}
        active_plans = {x["定期捐款編號"] for x in cls.p if x["定期定額狀態"] == "進行中"}
        active_plans |= {k for k, v in cls.last.items()
                         if k not in listed and (cls.as_of - v.replace(hour=0, minute=0, second=0)).days <= cls.gap}
        owner = {x["定期捐款編號"]: x["系統編號"] for x in cls.r}
        owner.update({x["定期捐款編號"]: x["系統編號"] for x in cls.p})
        cls.owner = owner
        cls.active_plans = active_plans

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def out_csv(self, name):
        with (self.out / f"{name}.csv").open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    def test_donor_count(self):
        ids = {x["系統編號"] for x in self.r} | {x["系統編號"] for x in self.o}
        self.assertEqual(self.summary["總覽"]["期間捐款人數"], len(ids))

    def test_total_amount(self):
        total = sum(int(float(x["總金額"])) for x in self.r + self.o)
        self.assertEqual(self.summary["總覽"]["捐款總額"], total)

    def test_plan_master_size(self):
        ids = {x["定期捐款編號"] for x in self.r} | {x["定期捐款編號"] for x in self.p}
        self.assertEqual(self.summary["總覽"]["定期方案數(還原後)"], len(ids))
        self.assertEqual(self.summary["總覽"]["其中④有列"], len(self.p))

    def test_failed_and_recovered(self):
        failed = recovered = 0
        for x in self.p:
            ok, req = (int(v) for v in _pair(x["成功次數/ 請款次數"]))
            if req > ok:
                failed += 1
                last_ok = max([d for d in [_dt(x["最近一筆收到日期"]), self.last.get(x["定期捐款編號"])] if d],
                              default=None)
                lf = _dt(x["最近失敗日期"])
                if last_ok and lf and last_ok > lf:
                    recovered += 1
        self.assertEqual(self.summary["總覽"]["扣款失敗方案數"], failed)
        self.assertEqual(self.summary["總覽"]["失敗後已恢復方案數"], recovered)

    def test_stale_detail_rows(self):
        """④ 顯示已失敗/待處理，但此人另有方案在扣款。"""
        n = 0
        for x in self.p:
            if x["定期定額狀態"] in ("已失敗", "待處理"):
                others = {k for k in self.active_plans if self.owner[k] == x["系統編號"] and k != x["定期捐款編號"]}
                n += bool(others)
        self.assertEqual(self.summary["總覽"]["④顯示失敗或待處理但此人仍在扣款"], n)

    def test_reminder_list_excludes_active_donors(self):
        rows = self.out_csv("t2_優先提醒名單")
        self.assertEqual(len(rows), len({x["系統編號"] for x in rows}))
        active_people = {self.owner[k] for k in self.active_plans}
        for x in rows:
            if x["狀態"] in yc.STOPPED_STATUSES or x["狀態"] == yc.COMPLETED_STATUS:
                self.assertNotIn(x["系統編號"], active_people, x)

    def test_young_share(self):
        seen, filled, young = set(), 0, 0
        for x in self.c:
            if x["系統編號"] in seen:
                continue
            seen.add(x["系統編號"])
            if x["年齡"] and x["年齡"] not in yc.AGE_MISSING:
                filled += 1
                young += x["年齡"] in yc.YOUNG_AGE_BANDS
        self.assertAlmostEqual(self.summary["總覽"]["25歲以下占聯絡人(已填)"], round(young / filled, 4))

    def test_new_donor_count(self):
        created = {}
        for x in self.c:
            d = _dt(x["聯絡人建立日期"])
            created[x["系統編號"]] = min(created.get(x["系統編號"], d), d)
        ids = {x["系統編號"] for x in self.r} | {x["系統編號"] for x in self.o}
        n = sum(1 for i in ids if created[i] >= self.start)
        self.assertEqual(self.summary["總覽"]["期間新捐款人數"], n)

    def test_monthly_churn_december(self):
        last = {}
        for x in self.r:
            last[x["系統編號"]] = max(last.get(x["系統編號"], ""), x["捐款建立日期"][:7])
        dec = sum(1 for v in last.values() if v == "2025-12")
        row = [x for x in self.out_csv("t2_月流失率") if x["月份"] == "2025-12"][0]
        self.assertEqual(int(float(row["之後不再扣款"])), dec)

    def test_matches_official_data_page(self):
        """對照主辦方的資料說明頁（datadotgo.github.io/connect-the-dots-web/yewan_data.html）公布的數字。"""
        if len(self.r) != 24101:
            self.skipTest("不是官方 v1 資料")
        page_monthly = {"2025-11": 1610, "2025-12": 1735, "2026-01": 1789, "2026-02": 1856, "2026-03": 1935,
                        "2026-04": 2013, "2026-05": 2085, "2026-06": 2146, "2026-07": 2173, "2026-08": 2222,
                        "2026-09": 2251, "2026-10": 2250}
        got = {x["月份"]: int(x["當月成功扣款人數"]) for x in self.out_csv("t2_月流失率")}
        self.assertEqual(got, page_monthly)
        self.assertEqual(len({x["系統編號"] for x in self.r}), 2903)
        self.assertEqual(len({x["定期捐款編號"] for x in self.r}), 2958)
        self.assertEqual(len({x["系統編號"] for x in self.o}), 4169)
        self.assertEqual(sum(1 for x in self.o if float(x["總金額"]) >= 30000), 9)
        status = defaultdict(int)
        for x in self.p:
            status[x["定期定額狀態"]] += 1
        self.assertEqual(dict(status), {"進行中": 1671, "已完成": 367, "已失敗": 352, "待處理": 258,
                                        "暫停中": 245, "已取消": 10})
        self.assertEqual(sum(1 for x in self.p if not x["收到金額總計"]), 523)

    def test_stop_hazard_period0(self):
        new = set()
        for x in self.r:
            if x["交易編號"].endswith("_1"):
                new.add(x["定期捐款編號"])
        for x in self.p:
            if _dt(x["開始日期"]) >= self.start:
                new.add(x["定期捐款編號"])
        row = self.out_csv("t2_停止期數")[0]
        self.assertEqual(row["期數"], "0")
        self.assertEqual(int(row["仍在觀察中的方案數"]), len(new))


if __name__ == "__main__":
    for flag in ("--data", "--data-dir"):
        if flag in sys.argv:
            i = sys.argv.index(flag)
            DATA = Path(sys.argv[i + 1])
            del sys.argv[i:i + 2]
    unittest.main(verbosity=2)
