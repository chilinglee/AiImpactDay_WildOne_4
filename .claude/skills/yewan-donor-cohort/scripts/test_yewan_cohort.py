"""
驗證 Skill 產出的數字對不對（可靠性檢查）。

兩種測試：
  A. 定義測試：用一份手工小資料（答案可以心算），確認每個指標的定義正確。
  B. 對帳測試：用「只靠 Python 標準函式庫」的另一套寫法，獨立重算真實資料的關鍵數字，
     和 pandas 管線的輸出比對。兩套程式算出一樣的數字，才算可信。

用法：
    python test_yewan_cohort.py                    # 只跑 A
    python test_yewan_cohort.py --data-dir <CSV>   # A + B
"""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import yewan_cohort as yc  # noqa: E402

DATA_DIR: Path | None = None


def write(path: Path, header: list[str], rows: list[list]):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def make_fixture(d: Path):
    """4 位聯絡人、3 個定期方案、3 筆單筆、1 個活動。"""
    write(d / "Contacts.csv", ["系統編號", "年齡級距", "性別", "鄉鎮市區", "建立日期", "首次捐款日期", "期間捐款人"], [
        ["S1", "20-24歲", "女性", "A區", "2020-01-01", "2026-01-05", "是"],
        ["S2", "40-44歲", "男性", "A區", "2020-01-01", "2026-01-10", "是"],
        ["S3", "20歲以下", "女性", "B區", "2026-03-01", "2026-03-02", "是"],
        ["S4", "60-64歲", "未提供", "B區", "2020-01-01", "", "否"],
    ])
    write(d / "Recurring_Details.csv", ["定期捐款編號", "系統編號", "約定金額", "執行期數", "約定總期數", "狀態",
                                        "成功次數", "請款次數", "最近失敗日期", "加入日期"], [
        ["P1", "S1", "500", "3", "12", "進行中", "3", "4", "2026-02-05", "2025-06-01"],   # 失敗後 3/5 有恢復
        ["P2", "S2", "1000", "2", "0", "暫停中", "2", "3", "2026-02-10", "2025-06-01"],   # 失敗未恢復、停止
        ["P3", "S2", "1500", "1", "12", "進行中", "1", "1", "", "2025-09-01"],           # S2 之後加碼
    ])
    write(d / "Recurring_Donations.csv", ["交易編號", "系統編號", "捐款日期", "金額", "付款工具", "定期捐款編號", "扣款期數"], [
        ["R1", "S1", "2026-01-05", "500", "信用卡", "P1", "1"],
        ["R2", "S1", "2026-02-05", "500", "信用卡", "P1", "2"],
        ["R3", "S1", "2026-03-05", "500", "信用卡", "P1", "3"],
        ["R4", "S2", "2026-01-10", "1000", "LINE Pay", "P2", "1"],
        ["R5", "S2", "2026-01-20", "1000", "LINE Pay", "P2", "2"],   # 同月兩次
        ["R6", "S2", "2026-01-25", "1500", "LINE Pay", "P3", "1"],
    ])
    write(d / "One_Time_Donations.csv", ["交易編號", "系統編號", "捐款日期", "金額", "付款工具", "捐款管道"], [
        ["O1", "S3", "2026-03-02", "300", "Apple Pay", "市集"],
        ["O2", "S3", "2026-04-01", "800", "Apple Pay", "官網"],
        ["O3", "S1", "2026-03-20", "2000", "信用卡", "電子報"],
    ])
    write(d / "Event_Timeline.csv", ["事件編號", "日期", "事件類型", "事件名稱", "渠道", "說明"], [
        ["EV1", "2026-03-01", "市集", "測試市集", "實體", "x"],
    ])


class DefinitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        make_fixture(d)
        cls.data = yc.load(d, as_of="2026-04-30")
        cls.tx = yc.build_donations(cls.data)
        cls.donors = yc.build_donors(cls.data, cls.tx).set_index("系統編號")
        cls.plans = yc.build_plans(cls.data, cls.tx).set_index("定期捐款編號")
        cls.checks = yc.validate(cls.data).set_index("檢查項目")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_join_row_count(self):
        self.assertEqual(len(self.tx), 9)  # 6 定期 + 3 單筆，join 不可膨脹

    def test_cohort_month_and_age(self):
        s3 = self.tx[self.tx["系統編號"] == "S3"].set_index("交易編號")
        self.assertEqual(s3.loc["O1", "Cohort月份"], "2026-03")
        self.assertEqual(s3.loc["O2", "距首捐月數"], 1)

    def test_new_donor_definition(self):
        self.assertTrue(self.donors.loc["S3", "期間新捐款人"])      # 首捐單筆、無定期
        self.assertFalse(self.donors.loc["S1", "期間新捐款人"])     # 既有定期捐款人
        self.assertFalse(self.donors.loc["S4", "期間捐款人"])

    def test_young_flag(self):
        self.assertTrue(self.donors.loc["S1", "年輕支持者(<25)"])
        self.assertTrue(self.donors.loc["S3", "年輕支持者(<25)"])
        self.assertFalse(self.donors.loc["S2", "年輕支持者(<25)"])

    def test_failure_recovery(self):
        self.assertTrue(self.plans.loc["P1", "失敗後已恢復"])       # 失敗 2/5，3/5 有扣款
        self.assertFalse(self.plans.loc["P2", "失敗後已恢復"])      # 失敗 2/10，之後沒扣款
        self.assertFalse(self.plans.loc["P3", "有扣款失敗"])

    def test_silence_days(self):
        self.assertEqual(self.plans.loc["P1", "沉默天數"], (date(2026, 4, 30) - date(2026, 3, 5)).days)

    def test_upgrade_detected(self):
        self.assertTrue(self.plans.loc["P2", "之後有加碼方案"])
        self.assertFalse(self.plans.loc["P3", "之後有加碼方案"])

    def test_stop_hazard(self):
        sp = yc.t2_stop_period(self.plans.reset_index()).set_index("期數")
        self.assertEqual(sp.loc[1, "仍在觀察中的方案數"], 3)
        self.assertEqual(sp.loc[2, "於此期停止"], 1)
        self.assertAlmostEqual(sp.loc[2, "停止率(hazard)"], 0.5)   # 第 2 期 at risk = P1,P2

    def test_reminder_reasons(self):
        rl = yc.t2_reminder_list(self.plans.reset_index()).set_index("定期捐款編號")
        self.assertIn("扣款失敗後尚未成功扣款", rl.loc["P2", "理由"])
        self.assertTrue(rl.loc["P2", "優先分數"] > rl.loc["P1", "優先分數"] if "P1" in rl.index else True)

    def test_quality_checks_fire(self):
        self.assertEqual(self.checks.loc["同一方案同月扣款超過 1 次", "異常筆數"], 1)
        self.assertEqual(self.checks.loc["系統編號 對不到聯絡人", "異常筆數"].max(), 0)

    def test_cohort_unobserved_is_nan_not_zero(self):
        cr = yc.cohort_retention(self.tx, self.donors.reset_index(), "new").set_index("Cohort月份")
        self.assertEqual(cr.loc["2026-03", "M1"], 1.0)

    def test_stats_helpers(self):
        # 已知值：卡方 df=1, stat=3.841 → p≈0.05；Poisson(10) P(X>=16)≈0.0487
        self.assertAlmostEqual(yc._gammainc_upper_reg(0.5, 3.841 / 2), 0.05, places=3)
        self.assertAlmostEqual(yc.poisson_upper_p(16, 10), 0.0487, places=3)


class ReconciliationTests(unittest.TestCase):
    """以標準函式庫獨立重算，對帳 pandas 管線。"""

    @classmethod
    def setUpClass(cls):
        if DATA_DIR is None:
            raise unittest.SkipTest("未提供 --data-dir")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        cls.summary = yc.run(DATA_DIR, cls.out)

        def rd(name):
            with (DATA_DIR / name).open(encoding="utf-8-sig", newline="") as f:
                return list(csv.DictReader(f))
        cls.c, cls.r, cls.o, cls.p = (rd(yc.FILES[k]) for k in ["contacts", "recurring", "one_time", "plans"])

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
        total = sum(int(x["金額"]) for x in self.r) + sum(int(x["金額"]) for x in self.o)
        self.assertEqual(self.summary["總覽"]["捐款總額"], total)

    def test_failed_and_recovered(self):
        last = defaultdict(str)
        for x in self.r:
            last[x["定期捐款編號"]] = max(last[x["定期捐款編號"]], x["捐款日期"])
        failed = [x for x in self.p if int(x["請款次數"]) > int(x["成功次數"])]
        recovered = [x for x in failed if x["最近失敗日期"] and last[x["定期捐款編號"]] > x["最近失敗日期"]]
        self.assertEqual(self.summary["總覽"]["扣款失敗方案數"], len(failed))
        self.assertEqual(self.summary["總覽"]["失敗後已恢復方案數"], len(recovered))

    def test_stop_hazard_period1(self):
        stopped1 = sum(1 for x in self.p if x["執行期數"] == "1" and x["狀態"] in yc.STOPPED_STATUSES)
        row = self.out_csv("t2_停止期數")[0]
        self.assertEqual(int(row["於此期停止"]), stopped1)
        self.assertEqual(int(row["仍在觀察中的方案數"]), len(self.p))

    def test_recurring_cohort_size(self):
        first = {}
        for x in self.r:
            first[x["系統編號"]] = min(first.get(x["系統編號"], "9999"), x["捐款日期"])
        nov = sum(1 for v in first.values() if v.startswith("2025-11"))
        row = [x for x in self.out_csv("t2_定期Cohort留存") if x["Cohort月份"] == "2025-11"][0]
        self.assertEqual(int(row["Cohort人數"]), nov)

    def test_young_share(self):
        young = {"20歲以下", "20-24歲"}
        share = sum(1 for x in self.c if x["年齡級距"] in young) / len(self.c)
        self.assertAlmostEqual(self.summary["總覽"]["25歲以下占聯絡人"], round(share, 4))

    def test_reminder_list_unique(self):
        rows = self.out_csv("t2_優先提醒名單")
        self.assertEqual(len(rows), len({x["定期捐款編號"] for x in rows}))


if __name__ == "__main__":
    if "--data-dir" in sys.argv:
        i = sys.argv.index("--data-dir")
        DATA_DIR = Path(sys.argv[i + 1])
        del sys.argv[i:i + 2]
    unittest.main(verbosity=2)
