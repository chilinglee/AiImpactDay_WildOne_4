"""把分析結果做成一封「給公益勸募科的月報信」：每個問題一句答案＋一張圖＋行銷名單（附件）。

用法（先跑過 yewan_cohort.py）：
    python email_digest.py --output-dir yewan_output

產出 <output-dir>/email/：
    c1_*.png … c6_*.png          內嵌圖表（依 manifest 順序 = cid:att0, att1, …）。檔名必須是英數字：
                                  Gmail 草稿工具遇到中文檔名的內嵌圖片會整張丟掉
    名單_*.csv                    行銷名單附件（UTF-8 BOM，Excel / Google Sheets 直接開）
    email.html                    信件 HTML（Gmail 相容：全部 inline style、圖用 cid）
    manifest.json                 主旨、附件順序與 MIME 類型（給 Gmail 草稿工具用）

答案文字全部由程式依『判定』欄位產生：不顯著的差異只會寫「資料不支持」，不會寫成結論。
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# 參考調色盤（dataviz skill，淺色）：系列色依固定順序，文字用文字色，不用系列色
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
LIGHT = "#b7d3f6"     # S1 的淺階：非重點的長條
CJK_FONTS = ["Microsoft JhengHei", "Noto Sans CJK TC", "Noto Sans TC", "PingFang TC", "Heiti TC",
             "WenQuanYi Zen Hei", "Arial Unicode MS"]
E = html.escape


def _setup_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    picked = [f for f in CJK_FONTS if f in have]
    if not picked:
        print("⚠ 找不到中文字型，圖中的中文會變方塊。Colab 可先執行：!apt -y install fonts-noto-cjk")
    plt.rcParams.update({
        "font.sans-serif": picked + ["DejaVu Sans"], "font.family": "sans-serif", "axes.unicode_minus": False,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "axes.titlelocation": "left", "font.size": 11, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
    })


def _pct(v, d=1):
    return "—" if pd.isna(v) else f"{v * 100:.{d}f}%"


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _read(out: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(out / f"{name}.csv", encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# 圖
# ---------------------------------------------------------------------------
def chart_stop_period(sp, risky, pooled, path):
    s = sp[sp["期數"] <= 24]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    colors = [S1 if k in risky else LIGHT for k in s["期數"]]
    ax.bar(s["期數"], s["停止率(hazard)"], color=colors, width=0.75, edgecolor=SURFACE, linewidth=1)
    err = [s["停止率(hazard)"] - s["停止率95%下界"], s["停止率95%上界"] - s["停止率(hazard)"]]
    ax.errorbar(s["期數"], s["停止率(hazard)"], yerr=err, fmt="none", ecolor=MUTED, elinewidth=1, capsize=0)
    ax.axhline(pooled, color=INK2, linestyle=(0, (4, 3)), linewidth=1)
    ax.text(-0.6, pooled, f"平均 {_pct(pooled)}", va="bottom", ha="left", color=INK2, fontsize=10)
    si = s.set_index("期數")
    for k in risky:
        ax.text(k, si.loc[k, "停止率95%上界"] + 0.002, _pct(si.loc[k, "停止率(hazard)"]), ha="center", va="bottom",
                color=INK, fontsize=9)
    ax.set_title("第幾期最容易停下來？（深藍＝顯著高於平均）")
    ax.set_xlabel("期數（第 0 期＝第一次扣款就沒成功）")
    ax.set_ylabel("該期停止率")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_xticks(range(0, 25, 2))
    ax.set_ylim(0, max(s["停止率95%上界"].max(), pooled) * 1.25)
    _save(fig, path)


def chart_progress(ratio, renewal, path):
    a = ratio[ratio["約定期數分組"] == "全部"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [2.3, 1]})
    top = a["每期流失率"].idxmax()
    colors = [S1 if i == top else LIGHT for i in a.index]
    labels = [str(x).replace("（", "\n（") for x in a["進度"]]
    ax.bar(labels, a["每期流失率"], color=colors, width=0.7, edgecolor=SURFACE, linewidth=1)
    ax.errorbar(range(len(a)), a["每期流失率"], yerr=[a["每期流失率"] - a["95%下界"], a["95%上界"] - a["每期流失率"]],
                fmt="none", ecolor=MUTED, elinewidth=1)
    for i, (v, up) in enumerate(zip(a["每期流失率"], a["95%上界"])):
        ax.text(i, up + 0.001, _pct(v), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_title("扣到約定的幾成時最容易停？")
    ax.set_ylabel("每經過一期的流失率")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=1))
    ax.tick_params(axis="x", labelsize=9)
    n, kept = renewal["期滿方案數"], renewal["期滿後仍在扣款(續約/換方案)"]
    ax2.bar(["期滿後\n續約", "期滿後\n沒有續約"], [kept, n - kept], color=[S1, S2], width=0.6)
    for i, v in enumerate([kept, n - kept]):
        ax2.text(i, v + n * 0.02, f"{v} 個\n({_pct(v / n if n else 0, 0)})", ha="center", va="bottom", fontsize=9)
    ax2.set_title("約定期數扣滿之後")
    ax2.set_ylim(0, n * 1.3 if n else 1)
    ax2.grid(False)
    _save(fig, path)


def chart_term_cohort(tc, path):
    """設定期數 × 第幾期 留存熱圖。空白＝超過約定期數（方案已結束）或觀察數不足。"""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    cols = [c for c in tc.columns if c.startswith("第") and c.endswith("期")]
    m = tc[cols].to_numpy(dtype=float)
    ks = [int(c[1:-1]) for c in cols]
    cmap = LinearSegmentedColormap.from_list("seq", ["#e8f1fc", "#9ec5f4", "#3987e5", "#1c4f8f"])
    cmap.set_bad("#f4f3f0")
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(tc) + 1.6))
    vmin = np.floor(np.nanmin(m) * 10) / 10
    im = ax.imshow(np.ma.masked_invalid(m), aspect="auto", cmap=cmap, vmin=vmin, vmax=1)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=6.5,
                        color="#ffffff" if v > vmin + (1 - vmin) * 0.55 else INK)
    labels = [f"{r['設定期數']}（{int(r['方案數'])}）" for _, r in tc.iterrows()]
    ax.set_yticks(range(len(tc)), labels)
    ax.set_xticks(range(0, len(ks), 3), [str(k) for k in ks[::3]])
    ax.set_xlabel("第幾期（第 0 期＝第一次扣款）")
    ax.set_ylabel("捐款人設定的約定期數（方案數）")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i, (_, r) in enumerate(tc.iterrows()):
        txt = "—" if pd.isna(r["期滿後續約率"]) else _pct(r["期滿後續約率"], 0)
        ax.text(len(ks) - 0.3, i, f"  期滿續約 {txt}", va="center", ha="left", fontsize=9, color=INK2)
    ax.set_title("設定幾期的人，各期還有幾 % 在扣款？（數字＝留存 %；灰＝方案已期滿或樣本不足）")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.13)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    cb.outline.set_visible(False)
    _save(fig, path)


def chart_failure(fo, path):
    t = fo[fo["金流機制"] == "合計"].iloc[0]
    parts = [("同方案恢復", S1), ("另開方案在扣", S3), ("尚未恢復", S2)]
    fig, ax = plt.subplots(figsize=(9, 1.9))
    left = 0
    for name, col in parts:
        v = t[name]
        ax.barh([0], [v], left=left, color=col, edgecolor=SURFACE, linewidth=2, height=0.55)
        ax.text(left + v / 2, 0, f"{name}\n{int(v)}（{_pct(t[name + '比例'], 0)}）", ha="center", va="center",
                color="#ffffff" if col != S3 else INK, fontsize=10)
        left += v
    ax.set_xlim(0, left)
    ax.axis("off")
    ax.set_title(f"扣款失敗之後：期間內 {int(t['期間內有失敗的方案數'])} 個方案曾失敗，後來怎麼了？")
    _save(fig, path)


def chart_acquisition(ets, path):
    e = ets[ets["事件類型"] != "無對應活動"].dropna(subset=["新捐款人_增幅倍數"]) \
        .sort_values("新捐款人_增幅倍數", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    y = range(len(e))
    h = 0.36
    ax.barh([i + h / 2 for i in y], e["新捐款人_增幅倍數"], height=h, color=S1, label="新捐款人")
    ax.barh([i - h / 2 for i in y], e["新定期方案_增幅倍數"], height=h, color=S2, label="新定期方案")
    for i, (_, r) in enumerate(e.iterrows()):
        for col, dy in [("新捐款人", h / 2), ("新定期方案", -h / 2)]:
            k, n = int(r[col + "_顯著活動數"]), int(r[col + "_活動數"])
            tag = "" if k == 0 else (f"  ★ {k}/{n} 次顯著" if n > 1 else "  ★顯著")
            ax.text(r[col + "_增幅倍數"] + 0.05, i + dy, f"{r[col + '_增幅倍數']:.1f}×{tag}", va="center", fontsize=9)
    ax.axvline(1, color=INK2, linestyle=(0, (4, 3)), linewidth=1)
    ax.set_yticks(list(y), e["事件類型"])
    ax.set_xlabel("活動期間人數 ÷ 活動前的日常水準（1× ＝沒有增加）")
    ax.set_title("哪類活動帶來新捐款人？帶來新定期嗎？（★＝通過校正後檢定）")
    ax.legend(loc="lower right")
    ax.set_xlim(0, e[["新捐款人_增幅倍數", "新定期方案_增幅倍數"]].max().max() * 1.3)
    _save(fig, path)


def chart_retention(ets, path):
    e = ets.dropna(subset=["單筆首捐新捐款人"]).sort_values("轉定期率", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, col, title, colr in [(axes[0], "轉定期率", "之後轉成定期定額", S1),
                                 (axes[1], "90天再捐率", "90 天內再捐一次", S2)]:
        v = e[col]
        lo = (v - e[col + "95%下界"]).clip(lower=0)
        hi = (e[col + "95%上界"] - v).clip(lower=0)
        ax.barh(e["事件類型"], v, color=colr, height=0.6)
        ax.errorbar(v, e["事件類型"], xerr=[lo, hi], fmt="none", ecolor=MUTED, elinewidth=1)
        short = e["可觀察滿90天人數"].fillna(0).eq(0).to_numpy()
        for i, (x, up) in enumerate(zip(v, e[col + "95%上界"])):
            lab = "觀察不足 90 天" if (pd.isna(x) or short[i]) else _pct(x)
            ax.text((0 if pd.isna(up) else up) + 0.005, i, lab, va="center", fontsize=9, color=INK)
        ax.set_title(title)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
        ax.invert_yaxis() if ax is axes[0] else None
    fig.suptitle("從哪類活動進來的單筆新朋友，後來留得住？", x=0.01, ha="left", fontsize=14, fontweight="bold")
    _save(fig, path)


def chart_lists(prev, revive, path):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    p = prev.copy()
    p["週"] = pd.to_datetime(p["建議寄送日"]).dt.to_period("W-SUN").dt.end_time.dt.strftime("%m/%d 前")
    tab = pd.crosstab(p["週"], p["建議行動"].str.split("：").str[0])
    bottom = None
    for i, c in enumerate(tab.columns):
        ax.bar(tab.index, tab[c], bottom=bottom, color=[S1, S2, S3][i % 3], label=c, edgecolor=SURFACE, linewidth=1)
        bottom = tab[c] if bottom is None else bottom + tab[c]
    ax.set_title("預防性提醒：每週要寄幾封（扣款日前 7 天）")
    ax.legend(fontsize=9, loc="upper right")
    ax.tick_params(axis="x", labelsize=9)
    r = revive["建議行動"].str.split("：").str[0].value_counts().sort_values()
    ax2.barh(r.index, r.values, color=S1, height=0.6)
    for i, v in enumerate(r.values):
        ax2.text(v + 2, i, str(v), va="center", fontsize=9)
    ax2.set_title("起死回生：各要做什麼")
    ax2.set_xlim(0, r.max() * 1.2)
    _save(fig, path)


# ---------------------------------------------------------------------------
# 文字（依判定產生）
# ---------------------------------------------------------------------------
def _cell(v) -> str:
    if pd.isna(v):
        return ""
    if isinstance(v, float) and v.is_integer():
        return f"{int(v):,}"
    return str(v)


def _table(df: pd.DataFrame, cols: list[str], n: int = 10) -> str:
    th = "".join(f'<th style="text-align:left;padding:6px 8px;border-bottom:2px solid {GRID};color:{INK2};'
                 f'font-weight:600;white-space:nowrap">{E(c)}</th>' for c in cols)
    trs = []
    for _, r in df.head(n).iterrows():
        tds = "".join(f'<td style="padding:6px 8px;border-bottom:1px solid {GRID};vertical-align:top">'
                      f'{E(_cell(r[c]))}</td>' for c in cols)
        trs.append(f"<tr>{tds}</tr>")
    return (f'<table style="border-collapse:collapse;font-size:13px;width:100%;margin:8px 0">'
            f'<thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table>')


def build(out: Path) -> dict:
    _setup_font()
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    em = out / "email"
    em.mkdir(exist_ok=True)
    sp, ratio, fo = _read(out, "t2_停止期數"), _read(out, "t2_進度比例停止"), _read(out, "t2_失敗後結果")
    ets, prev, revive = _read(out, "t3_活動類型成效"), _read(out, "t2_預防性提醒名單"), _read(out, "t2_起死回生名單")
    tc = _read(out, "t2_設定期數Cohort留存")
    lift = _read(out, "t3_活動前後")
    risky, pooled = s["預防性提醒名單"]["高風險期數"], s["預防性提醒名單"]["平均停止率"]
    tests = s["檢定結果"]
    renewal = s["期滿續約"]

    charts = [("c1_stop_period.png", lambda p: chart_stop_period(sp, risky, pooled, p)),
              ("c2_term_progress.png", lambda p: chart_progress(ratio, renewal, p)),
              ("c3_after_failure.png", lambda p: chart_failure(fo, p)),
              ("c4_event_acquisition.png", lambda p: chart_acquisition(ets, p)),
              ("c5_event_retention.png", lambda p: chart_retention(ets, p)),
              ("c6_list_schedule.png", lambda p: chart_lists(prev, revive, p)),
              ("c7_term_cohort.png", lambda p: chart_term_cohort(tc, p))]
    for name, fn in charts:
        fn(em / name)

    # 名單附件
    lists = {"名單_預防性提醒.csv": prev, "名單_起死回生.csv": revive}
    for name, df in lists.items():
        df.to_csv(em / name, index=False, encoding="utf-8-sig")

    # ---- 答案句
    hz = sp.set_index("期數")["停止率(hazard)"]
    if tests["t2_停止期數"]["判定"] == "差異顯著" and risky:
        a1 = ("最容易停在 " + "、".join(f"第 {k} 期（{_pct(hz[k])}）" for k in risky)
              + f"，都顯著高於平均的 {_pct(pooled)}。"
              + ("前兩期是剛開始的不穩定期；" if {1, 2} & set(risky) else "")
              + ("第 12、24 期正好是常見的約定到期點。" if {12, 24} & set(risky) else ""))
    else:
        a1 = "各期停止率沒有顯著差異，資料不支持「某一期特別危險」。"
    a = ratio[ratio["約定期數分組"] == "全部"].set_index("進度")
    top = a["每期流失率"].idxmax()
    n_end, kept = renewal["期滿方案數"], renewal["期滿後仍在扣款(續約/換方案)"]
    a2 = (f"有約定期數的方案，在扣到約定的 <b>{E(str(top))}</b> 時每期流失率最高（{_pct(a.loc[top, '每期流失率'])}），"
          f"之後明顯下降。" if tests["t2_進度比例停止"]["判定"] == "差異顯著"
          else "約定進度和流失沒有顯著關係。") + \
         (f"更大的流失在<b>期滿之後</b>：{n_end} 個扣滿約定期數的方案，只有 {kept} 個（{_pct(kept / n_end if n_end else 0)}）"
          f"之後還在扣款。" if n_end else "")
    t = fo[fo["金流機制"] == "合計"].iloc[0]
    a3 = (f"期間內 {int(t['期間內有失敗的方案數'])} 個方案曾扣款失敗：{_pct(t['同方案恢復比例'], 0)} 同一方案就恢復了，"
          f"{_pct(t['另開方案在扣比例'], 0)} 另開新方案（換卡），只有 {_pct(t['尚未恢復比例'], 0)}（{int(t['尚未恢復'])} 個）"
          f"沒有回來——這些人在「起死回生名單」。")
    fail_t = tests.get("t2_預防性提醒名單", {})
    a3 += (" 「曾經扣款失敗」本身<b>不會</b>讓人比較容易流失（"
           f"p = {fail_t.get('p值')}，{E(fail_t.get('判定', ''))}），所以預防性提醒不用它當理由。"
           if fail_t.get("判定") != "差異顯著" else " 曾經扣款失敗的人流失率顯著較高，已列入預防性提醒理由。")
    sig_new = ets[(ets["新捐款人_顯著活動數"] > 0)].sort_values("新捐款人_增幅倍數", ascending=False)
    sig_rec = lift[(lift["指標"] == "新定期方案") & (lift["判定"] == "顯著增加")]
    sig_ev = lift[(lift["指標"] == "新捐款人") & (lift["判定"] == "顯著增加")].sort_values("增幅倍數", ascending=False)
    a4 = ("顯著帶來新捐款人的活動：" + "、".join(f"{r['事件名稱']}（{r['事件類型']}，{r['增幅倍數']:.1f} 倍）"
                                       for _, r in sig_ev.iterrows()) + "。"
          if len(sig_ev) else "沒有任何活動顯著帶來新捐款人。")
    if len(sig_new):
        a4 += f"以類型來看，{sig_new.iloc[0]['事件類型']}的整體增幅最高（{sig_new.iloc[0]['新捐款人_增幅倍數']:.1f} 倍）。"
    a4 += ("但新定期方案幾乎不受活動影響：只有「" + "、".join(sig_rec["事件名稱"]) + "」顯著增加。"
           if len(sig_rec) else "但沒有任何活動顯著帶來新的定期方案。")
    seg = s["分群檢定"]
    conv_v = seg.get("轉為定期率（單筆首捐）｜首捐時的活動類型", {}).get("判定", "")
    r90_v = seg.get("90天內再捐率（單筆首捐）｜首捐時的活動類型", {}).get("判定", "")
    e = ets.dropna(subset=["轉定期率"])
    e = e[(e["單筆首捐新捐款人"] >= 30) & (e["可觀察滿90天人數"] > 0)].sort_values("轉定期率", ascending=False)
    low = e[e["事件類型"] != "無對應活動"].tail(2)
    a5 = (f"從「{e.iloc[0]['事件類型']}」進來的單筆新朋友最常轉定期（{_pct(e.iloc[0]['轉定期率'])}），"
          f"「{e.iloc[1]['事件類型']}」次之（{_pct(e.iloc[1]['轉定期率'])}）；"
          + "、".join(f"{r['事件類型']}（{_pct(r['轉定期率'])}）" for _, r in low.iterrows()) + "進來的人很少轉定期。"
          if conv_v == "差異顯著" and len(e) >= 2 else "各類活動進來的人轉定期率沒有顯著差異。")
    a5 += (" 90 天內再捐率各類活動<b>沒有顯著差異</b>（資料不支持「哪類活動的人比較會再捐」）。"
           if r90_v != "差異顯著" else " 90 天內再捐率各類活動有顯著差異。")
    a5 += " ⚠ 9 月媒體報導進來的人觀察不到 90 天，轉定期率會被低估。"
    t = tc.set_index("設定期數")
    a2b = ""
    if "第2期" in t and t["第2期"].notna().any():
        a2b = f"不論設定幾期，扣到第 2 期時都只剩 {_pct(t['第2期'].min(), 0)}–{_pct(t['第2期'].max(), 0)}。"
    for g in t.sort_values("方案數", ascending=False).index:   # 用方案數最多的固定期數當例子
        term = int(g[:-1]) if g[:-1].isdigit() else None
        col = f"第{term}期" if term else None
        if term and col in t and pd.notna(t.loc[g, col]):
            a2b += (f"設定 {term} 期的人扣到第 {term} 期還有 {_pct(t.loc[g, col], 0)}，"
                    f"但期滿後只有 {_pct(t.loc[g, '期滿後續約率'], 0)} 續約。")
            break
    if "不限期" in t.index and "第24期" in t:
        a2b += f"不限期的人第 12 期 {_pct(t.loc['不限期', '第12期'], 0)}、第 24 期 {_pct(t.loc['不限期', '第24期'], 0)}。"
    a2b += "（灰色＝方案已期滿或觀察中的方案不到 20 個。）"
    a6 = (f"在扣款日前 7 天寄，對象是下一期落在第 {'、'.join(map(str, risky))} 期，或約定期數只剩 0–2 期的人。"
          f"本月預防性提醒名單共 {s['預防性提醒名單']['人數']} 人；起死回生名單 {s['起死回生名單']['人數']} 人。")

    # ---- HTML
    def section(no, q, ans, cid, extra=""):
        return (f'<h2 style="font-size:18px;margin:28px 0 6px;color:{INK}">{no}. {E(q)}</h2>'
                f'<p style="margin:0 0 8px;color:{INK};line-height:1.7">{ans}</p>'
                f'<img src="cid:{cid}" width="640" style="width:100%;max-width:640px;height:auto;display:block;'
                f'border:1px solid {GRID};border-radius:8px" alt="{E(q)}">{extra}')

    prev_cols = ["系統編號", "建議寄送日", "建議行動", "理由", "約定金額"]
    rev_cols = ["系統編號", "建議行動", "理由", "狀態", "約定金額"]
    o = s["總覽"]
    body = f"""<div style="font-family:'Microsoft JhengHei','Noto Sans TC',Arial,sans-serif;max-width:680px;color:{INK}">
<p style="color:{INK2};margin:0 0 4px">野灣公益勸募科｜定期定額月報（資料期間 {E(o['資料期間'])}，示範資料皆為虛構）</p>
<h1 style="font-size:22px;margin:0 0 12px">本月先聯絡 {s['預防性提醒名單']['人數'] + s['起死回生名單']['人數']} 人：預防性 {s['預防性提醒名單']['人數']}、起死回生 {s['起死回生名單']['人數']}</h1>
<p style="line-height:1.7;margin:0 0 4px">兩份名單都在附件（CSV，可直接匯入 Google Sheets）。每一列都有「建議寄送日／建議行動／理由」，可以直接分工。
下面每張圖回答一個問題；只有通過統計檢定的差異才寫成結論，沒通過的會明寫「資料不支持」。</p>
<h2 style="font-size:18px;margin:24px 0 6px">留住定期定額</h2>
{section(1, "人最常在第幾期停下來？", a1, "att0")}
{section(2, "設定了多少期、扣到幾成的時候最容易停？", a2, "att1")}
{section("2-1", "設定幾期的人，各期還有幾 % 在扣款？（Cohort）", a2b, "att6")}
{section(3, "扣款失敗之後：事後挽回與預防", a3, "att2")}
<h2 style="font-size:18px;margin:32px 0 6px">讓人開始支持</h2>
{section(4, "從什麼活動獲客率更高？能提升定期定額嗎？", a4, "att3")}
{section(5, "哪類活動進來的人留存率較好？", a5, "att4")}
<h2 style="font-size:18px;margin:32px 0 6px">本月行銷名單</h2>
{section(6, "什麼時間點應該預防性提醒？", a6, "att5")}
<h3 style="font-size:15px;margin:20px 0 4px">預防性提醒名單（前 10 位，完整見附件「名單_預防性提醒.csv」）</h3>
{_table(prev, prev_cols)}
<h3 style="font-size:15px;margin:20px 0 4px">起死回生名單（前 10 位，完整見附件「名單_起死回生.csv」）</h3>
{_table(revive, rev_cols)}
<p style="color:{MUTED};font-size:12px;line-height:1.6;margin-top:24px">
說明：起死回生名單已排除「另有方案正在扣款」的人（換卡、續約），不會打擾仍在捐款的人。
停止率附 95% 信賴區間；活動成效以活動前 14 天為基準並做多重比較校正。
資料缺口：④ 定期明細一人只有一列、捐款頁沒有來源欄位——建議匯出改為一個方案一列，捐款連結加上活動代碼。<br>
由 yewan-donor-cohort skill 自動產生。</p></div>"""
    (em / "email.html").write_text(body, encoding="utf-8")
    plain = "\n\n".join(["野灣定期定額月報", "1. " + a1, "2. " + a2, "2-1. " + a2b, "3. " + a3, "4. " + a4, "5. " + a5, "6. " + a6,
                         "名單見附件。"]).replace("<b>", "").replace("</b>", "")
    (em / "email.txt").write_text(plain, encoding="utf-8")
    manifest = {
        "subject": f"【野灣定期定額月報】本月先聯絡 {s['預防性提醒名單']['人數'] + s['起死回生名單']['人數']} 人：預防性提醒＋起死回生名單",
        "html": str((em / "email.html").resolve()), "text": str((em / "email.txt").resolve()),
        "attachments": [{"path": str((em / n).resolve()), "filename": n, "mimeType": "image/png", "inline": True}
                        for n, _ in charts]
        + [{"path": str((em / n).resolve()), "filename": n, "mimeType": "text/csv", "inline": False} for n in lists],
    }
    (em / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("yewan_output"))
    a = ap.parse_args()
    m = build(a.output_dir)
    print(m["subject"])
    print(f"信件與附件：{(a.output_dir / 'email').resolve()}")


if __name__ == "__main__":
    main()
