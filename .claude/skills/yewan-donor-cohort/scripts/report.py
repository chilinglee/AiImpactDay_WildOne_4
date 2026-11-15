"""產生 report.html：KPI、可信度標記、SVG 圖表（hover 提示）、可排序/篩選表格。純 HTML，無外部依賴。"""
from __future__ import annotations

import html
import math
from pathlib import Path

import pandas as pd

CSS = """
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--line:#e4e3df;
--s1:#2a78d6;--s2:#eb6834;--good:#0ca30c;--warn:#fab219;--crit:#d03b3b;--chip:#f0efec}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--card:#232322;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#8f8e86;--line:#383835;--s1:#3987e5;--s2:#d95926;--chip:#383835}}
:root[data-theme="dark"]{--bg:#1a1a19;--card:#232322;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8e86;--line:#383835;
--s1:#3987e5;--s2:#d95926;--chip:#383835}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 "Noto Sans TC","Microsoft JhengHei",system-ui,sans-serif}
main{max-width:1120px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 8px;padding-top:8px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:24px 0 6px}p.lead{color:var(--ink2);margin:0 0 16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:16px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:24px;font-variant-numeric:tabular-nums}.kpi span{color:var(--ink2);font-size:13px}
.badge{display:inline-block;font-size:12px;padding:1px 8px;border-radius:99px;background:var(--chip);white-space:nowrap}
.badge::before{margin-right:4px}.b-ok::before{content:"✔";color:var(--good)}.b-no::before{content:"⚠";color:var(--warn)}
.b-err::before{content:"✖";color:var(--crit)}
.note{background:var(--card);border-left:4px solid var(--warn);padding:10px 14px;border-radius:6px;margin:12px 0;color:var(--ink2)}
.tbl-wrap{overflow:auto;max-height:520px;border:1px solid var(--line);border-radius:8px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{position:sticky;top:0;background:var(--card);cursor:pointer;user-select:none;color:var(--ink2)}
th:hover{color:var(--ink)}td.num{text-align:right}
.filter{margin:6px 0;padding:6px 10px;width:min(340px,100%);border:1px solid var(--line);border-radius:6px;
background:var(--card);color:var(--ink)}
.chart{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:8px 0}
.chart svg{width:100%;height:auto;display:block}.chart text{fill:var(--ink2);font-size:11px}
.chart .grid{stroke:var(--line)}.chart .bar{fill:var(--s1)}.chart .bar:hover,.chart .dot:hover{opacity:.75}
.chart .ln{fill:none;stroke:var(--s1);stroke-width:2}.chart .ev{stroke:var(--s2);stroke-dasharray:3 3}
.chart .whisk{stroke:var(--ink2);stroke-width:1}.chart .ref{stroke:var(--muted);stroke-dasharray:4 4}
.cap{color:var(--muted);font-size:12px;margin-top:4px}
.heat td.c{text-align:center;color:#0b0b0b}
"""

JS = """
document.querySelectorAll('table.sortable').forEach(t=>{
 t.querySelectorAll('th').forEach((th,i)=>th.addEventListener('click',()=>{
  const tb=t.tBodies[0],rows=[...tb.rows],asc=th.dataset.dir!=='asc';th.dataset.dir=asc?'asc':'desc';
  const v=r=>{const x=r.cells[i].dataset.v??r.cells[i].innerText;const n=parseFloat(x);return isNaN(n)?x:n};
  rows.sort((a,b)=>{const A=v(a),B=v(b);return (A>B?1:A<B?-1:0)*(asc?1:-1)});rows.forEach(r=>tb.appendChild(r));}));});
document.querySelectorAll('input.filter').forEach(inp=>inp.addEventListener('input',()=>{
 const q=inp.value.trim().toLowerCase();document.getElementById(inp.dataset.t).querySelectorAll('tbody tr')
 .forEach(r=>r.style.display=!q||r.innerText.toLowerCase().includes(q)?'':'none');}));
"""

E = html.escape


def badge(v: str) -> str:
    v = str(v)
    cls = "b-ok" if ("顯著" in v and "無" not in v and "不宜" not in v) or v == "OK" else \
          "b-err" if v == "ERROR" else "b-no"
    return f'<span class="badge {cls}">{E(v)}</span>'


def fmt(x) -> tuple[str, str, bool]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "", "", True
    if isinstance(x, (int,)) or (isinstance(x, float) and float(x).is_integer() and abs(x) >= 1):
        return f"{int(x):,}", str(x), True
    if isinstance(x, float):
        return f"{x:.4g}" if abs(x) < 1 else f"{x:,.2f}", str(x), True
    return E(str(x)), E(str(x)), False


def table(df: pd.DataFrame, tid: str, filt: bool = False, limit: int | None = None) -> str:
    df = df.head(limit) if limit else df
    head = "".join(f"<th>{E(str(c))}</th>" for c in df.columns)
    body = []
    for row in df.itertuples(index=False):
        cells = []
        for col, x in zip(df.columns, row):
            if col in ("判定", "嚴重度"):
                cells.append(f"<td>{badge(x)}</td>")
                continue
            s, v, num = fmt(x)
            cells.append(f'<td class="{"num" if num else ""}" data-v="{v}">{s}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    f = f'<input class="filter" data-t="{tid}" placeholder="篩選（輸入任何文字）…">' if filt else ""
    return f'{f}<div class="tbl-wrap"><table id="{tid}" class="sortable"><thead><tr>{head}</tr></thead>' \
           f'<tbody>{"".join(body)}</tbody></table></div>'


def bar_chart(labels, values, lo=None, hi=None, ref=None, ylabel="", fmt_v=lambda v: f"{v:.1%}") -> str:
    W, H, L, B, T = 720, 260, 48, 28, 12
    vmax = max([v for v in (hi or values) if v == v] + [ref or 0]) * 1.15 or 1
    n = len(values)
    bw = (W - L - 8) / n
    y = lambda v: T + (H - T - B) * (1 - v / vmax)
    parts = []
    for k in range(5):
        gv = vmax * k / 4
        parts.append(f'<line class="grid" x1="{L}" x2="{W}" y1="{y(gv):.1f}" y2="{y(gv):.1f}"/>'
                     f'<text x="{L-6}" y="{y(gv)+4:.1f}" text-anchor="end">{fmt_v(gv)}</text>')
    for i, (lab, v) in enumerate(zip(labels, values)):
        x = L + i * bw + 2
        w = max(bw - 4, 2)
        tip = f"{lab}：{fmt_v(v)}" + (f"（95% 區間 {fmt_v(lo[i])}–{fmt_v(hi[i])}）" if lo else "")
        parts.append(f'<rect class="bar" x="{x:.1f}" y="{y(v):.1f}" width="{w:.1f}" height="{max(y(0)-y(v),0):.1f}" '
                     f'rx="3"><title>{E(tip)}</title></rect>')
        if lo:
            cx = x + w / 2
            parts.append(f'<line class="whisk" x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(hi[i]):.1f}" y2="{y(lo[i]):.1f}"/>')
        parts.append(f'<text x="{x + w/2:.1f}" y="{H-10}" text-anchor="middle">{E(str(lab))}</text>')
    if ref is not None:
        parts.append(f'<line class="ref" x1="{L}" x2="{W}" y1="{y(ref):.1f}" y2="{y(ref):.1f}"/>'
                     f'<text x="{W-4}" y="{y(ref)-4:.1f}" text-anchor="end">平均 {fmt_v(ref)}</text>')
    return f'<div class="chart"><svg viewBox="0 0 {W} {H}" role="img" aria-label="{E(ylabel)}">{"".join(parts)}</svg></div>'


def line_with_events(dates, values, events) -> str:
    W, H, L, B, T = 720, 260, 40, 28, 16
    d0, d1 = dates.min(), dates.max()
    span = (d1 - d0).days or 1
    vmax = max(values) * 1.15 or 1
    x = lambda d: L + (W - L - 8) * (d - d0).days / span
    y = lambda v: T + (H - T - B) * (1 - v / vmax)
    pts = " ".join(f"{x(d):.1f},{y(v):.1f}" for d, v in zip(dates, values))
    parts = []
    for k in range(5):
        gv = vmax * k / 4
        parts.append(f'<line class="grid" x1="{L}" x2="{W}" y1="{y(gv):.1f}" y2="{y(gv):.1f}"/>'
                     f'<text x="{L-6}" y="{y(gv)+4:.1f}" text-anchor="end">{gv:.0f}</text>')
    for _, ev in events.iterrows():
        ex = x(pd.Timestamp(ev["日期"]))
        parts.append(f'<line class="ev" x1="{ex:.1f}" x2="{ex:.1f}" y1="{T}" y2="{H-B}"><title>{E(ev["事件編號"] + " " + ev["事件名稱"])}</title></line>'
                     f'<text x="{ex+2:.1f}" y="{T+8}">{E(ev["事件編號"][-2:])}</text>')
    parts.append(f'<polyline class="ln" points="{pts}"/>')
    for d, v in zip(dates, values):
        parts.append(f'<circle class="dot" cx="{x(d):.1f}" cy="{y(v):.1f}" r="5" fill="transparent">'
                     f'<title>{d.date()} 當週：{v} 人</title></circle>')
    for m in pd.date_range(d0, d1, freq="MS"):
        parts.append(f'<text x="{x(m):.1f}" y="{H-10}" text-anchor="middle">{m.strftime("%y/%m")}</text>')
    return f'<div class="chart"><svg viewBox="0 0 {W} {H}" role="img" aria-label="每週新捐款人">{"".join(parts)}</svg>' \
           f'<div class="cap">藍線＝每週新捐款人；橘色虛線＝活動時間軸事件（滑過可看名稱）。</div></div>'


def heatmap(df: pd.DataFrame) -> str:
    ramp = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf"]
    mcols = [c for c in df.columns if c.startswith("M")]
    head = "<th>Cohort月份</th><th>人數</th>" + "".join(f"<th>{c}</th>" for c in mcols)
    rows = []
    for _, r in df.iterrows():
        cells = [f"<td>{E(str(r['Cohort月份']))}</td><td class='num'>{int(r['Cohort人數']):,}</td>"]
        for c in mcols:
            v = r[c]
            if pd.isna(v):
                cells.append("<td></td>")
            else:
                col = ramp[min(int(v * len(ramp)), len(ramp) - 1)]
                fg = "#ffffff" if v >= 0.55 else "#0b0b0b"
                cells.append(f'<td class="c" style="background:{col};color:{fg}" title="{r["Cohort月份"]} {c}：{v:.1%}">{v:.0%}</td>')
        small = int(r["Cohort人數"]) < 30
        rows.append(f'<tr style="opacity:{.35 if small else 1}" title="{"人數 < 30，比率不穩定" if small else ""}">'
                    + "".join(cells) + "</tr>")
    return f'<div class="tbl-wrap heat"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>' \
           '<div class="cap">M0＝首月，M1＝下個月…；空白＝尚無法觀察（不是 0）。顏色越深＝留存越高。淡化的列＝人數不足 30，比率不穩定，勿解讀。</div>'


def build_report(out_dir: Path, summary: dict, checks: pd.DataFrame, res: dict) -> None:
    o = summary["總覽"]
    kpis = [("期間捐款人", f"{o['期間捐款人數']:,}", f"占聯絡人 {o['期間捐款率']:.1%}"),
            ("定期捐款人", f"{o['定期捐款人數']:,}", f"目前扣款中 {o['目前扣款中人數']:,} 人"),
            ("真流失方案", f"{o['期間真流失方案數']:,}", f"其中扣款失敗 {o['其中被動流失(扣款失敗)']:,}"),
            ("換方案（不是流失）", f"{o['換方案非流失數']:,}", "換卡／續約後仍在扣款"),
            ("期間新捐款人", f"{o['期間新捐款人數']:,}", "聯絡人在期間內建立"),
            ("25 歲以下", f"{o['25歲以下占捐款人(已填)']:.1%}", f"占已填年齡捐款人（填寫率 {o['年齡填寫率(捐款人)']:.0%}）")]
    kpi_html = "".join(f'<div class="kpi"><span>{a}</span><b>{b}</b><span>{c}</span></div>' for a, b, c in kpis)

    issues = checks[checks["嚴重度"] != "OK"]
    warn_n = (issues["嚴重度"] == "WARN").sum()
    err_n = (issues["嚴重度"] == "ERROR").sum()

    sp = res["t2_停止期數"]
    sp_show = sp[sp["期數"] <= 24]
    overall = sp_show["於此期流失"].sum() / sp_show["仍在觀察中的方案數"].sum()
    stop_chart = bar_chart(sp_show["期數"].tolist(), sp_show["停止率(hazard)"].tolist(),
                           sp_show["停止率95%下界"].tolist(), sp_show["停止率95%上界"].tolist(), overall, "各期停止率")

    weekly = res["t3_每週新捐款人"]
    ev_df = res["t3_活動前後"].drop_duplicates("事件編號")[["事件編號", "事件名稱", "日期"]]
    wk_chart = line_with_events(pd.to_datetime(weekly["週起始日"]), weekly["新捐款人數"].tolist(), ev_df)

    def tests_list(keys):
        out = []
        for k in keys:
            t = summary["檢定結果"].get(k)
            if t:
                out.append(f"<li>{E(t['檢定'])}：p = {t['p值']} {badge(t['判定'])}</li>")
        return "<ul>" + "".join(out) + "</ul>"

    seg_rows = "".join(f"<tr><td>{E(k)}</td><td class='num'>{v['p值']}</td><td class='num'>{v['校正後p值']}</td><td>{badge(v['判定'])}</td></tr>"
                       for k, v in summary["分群檢定"].items())
    seg_tbl = f'<div class="tbl-wrap"><table class="sortable"><thead><tr><th>指標｜分群維度</th><th>p值</th><th>校正後p值</th><th>判定</th></tr></thead><tbody>{seg_rows}</tbody></table></div>'

    page = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>野灣捐款留存分析</title><style>{CSS}</style></head>
<body><main>
<h1>野灣捐款留存分析報告</h1>
<p class="lead">資料期間 {E(o['資料期間'])}（虛構示範資料）。每個發現都附上可信度判定：<span class="badge b-ok">差異顯著</span> 可以拿來做決策；
<span class="badge b-no">不宜下結論</span> 代表差異可能只是隨機波動。</p>
<div class="kpis">{kpi_html}</div>

<h2>0. 資料可信度檢查</h2>
<p class="lead">ERROR {err_n} 項、WARN {warn_n} 項。WARN 不會阻擋分析，但會改變「哪個欄位可以信」，下方各章節已依此調整算法。</p>
<div class="note">④ 定期定額明細「一人只有一列」，常常是舊方案。本報告先用 ② 的定期捐款編號還原出 {o['定期方案數(還原後)']:,} 個方案
（④ 只列 {o['其中④有列']:,} 個）。④ 顯示「已失敗／待處理」的人裡，有 {o['④顯示失敗或待處理但此人仍在扣款']:,} 位其實還有另一個方案在扣款
——直接用 ④ 做名單，會打擾到正在捐款的人。</div>
{table(issues, "tbl-checks", filt=True)}

<h2>任務一：誰在支持野灣（輪廓）</h2>
{tests_list(["t1_支持者輪廓"])}
{table(res["t1_支持者輪廓"], "tbl-t1")}
<p class="cap">25 歲以下比例的分母＝已填年齡的人；年齡填寫率一併列出。</p>

<h2>任務二：留住定期定額捐款人</h2>
<h3>停下來的方案，哪些是真的流失？</h3>
<p class="lead">「換方案」＝方案停了，但同一人另有方案正在扣款（官網換卡必須重設方案）。只有「真流失」需要挽回。</p>
{table(res["t2_流失類型"], "tbl-ctype")}

<h3>捐款人最常在第幾期停止？</h3>
<p class="lead">期數取自 ② 交易編號後段（＝④ 執行期數）。舊方案從期間內第一期開始觀察（延遲進入），期間新開的方案從第 0 期開始；
第 0 期＝第一次扣款就沒成功。誤差線＝95% 信賴區間（圖只畫到第 24 期）。</p>
{stop_chart}
{tests_list(["t2_停止期數"])}
{table(sp, "tbl-stop")}

<h3>設定期數 vs 實際</h3>
{tests_list(["t2_設定期數vs實際"])}
{table(res["t2_設定期數vs實際"], "tbl-term")}

<h3>哪些扣款失敗可以挽回？</h3>
{tests_list(["t2_扣款失敗與恢復"])}
{table(res["t2_扣款失敗與恢復"], "tbl-fail")}

<h3>流失預警：距最近一次扣款幾天</h3>
{table(res["t2_沉默天數分布"], "tbl-inact")}
<h3>月流失率（以人計）</h3>
<p class="lead">某人最後一次定期扣款在 m 月、之後到截止日都沒再扣＝m 月流失。以「人」計算，換卡不會被誤算成流失。
距截止日不到「扣款間隔 P90」天的月份不判定。</p>
{table(res["t2_月流失率"], "tbl-churn")}

<h3>新定期捐款人 Cohort 留存（以首次扣款月份分群）</h3>
{heatmap(res["t2_定期Cohort留存"])}

<h3>哪一群人比較容易流失？（分群檢定總表）</h3>
{seg_tbl}
<p class="cap">完整分群比率與信賴區間見 t2_分群比較_流失.csv、t3_新捐款人後續行為.csv。</p>

<h3>優先提醒名單（一人一列，前 200 筆，可排序／篩選）</h3>
<p class="lead">分數規則透明：每一分都寫在「理由」欄。此人另有方案在扣款的舊方案已排除。完整名單見 t2_優先提醒名單.csv。</p>
{table(res["t2_優先提醒名單"], "tbl-remind", filt=True, limit=200)}

<h2>任務三：找出讓人開始支持的時刻</h2>
<p class="lead">新捐款人＝期間內有捐款、且聯絡人在期間內才建立的人（官網多半在捐款當下建立聯絡人）。</p>
{wk_chart}
<h3>每個活動窗內，新捐款人／新定期方案是否顯著增加？</h3>
<p class="lead">活動窗＝活動日起 7 天；備註有「至 M/D」的以該日為止。基準＝活動前 14 天日均。p 值以活動數做 Bonferroni 校正。</p>
{table(res["t3_活動前後"], "tbl-ev", filt=True)}
<h3>時間軸上找不到原因的高峰</h3>
<p class="lead">單日數量遠高於前 28 天中位數（p &lt; 0.001 且 ≥ 2 倍）的日子。「（找不到）」＝需要勸募科回想當天發生什麼、並開始記錄。</p>
{table(res["t3_未解釋高峰"], "tbl-spike")}
<h3>新捐款人之後有沒有再捐、轉定期？</h3>
{table(res["t3_新捐款人後續行為"], "tbl-follow", filt=True)}
<h3>新捐款人 Cohort 留存</h3>
{heatmap(res["t3_新捐款人Cohort留存"])}
<p class="cap">由 yewan-donor-cohort skill 產生。所有數字可由 scripts/test_yewan_cohort.py 獨立重算驗證。</p>
</main><script>{JS}</script></body></html>"""
    (out_dir / "report.html").write_text(page, encoding="utf-8")
