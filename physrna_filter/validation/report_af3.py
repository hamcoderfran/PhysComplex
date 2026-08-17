"""
Generate shareable HTML (and optional PDF) reports from AF3 screening CSVs.

Designed for biologists: one page per run with ranked table, verdict colours,
and plain-language summaries.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

_VERDICT_CSS = {
    "PASS": "#1a7f37",
    "WARN": "#9a6700",
    "FAIL": "#cf222e",
    "ERROR": "#6e7781",
    "UNKNOWN": "#57606a",
}


def _esc(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    return html.escape(str(val))


def _verdict_badge(verdict: str | None) -> str:
    v = (verdict or "UNKNOWN").upper()
    color = _VERDICT_CSS.get(v, _VERDICT_CSS["UNKNOWN"])
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-weight:600;">{_esc(v)}</span>'
    )


def _recommendation_row(row: pd.Series) -> str:
    combined = str(row.get("combined_verdict", "")).upper()
    bio = str(row.get("bio_verdict", "")).upper()
    rank = row.get("rank")
    if combined == "ERROR":
        return "Could not score — check input file"
    if bio == "FAIL":
        return "Do not pursue — RNA partner likely wrong"
    if combined == "FAIL":
        return "Low confidence — deprioritize"
    if rank is not None and int(rank) <= 3:
        return "Top candidate — consider for experiments"
    if combined == "WARN":
        return "Promising — review structure manually"
    return "Acceptable — lower priority"


def build_html_report(
    df: pd.DataFrame,
    *,
    title: str = "PhysRNA AF3 Screening Report",
    subtitle: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> str:
    work = df.copy()
    if "rank" not in work.columns and "composite_score" in work.columns:
        work = work.sort_values("composite_score", kind="mergesort")
        work.insert(0, "rank", range(1, len(work) + 1))

    cols = [
        c for c in (
            "rank", "job_name", "file", "rbp_name", "rna_sequence",
            "af3_iptm", "af3_ptm", "composite_score", "combined_verdict",
            "bio_verdict", "gt_score_norm", "clash_n_severe", "recommendation",
        )
        if c in work.columns
    ]
    if "recommendation" not in work.columns:
        work["recommendation"] = work.apply(_recommendation_row, axis=1)
        if "recommendation" not in cols:
            cols.append("recommendation")

    n_pass = int((work.get("combined_verdict") == "PASS").sum()) if "combined_verdict" in work else 0
    n_warn = int((work.get("combined_verdict") == "WARN").sum()) if "combined_verdict" in work else 0
    n_fail = int((work.get("combined_verdict") == "FAIL").sum()) if "combined_verdict" in work else 0
    n_err = int((work.get("combined_verdict") == "ERROR").sum()) if "combined_verdict" in work else 0

    metrics_block = ""
    if metrics:
        items = "".join(
            f"<li><strong>{_esc(k)}</strong>: {_esc(v)}</li>"
            for k, v in metrics.items()
        )
        metrics_block = f"<ul class='metrics'>{items}</ul>"

    header_rows = "".join(f"<th>{_esc(c)}</th>" for c in cols)
    body_rows = []
    for _, row in work.iterrows():
        cells = []
        for c in cols:
            val = row.get(c)
            if c in ("combined_verdict", "bio_verdict"):
                cells.append(f"<td>{_verdict_badge(str(val) if val is not None else None)}</td>")
            elif c in ("af3_iptm", "af3_ptm", "composite_score", "gt_score_norm"):
                try:
                    cells.append(f"<td>{float(val):.3f}</td>" if pd.notna(val) else "<td>—</td>")
                except (TypeError, ValueError):
                    cells.append(f"<td>{_esc(val)}</td>")
            else:
                cells.append(f"<td>{_esc(val)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sub = f"<p class='sub'>{_esc(subtitle)}</p>" if subtitle else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{_esc(title)}</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 2rem; color: #1f2328; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 0.2rem; }}
  .sub, .ts {{ color: #57606a; }}
  .summary {{ display: flex; gap: 1rem; flex-wrap: wrap; margin: 1.2rem 0; }}
  .card {{ border: 1px solid #d0d7de; border-radius: 8px; padding: 0.8rem 1rem; min-width: 120px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.92rem; margin-top: 1rem; }}
  th, td {{ border: 1px solid #d0d7de; padding: 0.45rem 0.55rem; text-align: left; vertical-align: top; }}
  th {{ background: #f6f8fa; }}
  tr:nth-child(even) {{ background: #f6f8fa; }}
  .metrics {{ margin-top: 0.5rem; }}
  @media print {{ body {{ margin: 0.5rem; }} }}
</style>
</head>
<body>
<h1>{_esc(title)}</h1>
{sub}
<p class="ts">Generated {ts}</p>
<div class="summary">
  <div class="card"><div>PASS</div><strong>{n_pass}</strong></div>
  <div class="card"><div>WARN</div><strong>{n_warn}</strong></div>
  <div class="card"><div>FAIL</div><strong>{n_fail}</strong></div>
  <div class="card"><div>ERROR</div><strong>{n_err}</strong></div>
  <div class="card"><div>Total</div><strong>{len(work)}</strong></div>
</div>
{metrics_block}
<p><strong>How to read this:</strong> Lower <em>composite_score</em> is better for PhysRNA.
Higher <em>af3_iptm</em> is better for AF3. Use <em>recommendation</em> for go/no-go.</p>
<table>
<thead><tr>{header_rows}</tr></thead>
<tbody>
{"".join(body_rows)}
</tbody>
</table>
<p class="ts">Save as PDF: open in Chrome/Edge → Print → Save as PDF</p>
</body>
</html>"""


def write_html_report(
    csv_path: str | Path,
    output_html: str | Path,
    *,
    title: str | None = None,
    metrics_json: str | Path | None = None,
) -> Path:
    csv_path = Path(csv_path)
    output_html = Path(output_html)
    df = pd.read_csv(csv_path)
    metrics = None
    if metrics_json and Path(metrics_json).is_file():
        with open(metrics_json, encoding="utf-8") as fh:
            metrics = json.load(fh)
    html_text = build_html_report(
        df,
        title=title or "PhysRNA AF3 Screening Report",
        subtitle=f"Source: {csv_path.name}",
        metrics=metrics,
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_text, encoding="utf-8")
    return output_html


def write_pdf_report(html_path: str | Path, output_pdf: str | Path) -> Path | None:
    """Optional PDF via weasyprint when installed."""
    html_path = Path(html_path)
    output_pdf = Path(output_pdf)
    try:
        from weasyprint import HTML  # type: ignore
    except ImportError:
        return None
    HTML(filename=str(html_path)).write_pdf(str(output_pdf))
    return output_pdf


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Build HTML report from AF3 screening CSV")
    ap.add_argument("csv", help="Screening or ranking CSV")
    ap.add_argument("--output", "-o", default="af3_report.html")
    ap.add_argument("--metrics-json", default=None)
    ap.add_argument("--pdf", action="store_true", help="Also write PDF (needs weasyprint)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    html_path = write_html_report(
        args.csv,
        args.output,
        title=args.title,
        metrics_json=args.metrics_json,
    )
    print(f"Wrote {html_path}")
    if args.pdf:
        pdf_path = write_pdf_report(html_path, Path(args.output).with_suffix(".pdf"))
        if pdf_path:
            print(f"Wrote {pdf_path}")
        else:
            print("PDF skipped — install weasyprint or use browser Print → Save as PDF")


if __name__ == "__main__":
    main()
