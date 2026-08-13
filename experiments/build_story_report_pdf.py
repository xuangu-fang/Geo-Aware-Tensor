#!/usr/bin/env python3
"""Render the Chinese story/progress report to a self-contained PDF."""
from __future__ import annotations

import argparse
from pathlib import Path

import markdown
from weasyprint import HTML


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FONT = REPO_ROOT / ".python-packages/fonts/NotoSansCJKsc-Regular.otf"

CSS = r"""
@font-face {
  font-family: NotoCJK;
  src: url('__FONT_URI__');
}
@page {
  size: A4;
  margin: 17mm 16mm 18mm 16mm;
  @bottom-center { content: counter(page); font-size: 8pt; color: #667085; }
}
html { font-family: NotoCJK, sans-serif; color: #172033; font-size: 10.2pt; line-height: 1.58; }
body { margin: 0; }
h1 { color: #123a63; font-size: 23pt; line-height: 1.25; margin: 0 0 10mm; border-bottom: 3px solid #2c7fb8; padding-bottom: 5mm; }
h2 { color: #174f78; font-size: 16pt; margin-top: 8mm; border-bottom: 1px solid #b9d5e8; padding-bottom: 2mm; break-after: avoid; }
h3 { color: #24698f; font-size: 12.5pt; margin-top: 5mm; break-after: avoid; }
p { margin: 2.3mm 0; }
blockquote { border-left: 4px solid #2c7fb8; margin: 4mm 0; padding: 2mm 4mm; background: #eef7fc; color: #153d57; }
table { border-collapse: collapse; width: 100%; margin: 4mm 0 6mm; font-size: 8.6pt; break-inside: avoid; }
th { background: #dfeff8; color: #123a63; font-weight: 700; }
th, td { border: 0.5px solid #9fb9c8; padding: 1.8mm 2mm; vertical-align: top; }
tr:nth-child(even) td { background: #f7fafc; }
pre { background: #f2f5f7; border: 0.5px solid #cbd5dc; padding: 3mm; font-size: 8.7pt; line-height: 1.4; white-space: pre-wrap; break-inside: avoid; }
code { font-family: monospace; color: #7b2d26; }
img { display: block; max-width: 94%; max-height: 118mm; margin: 5mm auto 7mm; object-fit: contain; }
ul, ol { margin-top: 2mm; padding-left: 7mm; }
li { margin: 1.2mm 0; }
hr { border: 0; border-top: 1px solid #b9d5e8; margin: 8mm 0; }
strong { color: #123a63; }
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, nargs="?",
                        default=Path("papers/zh/最新版本故事线与进展总报告.md"))
    parser.add_argument("--output", type=Path,
                        default=Path("papers/zh/最新版本故事线与进展总报告.pdf"))
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT,
                        help="CJK OpenType font embedded in the PDF")
    args = parser.parse_args()
    if not args.font.is_file():
        raise SystemExit(
            f"missing CJK font: {args.font}\n"
            "Pass --font /path/to/a/CJK-font.otf. The committed PDF is already self-contained."
        )
    body = markdown.markdown(
        args.source.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    css = CSS.replace("__FONT_URI__", args.font.resolve().as_uri())
    html = f"<!doctype html><html lang='zh-CN'><meta charset='utf-8'><style>{css}</style><body>{body}</body></html>"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html, base_url=str(args.source.parent.resolve())).write_pdf(args.output)
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
