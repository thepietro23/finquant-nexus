"""
make_template.py — Create rru_template.docx for pandoc --reference-doc.
Run once: python make_template.py
"""
from pathlib import Path
from docx import Document
from docx.shared import Pt, Mm
from docx.enum.text import WD_LINE_SPACING

BASE     = Path(__file__).parent
TEMPLATE = BASE / "rru_template.docx"
FONT     = "Times New Roman"


def _font(style, size, bold=False, italic=False):
    style.font.name  = FONT
    style.font.size  = Pt(size)
    style.font.bold  = bold
    style.font.italic = italic


def _spacing(style, before=0, after=6, ls=WD_LINE_SPACING.ONE_POINT_FIVE):
    pf = style.paragraph_format
    pf.space_before      = Pt(before)
    pf.space_after       = Pt(after)
    pf.line_spacing_rule = ls


def create():
    doc = Document()

    # ── Page setup ────────────────────────────────────────────────────────────
    sec = doc.sections[0]
    sec.page_width    = Mm(210)
    sec.page_height   = Mm(297)
    sec.left_margin   = Mm(38)
    sec.right_margin  = Mm(25.4)
    sec.top_margin    = Mm(25.4)
    sec.bottom_margin = Mm(25.4)

    # ── Styles ────────────────────────────────────────────────────────────────
    _font(doc.styles['Normal'],    12);  _spacing(doc.styles['Normal'],    0,  6)
    _font(doc.styles['Heading 1'], 14, bold=True);  _spacing(doc.styles['Heading 1'], 12, 4)
    _font(doc.styles['Heading 2'], 12, bold=True);  _spacing(doc.styles['Heading 2'],  8, 4)
    _font(doc.styles['Heading 3'], 12, bold=True, italic=True);  _spacing(doc.styles['Heading 3'], 6, 2)
    _font(doc.styles['Heading 4'], 12, italic=True);             _spacing(doc.styles['Heading 4'], 4, 2)

    # Pandoc maps block-code to "verbatim" — use Courier New 10pt
    if 'verbatim' not in [s.name for s in doc.styles]:
        vs = doc.styles.add_style('verbatim', 1)  # 1 = paragraph style
    else:
        vs = doc.styles['verbatim']
    vs.font.name = 'Courier New'
    vs.font.size = Pt(9)
    _spacing(vs, 4, 4, WD_LINE_SPACING.SINGLE)

    # Placeholder paragraph (pandoc needs at least one paragraph in template)
    doc.add_paragraph("", style='Normal')

    doc.save(str(TEMPLATE))
    print(f"Created: {TEMPLATE}")


if __name__ == '__main__':
    create()
