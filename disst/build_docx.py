"""
build_docx.py — Convert dissertation MD files → DOCX via pandoc.

Equations: 4-space indented lines in chapter files are converted to proper
LaTeX $$...$$ blocks → pandoc → native Word OMML equation objects.

Usage:
  python make_template.py   # once
  python build_docx.py      # every time
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

BASE     = Path(__file__).parent
SRC      = BASE / "chap_content"
DST      = BASE / "dt_docs2"
TEMPLATE = BASE / "rru_template.docx"
DST.mkdir(parents=True, exist_ok=True)

# ── Pandoc executable ────────────────────────────────────────────────────────
# Try PATH first, then winget default install location
_PANDOC_CANDIDATES = [
    'pandoc',
    r'C:\Users\praveen.rawal\AppData\Local\Pandoc\pandoc.exe',
    r'C:\Program Files\Pandoc\pandoc.exe',
]

def _find_pandoc():
    import shutil
    for p in _PANDOC_CANDIDATES:
        if shutil.which(p) or Path(p).exists():
            return p
    return None

PANDOC = _find_pandoc()

# ── Files that contain math equations (not just indented centered text) ───────
CHAPTER_PREFIXES = ('03_', '04_', '05_', '06_', '07_', '08_')

# ── Equation map: original text → proper LaTeX ───────────────────────────────
# Every 4-space indented equation line in the chapter files is listed here.
# If a line is not in the map it falls back to wrapping the text as-is in $$.
LATEX_MAP = {
    # ── Chapter 2 — Literature Review ────────────────────────────────────────
    "L_CLIP = E[ min( r_t(θ) * A_t ,  clip(r_t(θ), 1-ε, 1+ε) * A_t ) ]":
        r"L_{\text{CLIP}} = \mathbb{E}\!\left["
        r"\min\!\left(r_t(\theta)\cdot A_t,\;"
        r"\text{clip}\!\left(r_t(\theta),\,1{-}\varepsilon,\,1{+}\varepsilon\right)\cdot A_t"
        r"\right)\right]",

    "e_ij = LeakyReLU( aᵀ [ W * h_i  ‖  W * h_j ] )":
        r"e_{ij} = \text{LeakyReLU}\!\left(\mathbf{a}^{T}"
        r"\!\left[Wh_i \;\left\|\; Wh_j\right.\right]\right)",

    "α_ij = exp(e_ij) / Σ over k in N_i of exp(e_ik)":
        r"\alpha_{ij} = \frac{\exp(e_{ij})}{\displaystyle\sum_{k\in\mathcal{N}_i}\exp(e_{ik})}",

    "h'_i = σ ( Σ over j in N_i of α_ij * W * h_j )":
        r"h'_i = \sigma\!\left(\sum_{j\in\mathcal{N}_i}\alpha_{ij}\cdot W\cdot h_j\right)",

    "sentiment_score = P(positive) − P(negative)":
        r"\text{sentiment\_score} = P(\text{positive}) - P(\text{negative})",

    "w_{t+1} = Σ over k of  (n_k / n)  *  w_{t+1}^k":
        r"w_{t+1} = \sum_k \frac{n_k}{n}\cdot w_{t+1}^k",

    "h_k(w, w_t) = L_k(w)  +  (μ/2) * ‖w - w_t‖²":
        r"h_k(w,\,w_t) = L_k(w) + \frac{\mu}{2}\left\|w - w_t\right\|^2",

    # ── Chapter 3 — Methodology ──────────────────────────────────────────────
    "Z(t) = (X(t) − rolling_mean(X, 252)) / rolling_std(X, 252)":
        r"Z(t) = \frac{X(t) - \text{rolling\_mean}(X,\,252)}{\text{rolling\_std}(X,\,252)}",

    "score = P(positive) − P(negative)":
        r"\text{score} = P(\text{positive}) - P(\text{negative})",

    "α_ij^r = softmax_j [ LeakyReLU( a_rᵀ [W_r*h_i ‖ W_r*h_j] ) ]":
        r"\alpha_{ij}^r = \text{softmax}_j\!\left[\text{LeakyReLU}\!\left("
        r"\mathbf{a}_r^T\left[W_r h_i \;\left\|\; W_r h_j\right.\right]\right)\right]",

    "h'_i^r = σ( Σ_j α_ij^r * W_r * h_j )":
        r"h'^{\,r}_i = \sigma\!\left(\sum_j \alpha_{ij}^r\cdot W_r\cdot h_j\right)",

    "h'_i = h'_i^sector ‖ h'_i^supply ‖ h'_i^correlation":
        r"h'_i = h'^{\text{sector}}_i \;\big\|\; h'^{\text{supply}}_i \;\big\|\; h'^{\text{correlation}}_i",

    "w_i = exp(a_i) / Σ over j of exp(a_j)":
        r"w_i = \frac{\exp(a_i)}{\displaystyle\sum_j \exp(a_j)}",

    "R(t) = Sharpe_rolling_30 − λ₁ * drawdown_penalty(t) − λ₂ * turnover_penalty(t)":
        r"R(t) = \text{Sharpe}_{\text{rolling,30}}"
        r"- \lambda_1\cdot\text{drawdown\_penalty}(t)"
        r"- \lambda_2\cdot\text{turnover\_penalty}(t)",

    "a_ensemble = (a_PPO + a_SAC + a_TD3 + a_A2C + a_DDPG) / 5":
        r"a_{\text{ensemble}} = \frac{"
        r"a_{\text{PPO}} + a_{\text{SAC}} + a_{\text{TD3}} + a_{\text{A2C}} + a_{\text{DDPG}}}{5}",

    "g_clipped = g * min(1, C / ‖g‖₂)":
        r"g_{\text{clipped}} = g\cdot\min\!\left(1,\;\frac{C}{\left\|g\right\|_2}\right)",

    "g_noisy = g_clipped + N(0, σ² * I)":
        r"g_{\text{noisy}} = g_{\text{clipped}} + \mathcal{N}\!\left(0,\;\sigma^2 I\right)",

    "minimize F_k(w) + (μ / 2) * ‖w − w_global‖²":
        r"\text{minimize}\quad F_k(w) + \frac{\mu}{2}\left\|w - w_{\text{global}}\right\|^2",

    # ── Chapter 4 — Implementation ───────────────────────────────────────────
    "Sharpe Ratio = (Annualized Portfolio Return − Risk-free Rate) / Annualized Standard Deviation of Returns":
        r"\text{Sharpe Ratio} = \frac{"
        r"\text{Annualized Portfolio Return} - R_f}{"
        r"\text{Annualized Standard Deviation}}",
}

# ── Metadata / instruction lines to strip ────────────────────────────────────
_SKIP_KW = [
    'Reference:', 'Status:', 'Target:', 'Word count:', 'Last updated:',
    'DISSERTATION_FORMATTING', 'prompt.md', 'Arabic page numbering',
    'page numbering from', 'update after all', 'update actual page', 'Annexure',
    'Writing rules:',
]


def _is_chapter(name: str) -> bool:
    return name.startswith(CHAPTER_PREFIXES)


def _is_math_line(text: str) -> bool:
    """Heuristic: does this line look like a math expression?"""
    math_chars = set('=∑∏∫αβγδεζηθλμνξπρστφχψω‖')
    return (
        any(c in text for c in math_chars)
        or bool(re.search(r'[_^]|exp\(|log\(|min\(|max\(|softmax|LeakyReLU|minimize', text))
    )


def preprocess(md_path: Path) -> str:
    """Pre-process a single MD file for pandoc."""
    is_chap = _is_chapter(md_path.name)
    lines   = md_path.read_text(encoding='utf-8', errors='replace').splitlines()
    out     = []

    for line in lines:
        stripped = line.strip()

        # Drop metadata / instruction lines
        if any(kw in stripped for kw in _SKIP_KW):
            continue

        # 4-space / tab indented lines
        if line.startswith('    ') or line.startswith('\t'):
            if is_chap and _is_math_line(stripped):
                latex = LATEX_MAP.get(stripped, stripped)
                out.append('')
                out.append(f'$$\n{latex}\n$$')
                out.append('')
            else:
                # Front matter centered text or reference URLs — plain paragraph
                out.append(stripped)
            continue

        out.append(line)

    return '\n'.join(out)


def convert(md_path: Path, docx_path: Path):
    content = preprocess(md_path)

    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.md', encoding='utf-8',
        delete=False, dir=BASE
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = subprocess.run(
            [
                PANDOC, str(tmp_path),
                '-o', str(docx_path),
                '--reference-doc', str(TEMPLATE),
                '--from', 'markdown+tex_math_dollars',
                '--columns', '999',
            ],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f'  OK   {docx_path.name}')
        else:
            print(f'  FAIL {md_path.name}: {result.stderr.strip()[:200]}')
    finally:
        tmp_path.unlink(missing_ok=True)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if not PANDOC:
        print("pandoc not found. Install: winget install pandoc   then restart terminal.")
        sys.exit(1)

    if not TEMPLATE.exists():
        print("Template missing. Run: python make_template.py")
        sys.exit(1)

    md_files = sorted(SRC.glob('*.md'))
    print(f"\nBuilding {len(md_files)} files -> {DST}\n")

    ok = fail = 0
    for md in md_files:
        docx = DST / (md.stem + '.docx')
        try:
            convert(md, docx)
            ok += 1
        except PermissionError:
            print(f'  SKIP {md.name}  (close it in Word first)')
            fail += 1
        except Exception as e:
            print(f'  ERR  {md.name}: {e}')
            fail += 1

    print(f'\nDone — {ok} OK, {fail} skipped/failed')
