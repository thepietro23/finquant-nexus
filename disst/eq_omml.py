"""
eq_omml.py — Convert math equation text to OMML for Word equation objects.

Handles: fractions, subscripts, superscripts, norms, parentheses, square
brackets, Greek letters, function names, and operator runs.

Strategy: recursive descent on a character-level scanner that tracks bracket
depth so that '/' and '=' splits are only done at depth 0.
"""

from lxml import etree
import re

# ── Namespaces ────────────────────────────────────────────────────────────────
M_NS   = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
XML_NS = 'http://www.w3.org/XML/1998/namespace'

# ── Greek / symbol map ────────────────────────────────────────────────────────
GREEK_MAP = {
    'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'epsilon': 'ε',
    'zeta': 'ζ', 'eta': 'η', 'theta': 'θ', 'iota': 'ι', 'kappa': 'κ',
    'lambda': 'λ', 'mu': 'μ', 'nu': 'ν', 'xi': 'ξ', 'pi': 'π', 'rho': 'ρ',
    'sigma': 'σ', 'tau': 'τ', 'upsilon': 'υ', 'phi': 'φ', 'chi': 'χ',
    'psi': 'ψ', 'omega': 'ω',
    'Alpha': 'Α', 'Beta': 'Β', 'Gamma': 'Γ', 'Delta': 'Δ', 'Theta': 'Θ',
    'Lambda': 'Λ', 'Mu': 'Μ', 'Pi': 'Π', 'Sigma': 'Σ', 'Omega': 'Ω',
}

# Characters that should always render as non-italic (upright)
NON_ITALIC_CHARS = set('=+−-×÷·∓±≠≤≥<>∈∉⊂⊆∪∩→←↔⇒⇔∀∃∑∏∫∂∇.,;:!|‖ ')

# Known function names → render upright
FUNC_NAMES = {
    'exp', 'log', 'ln', 'sin', 'cos', 'tan', 'min', 'max', 'sqrt',
    'softmax', 'clip', 'argmax', 'argmin', 'det', 'tr', 'diag',
    'LeakyReLU', 'ReLU', 'sigmoid', 'tanh', 'E', 'Var', 'Pr',
    'minimize', 'maximize', 'mean', 'std', 'sum',
}

# Words that are operator-like (non-italic)
OP_WORDS = {'over', 'of', 'in', 'where', 'and', 'or', 'not', 'with'}


# ── XML helpers ───────────────────────────────────────────────────────────────

def _el(tag):
    return etree.Element(f'{{{M_NS}}}{tag}')


def _sub_el(parent, tag):
    return etree.SubElement(parent, f'{{{M_NS}}}{tag}')


def _run(text, italic=True):
    """Create m:r with given text. italic=False for operators/numbers/functions."""
    r = _el('r')
    rpr = _sub_el(r, 'rPr')
    sty = _sub_el(rpr, 'sty')
    sty.set(f'{{{M_NS}}}val', 'i' if italic else 'p')
    t = _sub_el(r, 't')
    t.text = text
    if text and (text[0] == ' ' or text[-1] == ' '):
        t.set(f'{{{XML_NS}}}space', 'preserve')
    return r


def _frac(num_elements, den_elements):
    f = _el('f')
    num = _sub_el(f, 'num')
    for el in num_elements:
        num.append(el)
    den = _sub_el(f, 'den')
    for el in den_elements:
        den.append(el)
    return f


def _ssub(base_elements, sub_elements):
    s = _el('sSub')
    _sub_el(s, 'sSubPr')
    e = _sub_el(s, 'e')
    for el in base_elements:
        e.append(el)
    sub = _sub_el(s, 'sub')
    for el in sub_elements:
        sub.append(el)
    return s


def _ssup(base_elements, sup_elements):
    s = _el('sSup')
    _sub_el(s, 'sSupPr')
    e = _sub_el(s, 'e')
    for el in base_elements:
        e.append(el)
    sup = _sub_el(s, 'sup')
    for el in sup_elements:
        sup.append(el)
    return s


def _ssubsup(base_els, sub_els, sup_els):
    s = _el('sSubSup')
    _sub_el(s, 'sSubSupPr')
    e = _sub_el(s, 'e')
    for el in base_els:
        e.append(el)
    sb = _sub_el(s, 'sub')
    for el in sub_els:
        sb.append(el)
    sp = _sub_el(s, 'sup')
    for el in sup_els:
        sp.append(el)
    return s


def _bracket(inner_elements, beg='(', end=')'):
    d = _el('d')
    dpr = _sub_el(d, 'dPr')
    bc = _sub_el(dpr, 'begChr')
    bc.set(f'{{{M_NS}}}val', beg)
    ec = _sub_el(dpr, 'endChr')
    ec.set(f'{{{M_NS}}}val', end)
    e = _sub_el(d, 'e')
    for el in inner_elements:
        e.append(el)
    return d


# ── Greek substitution ────────────────────────────────────────────────────────

def _apply_greek(text):
    """Replace ASCII Greek names with Unicode symbols."""
    for name, sym in GREEK_MAP.items():
        text = re.sub(rf'\b{name}\b', sym, text)
    return text


# ── Bracket-depth utilities ────────────────────────────────────────────────────

OPEN_BRACKETS  = {'(', '[', '{'}
CLOSE_BRACKETS = {')', ']', '}'}
NORM_CHAR = '‖'


def _bracket_depth(text, pos):
    """Return bracket depth at position pos (after processing text[:pos])."""
    depth = 0
    norm_count = 0
    for i, ch in enumerate(text):
        if i == pos:
            break
        if ch in OPEN_BRACKETS:
            depth += 1
        elif ch in CLOSE_BRACKETS:
            depth -= 1
        elif ch == NORM_CHAR:
            norm_count += 1
    # Each pair of ‖ is a norm bracket; odd norm_count means we're inside one
    return depth + (norm_count % 2)


def _find_top_level(text, char):
    """Find index of first occurrence of char at bracket depth 0. Returns -1 if not found."""
    depth = 0
    norm_open = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in OPEN_BRACKETS:
            depth += 1
        elif ch in CLOSE_BRACKETS:
            depth -= 1
        elif ch == NORM_CHAR:
            norm_open = not norm_open
            i += 1
            continue
        if depth == 0 and not norm_open and ch == char:
            return i
        i += 1
    return -1


def _find_all_top_level(text, char):
    """Find all indices of char at bracket depth 0."""
    indices = []
    depth = 0
    norm_open = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in OPEN_BRACKETS:
            depth += 1
        elif ch in CLOSE_BRACKETS:
            depth -= 1
        elif ch == NORM_CHAR:
            norm_open = not norm_open
            i += 1
            continue
        if depth == 0 and not norm_open and ch == char:
            indices.append(i)
        i += 1
    return indices


def _split_top_level(text, char):
    """Split text on char at depth 0. Returns list of parts."""
    indices = _find_all_top_level(text, char)
    if not indices:
        return [text]
    parts = []
    prev = 0
    for idx in indices:
        parts.append(text[prev:idx])
        prev = idx + len(char)
    parts.append(text[prev:])
    return parts


def _find_matching_close(text, open_pos, open_ch, close_ch):
    """Find the closing bracket that matches open_ch at open_pos."""
    depth = 0
    i = open_pos
    while i < len(text):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _find_matching_norm(text, first_norm_pos):
    """Find the second ‖ after first_norm_pos."""
    i = first_norm_pos + 1
    while i < len(text):
        if text[i] == NORM_CHAR:
            return i
        i += 1
    return -1


# ── Token scanner ─────────────────────────────────────────────────────────────
# We parse equation text using a segment-based approach.
# A "segment" is one of:
#   - a bracketed group: (...), [...], {}, ‖...‖
#   - a fraction marker (top-level /)
#   - a subscript (_...) or superscript (^...) attached to prior base
#   - a word/identifier token
#   - an operator character


def _is_variable_char(ch):
    """True if character could be part of a variable/identifier name."""
    return ch.isalnum() or ch in ("'", '_', 'α', 'β', 'γ', 'δ', 'ε', 'ζ',
                                   'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ',
                                   'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ',
                                   'ω', 'Α', 'Β', 'Γ', 'Δ', 'Θ', 'Λ', 'Μ',
                                   'Π', 'Σ', 'Ω', '²', '³', 'ᵀ', '⁻', '¹',
                                   '₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇',
                                   '₈', '₉')


# ── Main parser ───────────────────────────────────────────────────────────────

def _parse(text: str) -> list:
    """
    Parse equation text into a list of OMML elements.
    This is the main recursive entry point.
    """
    text = _apply_greek(text.strip())
    if not text:
        return []

    # Step 1: Split on top-level '=' signs
    eq_parts = _split_top_level(text, '=')
    if len(eq_parts) > 1:
        # Re-join with '=' and build each part, interleaving '=' runs
        result = []
        for idx, part in enumerate(eq_parts):
            result.extend(_parse_no_eq(part.strip()))
            if idx < len(eq_parts) - 1:
                result.append(_run(' = ', italic=False))
        return result

    return _parse_no_eq(text)


def _parse_no_eq(text: str) -> list:
    """Parse text that has no top-level '=' (already split)."""
    text = text.strip()
    if not text:
        return []

    # Step 2: Split on top-level '/' for fractions
    slash_indices = _find_all_top_level(text, '/')
    if slash_indices:
        # Use the first top-level slash as fraction divider
        idx = slash_indices[0]
        num_text = text[:idx].strip()
        den_text = text[idx + 1:].strip()
        # Only make a fraction if both sides are non-trivial
        if num_text and den_text:
            num_els = _parse_flat(num_text)
            den_els = _parse_flat(den_text)
            return [_frac(num_els, den_els)]

    return _parse_flat(text)


def _parse_flat(text: str) -> list:
    """
    Parse text without top-level '=' or '/'.
    Handles: brackets, norms, subscripts, superscripts, words, operators.
    """
    text = text.strip()
    if not text:
        return []

    result = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        # ── Norm brackets ‖...‖ ──
        if ch == NORM_CHAR:
            close = _find_matching_norm(text, i)
            if close != -1:
                inner = text[i + 1:close].strip()
                inner_els = _parse_flat(inner)
                result.append(_bracket(inner_els, '‖', '‖'))
                i = close + 1
                continue
            else:
                result.append(_run(ch, italic=False))
                i += 1
                continue

        # ── Parentheses (...)  ──
        if ch == '(':
            close = _find_matching_close(text, i, '(', ')')
            if close != -1:
                inner = text[i + 1:close]
                # Check if preceded by a function name
                func_name = _pop_last_func(result)
                inner_els = _parse_no_eq(inner)
                bracket_el = _bracket(inner_els, '(', ')')
                if func_name:
                    # function name run + bracket
                    result.append(_run(func_name, italic=False))
                    result.append(bracket_el)
                else:
                    result.append(bracket_el)
                i = close + 1
                # After closing paren, check for _ or ^
                i = _consume_sub_sup(text, i, result)
                continue
            else:
                result.append(_run(ch, italic=False))
                i += 1
                continue

        # ── Square brackets [...] ──
        if ch == '[':
            close = _find_matching_close(text, i, '[', ']')
            if close != -1:
                inner = text[i + 1:close]
                inner_els = _parse_no_eq(inner)
                result.append(_bracket(inner_els, '[', ']'))
                i = close + 1
                continue
            else:
                result.append(_run(ch, italic=False))
                i += 1
                continue

        # ── Curly braces (used for grouping in sub/sup) — strip them ──
        if ch == '{':
            close = _find_matching_close(text, i, '{', '}')
            if close != -1:
                inner = text[i + 1:close]
                inner_els = _parse_flat(inner)
                result.extend(inner_els)
                i = close + 1
                continue
            else:
                i += 1
                continue

        if ch == '}':
            i += 1
            continue

        # ── Subscript _ ──
        if ch == '_':
            base_els = _pop_base(result)
            i += 1
            sub_text, i = _read_script_arg(text, i)
            sub_els = _parse_flat(sub_text)
            # Look ahead for ^ (combined sub+sup)
            if i < n and text[i] == '^':
                i += 1
                sup_text, i = _read_script_arg(text, i)
                sup_els = _parse_flat(sup_text)
                result.append(_ssubsup(base_els, sub_els, sup_els))
            else:
                result.append(_ssub(base_els, sub_els))
            continue

        # ── Superscript ^ ──
        if ch == '^':
            base_els = _pop_base(result)
            i += 1
            sup_text, i = _read_script_arg(text, i)
            sup_els = _parse_flat(sup_text)
            # Look ahead for _ (combined sub+sup)
            if i < n and text[i] == '_':
                i += 1
                sub_text, i = _read_script_arg(text, i)
                sub_els = _parse_flat(sub_text)
                result.append(_ssubsup(base_els, sub_els, sup_els))
            else:
                result.append(_ssup(base_els, sup_els))
            continue

        # ── Whitespace → preserve as operator space ──
        if ch == ' ':
            # Collect run of spaces
            j = i
            while j < n and text[j] == ' ':
                j += 1
            result.append(_run(' ', italic=False))
            i = j
            continue

        # ── Operators and special single chars ──
        if ch in NON_ITALIC_CHARS or ch in ('−', '*', '·', '∑', '∏', '∫'):
            # Collect consecutive operator chars
            j = i
            while j < n and text[j] in NON_ITALIC_CHARS and text[j] not in (' ',):
                j += 1
            op_text = text[i:j]
            result.append(_run(op_text, italic=False))
            i = j
            continue

        # ── Word / identifier token ──
        if ch.isalpha() or ch in ('α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ',
                                   'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'π', 'ρ',
                                   'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',
                                   'Α', 'Β', 'Γ', 'Δ', 'Θ', 'Λ', 'Μ', 'Π',
                                   'Σ', 'Ω'):
            j = i
            while j < n and _is_variable_char(text[j]) and text[j] not in ('_', '^'):
                j += 1
            word = text[i:j]
            is_func = word in FUNC_NAMES
            is_op_word = word in OP_WORDS
            result.append(_run(word, italic=(not is_func and not is_op_word)))
            i = j
            # Immediately handle _ and ^ attached to this word
            if i < n and (text[i] == '_' or text[i] == '^'):
                i = _consume_sub_sup(text, i, result)
            continue

        # ── Number ──
        if ch.isdigit() or ch == '.':
            j = i
            while j < n and (text[j].isdigit() or text[j] == '.'):
                j += 1
            result.append(_run(text[i:j], italic=False))
            i = j
            continue

        # ── Superscript/subscript Unicode chars attached to prior token ──
        if ch in ('²', '³', 'ᵀ', '⁻', '¹', '₀', '₁', '₂', '₃', '₄',
                  '₅', '₆', '₇', '₈', '₉'):
            # These are already Unicode sup/sub; just emit as a run
            # They look better as part of the preceding run if possible
            j = i
            while j < n and text[j] in ('²', '³', 'ᵀ', '⁻', '¹', '₀', '₁',
                                          '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'):
                j += 1
            result.append(_run(text[i:j], italic=False))
            i = j
            continue

        # ── Anything else: emit as-is ──
        result.append(_run(ch, italic=False))
        i += 1

    return result


def _read_script_arg(text, i):
    """
    Read the argument for _ or ^.
    If next char is '{', read until matching '}'. Otherwise read one token.
    Returns (arg_text, new_i).
    """
    n = len(text)
    if i >= n:
        return ('', i)

    if text[i] == '{':
        close = _find_matching_close(text, i, '{', '}')
        if close != -1:
            arg = text[i + 1:close]
            return (arg, close + 1)
        else:
            return ('', i + 1)

    # Read one "token": alnum run, or a single special char
    j = i
    if text[j] in OPEN_BRACKETS:
        close_map = {'(': ')', '[': ']', '{': '}'}
        close_ch = close_map[text[j]]
        close = _find_matching_close(text, j, text[j], close_ch)
        if close != -1:
            arg = text[j:close + 1]
            return (arg, close + 1)

    # Alnum token
    while j < n and _is_variable_char(text[j]) and text[j] not in ('_', '^', ' ', '('):
        j += 1
    if j == i:
        j = i + 1  # at least consume one char
    return (text[i:j], j)


def _pop_base(result):
    """
    Pop the last element from result to use as base for sub/sup.
    If result is empty, return a run with empty text.
    """
    if result:
        return [result.pop()]
    return [_run('', italic=True)]


def _pop_last_func(result):
    """
    If the last element in result is a plain run whose text is a function name,
    pop it and return the function name string. Otherwise return None.
    """
    if not result:
        return None
    last = result[-1]
    # Check if it's an m:r element
    tag = last.tag.split('}')[-1] if '}' in last.tag else last.tag
    if tag != 'r':
        return None
    # Get text content
    t_els = last.findall(f'{{{M_NS}}}t')
    if not t_els:
        return None
    text = t_els[0].text or ''
    if text.strip() in FUNC_NAMES:
        result.pop()
        return text.strip()
    return None


def _consume_sub_sup(text, i, result):
    """
    Starting at position i in text, consume any leading _ or ^ sequences
    and attach them to the last element in result.
    Returns new i.
    """
    n = len(text)
    while i < n and text[i] in ('_', '^'):
        ch = text[i]
        base_els = _pop_base(result)
        i += 1
        arg_text, i = _read_script_arg(text, i)
        arg_els = _parse_flat(arg_text)
        if ch == '_':
            # Check if next is ^
            if i < n and text[i] == '^':
                i += 1
                sup_text, i = _read_script_arg(text, i)
                sup_els = _parse_flat(sup_text)
                result.append(_ssubsup(base_els, arg_els, sup_els))
            else:
                result.append(_ssub(base_els, arg_els))
        else:  # '^'
            if i < n and text[i] == '_':
                i += 1
                sub_text, i = _read_script_arg(text, i)
                sub_els = _parse_flat(sub_text)
                result.append(_ssubsup(base_els, sub_els, arg_els))
            else:
                result.append(_ssup(base_els, arg_els))
    return i


# ── Public API ─────────────────────────────────────────────────────────────────

def build_omml_para(eq_text: str):
    """
    Main entry point. Takes equation text (may already have Unicode symbols),
    returns complete m:oMathPara XML element ready to insert into a Word
    paragraph's _p element.
    """
    para = _el('oMathPara')
    math = _sub_el(para, 'oMath')
    elements = _parse(eq_text.strip())
    for el in elements:
        math.append(el)
    return para


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    TEST_EQUATIONS = [
        "h_k(w, w_t) = L_k(w)  +  (μ/2) * ‖w - w_t‖²",
        "g_clipped = g * min(1, C / ‖g‖₂)",
        "g_noisy = g_clipped + N(0, σ² * I)",
        "minimize F_k(w) + (μ / 2) * ‖w − w_global‖²",
        "w_i = exp(a_i) / Σ over j of exp(a_j)",
        "R(t) = Sharpe_rolling_30 − λ₁ * drawdown_penalty(t) − λ₂ * turnover_penalty(t)",
        "a_ensemble = (a_PPO + a_SAC + a_TD3 + a_A2C + a_DDPG) / 5",
        "Sharpe Ratio = (Annualized Portfolio Return − Risk-free Rate) / Annualized Standard Deviation of Returns",
        "Z(t) = (X(t) − rolling_mean(X, 252)) / rolling_std(X, 252)",
        "L_CLIP = E[ min( r_t(θ) * A_t ,  clip(r_t(θ), 1-ε, 1+ε) * A_t ) ]",
        "e_ij = LeakyReLU( aᵀ [ W * h_i  ‖  W * h_j ] )",
        "α_ij = exp(e_ij) / Σ over k in N_i of exp(e_ik)",
        "h'_i = σ ( Σ over j in N_i of α_ij * W * h_j )",
        "sentiment_score = P(positive) − P(negative)",
        "w_{t+1} = Σ over k of  (n_k / n)  *  w_{t+1}^k",
        "α_ij^r = softmax_j [ LeakyReLU( a_rᵀ [W_r*h_i ‖ W_r*h_j] ) ]",
        "h'_i^r = σ( Σ_j α_ij^r * W_r * h_j )",
        "h'_i = h'_i^sector ‖ h'_i^supply ‖ h'_i^correlation",
        "score = P(positive) − P(negative)",
    ]

    ok = 0
    fail = 0
    for eq in TEST_EQUATIONS:
        try:
            para = build_omml_para(eq)
            xml_str = etree.tostring(para, pretty_print=False)
            assert len(xml_str) > 50
            print(f"  OK  {eq[:70]}")
            ok += 1
        except Exception as exc:
            print(f"  FAIL {eq[:70]}\n       {exc}")
            fail += 1

    print(f"\n{ok} passed, {fail} failed")
