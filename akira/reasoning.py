"""Checking a brain's working: every calculation it writes is re-done exactly, and wrong results fixed.

A small brain can lay out the right steps for a word problem ("50 - 10 - 8 = 32 cookies") and
still slip on the arithmetic. Akira recomputes each "expression = result" step with the
calculator, carries corrected values into the later steps that use them, and fixes the
final "The answer is ..." line to match.
"""
import re

from .lac import calculate

NUM = r"\d[\d,]*(?:\.\d+)?"
STEP = re.compile(rf"((?:\$?{NUM}%?\s*[-+*/x×÷]\s*)+\$?{NUM}%?)\s*=\s*\$?({NUM})")
ANSWER = re.compile(rf"(The answer is:?\s*\$?)({NUM})", re.I)
MATHY = re.compile(r"\d.*\b(how (many|much|long|far|old)|what is the (total|cost|price|difference|sum|product)|"
                   r"altogether|in total|left|remain|each|per|times|percent|average)\b|"
                   r"\b(calculate|solve|work out)\b", re.I | re.S)


def is_word_problem(text):
    return len(re.findall(r"\d", text)) >= 2 and bool(MATHY.search(text))


def _fmt(value):
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value:,}" if isinstance(value, int) and abs(value) >= 10000 else str(value)


def check(text):
    """(fixed text, number of corrections)."""
    fixes = {}                                   # wrong result -> right result, to carry into later steps
    count = 0

    def redo(m):
        nonlocal count
        expr, claimed = m.group(1), m.group(2)
        for wrong, right in fixes.items():       # a step that uses an earlier wrong result
            expr = re.sub(rf"(?<![\d.,]){re.escape(wrong)}(?![\d.,])", right, expr)
        try:
            value = calculate(expr.replace("x", "*").replace("$", ""))
        except Exception:
            return m.group(0)
        right = _fmt(value)
        if right.replace(",", "") != claimed.replace(",", ""):
            fixes[claimed] = right
            count += 1
            return f"{expr} = {right}"
        return f"{expr} = {claimed}"

    out = re.sub(r"\n?#### [^\n]*", "", text)             # a stray dataset answer marker
    out = STEP.sub(redo, out)
    if fixes:
        out = ANSWER.sub(lambda m: m.group(1) + fixes.get(m.group(2), m.group(2)), out)
    return out, count
