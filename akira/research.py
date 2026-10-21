"""How each Core tier goes beyond Akira's shared knowledge.

    Standard   one quick, sure answer: the offline encyclopedia (instant), else one web lookup.
    Pro        a research agent: splits the question, then for each part looks in the encyclopedia,
               searches the web, reads the best pages and the user's own files (LAC), and writes
               an answer from the sentences that answer it, with numbered sources.
    Ultra      deeper research: more pages per part, a second search when the first is thin, and
               facts that two independent sources agree on are marked as confirmed.

Every step is reported as a tool event, so the user sees the agent work. Nothing here guesses:
answers are sentences taken from the sources, and each one says where it came from.
"""
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from . import web
from .knowledge import SENT, stem, terms, tidy

DEPTH = {   # rank: (web results to read per part, extra searches when thin, sentences in the answer)
    1: (3, 0, 4),
    2: (5, 1, 6),
}
RESEARCH = re.compile(r"\b(research|compare|comparison|versus|vs\.?|difference between|differences|pros and cons|"
                      r"advantages|disadvantages|why|how does|how do|explain|tell me (all )?about|overview|history of|"
                      r"summari[sz]e|look into|find out|investigate|sources?)\b", re.I)


def wants_research(text):
    """A question about the world (not about Akira or the user) or a request to look into something."""
    if web.PERSONAL.search(text) and not RESEARCH.search(text):
        return False
    return bool(RESEARCH.search(text) or web.QUESTION.match(text) or web.NEEDS_WEB.search(text))


def plan(question):
    """The parts of a question to look into: "compare X and Y" -> X, Y."""
    q = web.clean_query(question)
    m = re.search(r"(?i)\b(?:compare|difference(?:s)? between|versus)\s+(.+?)\s+(?:and|with|to|vs\.?|versus)\s+(.+)$", q) \
        or re.search(r"(?i)^(.+?)\s+(?:vs\.?|versus)\s+(.+)$", q)
    if m:
        return [m.group(1).strip(" ?.,"), m.group(2).strip(" ?.,")]
    parts = [p.strip(" ?.,") for p in re.split(r"(?i)\?\s+|\band also\b|;\s*", q) if len(p.strip(" ?.,")) > 3]
    return parts[:3] or [q]


JUNK = re.compile(r"redirects here|for other uses|disambiguation|this article|learn (the basics|more)|click here|"
                  r"cookies?|subscribe|sign up|all rights reserved|privacy policy|see also|\bmenu\b|^\W", re.I)
DEFINES = re.compile(r"\b[A-Z][a-z]+(?: [a-z]+)?(?: \([^)]*\))?,? (?:is|are|was|were|also known as)\b")
WHY = re.compile(r"\b(because|due to|caused by|causes?|which is why|so that|as a result|this means|results? in)\b", re.I)


def best_sentences(text, query, k=3, question=""):
    """The sentences of a page that best answer the query, in page order."""
    want = {stem(w) for w in terms(query)}
    if not want:
        return []
    explain = bool(re.match(r"(?i)\s*(why|how)\b", question))
    head = [stem(w) for w in terms(query)][:2]
    scored = []
    for i, s in enumerate(SENT.split(re.sub(r"\s+", " ", text))):
        s = re.sub(r"\[\d+\]|\[citation needed\]", "", s)
        s = re.sub(r"^(\d{1,2},\s*)?\d{4}\s+(key points\s+)?", "", s, flags=re.I)   # "10, 2024 Key points ..."
        start = DEFINES.search(s)
        if start and start.start() > 0 and head:
            lead = s[: start.start()]
            if head[0] in {stem(w) for w in terms(s[start.start():start.start() + 60])} and not re.search(r"[.,;:]\s", lead[-3:]):
                s = s[start.start():]                         # "Skip to main content ... Wind power is ..." -> "Wind power is ..."
        if not (40 <= len(s) <= 400) or len(s.split()) < 8 or JUNK.search(s) or not s.rstrip().endswith((".", "!")):
            continue
        have = {stem(w) for w in terms(s)}
        cover = len(want & have) / len(want)
        if cover >= 0.5:
            opening = [stem(w) for w in terms(s)][:2] == head        # "Wind power is ..." defines it
            score = cover + (0.3 if explain and WHY.search(s) else 0) + (0.2 if opening else 0) - 0.002 * i
            scored.append((score, i, tidy(s)))
    top = sorted(scored, reverse=True)[:k]
    return [(sc, s) for sc, _, s in sorted(top, key=lambda t: t[1])]


def _site(url):
    return urlparse(url).netloc.replace("www.", "")


def _read(url, query, question=""):
    try:
        page = web.read(url, max_chars=20000)
    except Exception:
        return None
    found = best_sentences(page["text"], query, question=question)
    return {"url": url, "title": page["title"] or _site(url), "sentences": found} if found else None


def run(question, rank, knowledge=None, lac=None, personal=False, use_web=True):
    """Research a question. Yields tool events, then {"type": "result", "text", "sources"} (or nothing found)."""
    reads, extra, keep = DEPTH.get(rank, DEPTH[1])
    parts = plan(question)
    yield {"type": "tool", "name": "plan", "input": {"parts": parts}, "where": "akira"}
    findings = []                                   # (part, sentence, source dict)
    for part in parts:
        if knowledge is not None and knowledge.available():
            yield {"type": "tool", "name": "encyclopedia", "input": {"query": part}, "where": "akira"}
            # named articles, except a plain lowercase word ("work" in "how does a vaccine work")
            named = [d for n, d in knowledge.subjects(part) if n > 1 or d["title"].lower() not in part]
            docs = (named + [d for d in knowledge.search(part, n=2) if d["title"] not in {x["title"] for x in named}])[:2]
            for doc in docs:
                for sc, s in best_sentences(doc["body"], part, k=2, question=question):
                    findings.append((part, s, {"title": doc["title"] + " (Simple Wikipedia)", "url": doc["url"]}, sc))
        if personal and lac is not None:
            files = lac.search_files(part)[:2]
            if files:
                yield {"type": "tool", "name": "search_files", "input": {"query": part}, "where": "device"}
                for f in files:
                    findings.append((part, f["summary"], {"title": f["name"] + " (your files)", "url": ""}, 1.5))
        queries = [question.strip() if len(parts) == 1 else part]   # search engines like whole questions
        seen = set()
        for round_ in range(1 + extra if use_web else 0):
            if round_ >= len(queries):
                break
            yield {"type": "tool", "name": "web_search", "input": {"query": queries[round_]}, "where": "internet"}
            try:
                results = [r for r in web.search(queries[round_], 6) if r["url"] not in seen][:reads]
            except Exception:
                results = []
            seen |= {r["url"] for r in results}
            for r in results:
                yield {"type": "tool", "name": "read_page", "input": {"url": r["url"]}, "where": "internet"}
            with ThreadPoolExecutor(max_workers=4) as pool:
                pages = [p for p in pool.map(lambda r: _read(r["url"], part, question), results) if p]
            for p in pages:
                for sc, s in p["sentences"]:
                    findings.append((part, s, {"title": p["title"], "url": p["url"]}, sc))
            if extra and len(pages) < 2:                   # thin: search again in other words
                queries.append(part + " explained" if len(part.split()) > 1 else "what is " + part)
    if not findings:
        return
    yield {"type": "result", **compose(question, parts, findings, keep, confirm=rank >= 2)}


def _overlap(a, b):
    return len(a & b) / max(1, min(len(a), len(b)))


def compose(question, parts, findings, keep, confirm=False):
    sources, index = [], {}

    def cite(src):
        key = src["url"] or src["title"]
        if key not in index:
            sources.append(src)
            index[key] = len(sources)
        return index[key]

    lines = []
    for part in parts:
        mine = [(s, src, sc) for p, s, src, sc in findings if p == part]
        if not mine:
            continue
        # rank: answers the part best, then agreed on by other sources (same key words, other site)
        scored = []
        for s, src, sc in mine:
            have = {stem(w) for w in terms(s)}
            agree = {o_src["url"] or o_src["title"] for o, o_src, _ in mine
                     if _site(o_src["url"] or o_src["title"]) != _site(src["url"] or src["title"])
                     and _overlap(have, {stem(w) for w in terms(o)}) >= 0.5}
            scored.append((sc + 0.3 * min(len(agree), 2), s, src, agree))
        scored.sort(key=lambda t: -t[0])
        picked, used = [], []
        for _, s, src, agree in scored:
            have = {stem(w) for w in terms(s)}
            if any(_overlap(have, u) >= 0.6 for u in used):
                continue                                    # says what an earlier line already said
            used.append(have)
            mark = f" [{cite(src)}]"
            if confirm and agree:
                mark += " ✓"
            picked.append(f"- {s}{mark}")
            if len(picked) >= keep:
                break
        if len(parts) > 1:
            lines.append(f"**{part[:1].upper() + part[1:]}**")
        lines += picked
    body = "\n".join(lines)
    refs = "\n".join(f"{i}. [{s['title']}]({s['url']})" if s["url"] else f"{i}. {s['title']}"
                     for i, s in enumerate(sources, 1))
    note = "\n\n✓ = confirmed by at least two independent sources." if confirm and "✓" in body else ""
    return {"text": f"{body}{note}\n\n**Sources**\n{refs}", "sources": [s["url"] for s in sources if s["url"]]}
