"""Akira's shared knowledge: an offline encyclopedia every Core tier answers from.

Built once from Simple English Wikipedia (data/raw/simplewiki.parquet) into a SQLite full-text
index (models/knowledge.db), so common-knowledge questions get an accurate answer instantly,
without the internet and without a brain guessing:

    python3 -m akira.knowledge build

Standard, Pro and Ultra all read the same index, so they know the same facts; what differs
is how far each goes beyond it (see akira/research.py).
"""
import os
import re
import sqlite3
import sys
import threading


DB = os.environ.get("AKIRA_KNOWLEDGE", os.path.join(os.environ.get("AKIRA_MODELS", "models"), "knowledge.db"))
STOP = set("""a an the of in on at to for from by with about as is are was were be been being do does did what who whom
whose when where which why how much many tell me please can could would you explain describe define meaning mean
it its this that these those and or but if than then there their they he she his her i my we our your called name
named give show find know some any into out up down over under again more most other such only own same so too very
just also s t don isn aren wasn weren doesn didn has have had having will shall may might must""".split())
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")
BAD = re.compile(r"may refer to|This article is about|can mean:|disambiguation", re.I)
GAP = re.compile(r"\s{2,}|\(\s*[,;)]|\s[,.;]|\bby ,|is about  ")   # values missing from the dump: "about  high"
NUMERIC = re.compile(r"^\s*(how (many|much|tall|high|long|far|old|big|fast|deep|heavy)|when|what year|in what year)\b", re.I)
NUMBER = re.compile(r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|"
                    r"hundred|thousand|million|billion|dozen)\b", re.I)
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


IRREGULAR = {"wrote": "write", "written": "write", "writer": "write", "author": "write", "spoken": "speak",
             "spoke": "speak", "language": "speak", "languages": "speak", "built": "build", "made": "make",
             "began": "begin", "begun": "begin", "ended": "end", "found": "find", "founded": "found",
             "height": "tall", "high": "tall", "taller": "tall", "born": "birth", "died": "death", "dead": "death"}


def stem(word):
    """A rough root, the same on both sides of a comparison: invented/inventor -> invent, largest -> larg."""
    w = IRREGULAR.get(word.lower(), word.lower())
    for suffix in ("ations", "ation", "ings", "ing", "ors", "ed", "or", "es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[: -len(suffix)]
    return w


UNIT = re.compile(r"\b(metres?|meters?|feet|foot|ft|km|kilometres?|kilometers?|miles?|inches|cm|m)\b|°", re.I)
WHEN = re.compile(r"\b(1\d{3}|20\d\d|\d{3}\b|century|BC|AD)\b")
NAME = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|de|da|van|von))+")


RANKED = re.compile(r"\b(second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(st|nd|rd|th))[ -](largest|longest|"
                    r"biggest|smallest|highest|tallest|oldest|most|deepest)", re.I)


def answer_kind(question):
    """What a good answer must contain, by the kind of question; None when anything goes."""
    q = question.strip().lower()
    asked = {w.lower() for w in WORD.findall(question)}
    if re.match(r"(who|whom)\b", q) and not re.match(r"(?i:who (is|was|were|are)) [A-Z]", question.strip()):
        # "Who painted the Mona Lisa?" needs a person's name the question doesn't already contain
        return lambda s, doc: any(not set(m.lower().split()) <= asked for m in NAME.findall(s))
    many = re.match(r"how many ([a-z]+)", q)
    if many:
        noun = stem(many.group(1))[:5]
        pat = re.compile(r"\b(\d[\d,]*|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                         r"twenty|thirty|forty|fifty|hundred|thousand|million)\s+(\w+\s+)?" + re.escape(noun), re.I)
        return lambda s, doc: bool(pat.search(s))
    if re.match(r"how (tall|high|long|far|deep|big|wide)\b", q):
        return lambda s, doc: bool(UNIT.search(s)) and bool(re.search(r"\d", s))
    if re.match(r"(when|what year|in what year)\b", q):
        return lambda s, doc: bool(WHEN.search(s))
    if re.match(r"(what|which) is the (boiling|freezing|melting) point", q):
        return lambda s, doc: bool(re.search(r"\d", s))
    return None


def tidy(text):
    """Remove the husks the dump leaves where pronunciations and values were: "France ( or ; )"."""
    text = re.sub(r"\(\s*(?:or\s*)?[,;]?\s*(?:or\s*)?[,;]?\s*\)", "", text)
    return re.sub(r"\s{2,}", " ", text).replace(" ,", ",").strip()


def terms(text):
    """The words of a question that carry its meaning."""
    return [w for w in (x.lower().strip("'-") for x in WORD.findall(text)) if w and w not in STOP]


def build(parquet="data/raw/simplewiki.parquet", path=DB, max_chars=3000):
    import pyarrow.parquet as pq
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    con = sqlite3.connect(tmp)
    con.execute("CREATE VIRTUAL TABLE docs USING fts5(title, body, tokenize='porter unicode61')")
    con.execute("CREATE TABLE titles(t TEXT PRIMARY KEY, id INTEGER)")   # exact article names, lowercased
    rows = 0
    for batch in pq.ParquetFile(parquet).iter_batches(columns=["title", "text"], batch_size=5000):
        out = []
        for title, text in zip(batch.column("title").to_pylist(), batch.column("text").to_pylist()):
            paras = [p.strip() for p in (text or "").split("\n") if len(p.strip()) > 40]
            body = ""
            for p in paras:                            # the lead and what follows, up to max_chars
                if len(body) + len(p) > max_chars and body:
                    break
                body += p + "\n"
            if len(body) > 60:
                out.append((title, body.strip()))
        for title, body in out:
            cur = con.execute("INSERT INTO docs VALUES (?, ?)", (title, body))
            con.execute("INSERT OR IGNORE INTO titles VALUES (?, ?)", (title.lower(), cur.lastrowid))
        rows += len(out)
        print(f"{rows:,} articles", flush=True)
    con.commit()
    con.execute("INSERT INTO docs(docs) VALUES ('optimize')")
    con.commit()
    con.close()
    os.replace(tmp, path)
    return rows


class Knowledge:
    def __init__(self, path=DB):
        self.path = path
        self._local = threading.local()

    def available(self):
        return os.path.exists(self.path)

    def _con(self):
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
            self._local.con = con
        return con

    def search(self, query, n=5):
        """Articles about the query, best first: [{"title", "body", "url"}]."""
        words = terms(query)
        if not words or not self.available():
            return []
        quoted = [f'"{w}"' for w in words]
        sql = "SELECT title, body, bm25(docs, 8.0, 1.0) AS r FROM docs WHERE docs MATCH ? ORDER BY r LIMIT ?"
        rows = self._con().execute(sql, (" AND ".join(quoted), n * 3)).fetchall()   # articles with every word first
        got = {r[0] for r in rows}
        rows = [(t, b, r - 100) for t, b, r in rows]
        rows += [r for r in self._con().execute(sql, (" OR ".join(quoted), n * 3)).fetchall() if r[0] not in got]
        # an article whose title is a word of the question is usually what it's about
        want = {stem(w) for w in words}

        def rank(row):
            title_words = {stem(w) for w in terms(row[0])}
            covered = len(title_words & want) / max(1, len(title_words))
            return row[2] - 6.0 * covered
        rows.sort(key=rank)
        return [{"title": t, "body": b, "url": "https://simple.wikipedia.org/wiki/" + t.replace(" ", "_")}
                for t, b, _ in rows[:n]]

    def by_title(self, name):
        row = self._con().execute("SELECT d.title, d.body FROM titles t JOIN docs d ON d.rowid = t.id WHERE t.t = ?",
                                  (name.lower(),)).fetchone()
        if row is None:
            return None
        return {"title": row[0], "body": row[1], "url": "https://simple.wikipedia.org/wiki/" + row[0].replace(" ", "_")}

    def subjects(self, question):
        """Articles named in the question ("Albert Einstein", "Romeo and Juliet"), longest names first."""
        toks = WORD.findall(question)
        found, used = [], set()
        for n in range(min(5, len(toks)), 0, -1):
            for i in range(len(toks) - n + 1):
                span = set(range(i, i + n))
                gram = toks[i:i + n]
                if span & used or all(t.lower() in STOP for t in gram) or gram[0].lower() in STOP or gram[-1].lower() in STOP:
                    continue
                doc = self.by_title(" ".join(gram))
                if doc and not BAD.search(doc["body"][:200]):
                    found.append((n, doc))
                    used |= span
        return found

    def answer(self, question, max_sentences=2):
        """(answer, source) when the encyclopedia clearly answers the question, else (None, None).

        First the article the question is about (named in it, like "Mount Everest"), and in it the
        sentence that covers the rest of the question. Failing that, any top article with one
        sentence that covers the whole question. Otherwise no answer: better none than a wrong one.
        """
        words = terms(question)
        if not words or not self.available():
            return None, None
        want = {stem(w) for w in words}
        seq = [stem(w) for w in words]
        pairs = set(zip(seq, seq[1:]))                      # "largest planet", "capital france"
        number = bool(NUMERIC.match(question))
        kind = answer_kind(question)
        for _, doc in self.subjects(question):
            named = {stem(w) for w in terms(doc["title"])} & want
            rest = want - named
            hit = self._best_sentence(doc, rest, number, need=0.5, pairs=pairs, kind=kind)
            if hit is not None:
                return self._excerpt(doc, hit, max_sentences), doc
        best = None
        for rank, doc in enumerate(self.search(question, n=10)):
            if BAD.search(doc["body"][:200]) or doc["title"].startswith("List of"):
                continue
            hit = self._best_sentence(doc, want, number, need=1.0, pairs=pairs, kind=kind)
            if hit is not None and pairs and hit[2] < 1.0:
                continue                                    # every word, and in the question's order
            if hit is not None and (best is None or (hit[0], -rank) > (best[0][0], -best[2])):
                best = (hit, doc, rank)
        if best is None:
            return None, None
        return self._excerpt(best[1], best[0], max_sentences), best[1]

    @staticmethod
    def _sentences(doc):
        return SENT.split(doc["body"].replace("\n", " "))

    def _best_sentence(self, doc, rest, number, need, pairs=(), kind=None):
        best = None
        for i, sent in enumerate(self._sentences(doc)):
            if len(sent) < 25 or (number and (GAP.search(sent) or not NUMBER.search(sent))):
                continue                                    # a number question needs a number that survived the dump
            if RANKED.search(sent) and not RANKED.search(" ".join(rest)):
                continue                                    # "the tenth longest river" doesn't answer "the longest"
            if kind is not None and not kind(sent, doc):
                continue                                    # the wrong kind of answer: "who" without a name
            seq = [stem(w) for w in terms(sent)]
            if rest:
                cover = len(rest & set(seq)) / len(rest)
            else:
                cover = 1.0 if i == 0 else 0.0               # "What is DNA?": the opening sentence
            if cover < need:
                continue
            adjacent = set(zip(seq, seq[1:]))
            phrase = sum(p in adjacent for p in pairs) / len(pairs) if pairs else 0.0
            score = cover + 0.6 * phrase + (0.2 if i == 0 else 0) - 0.02 * i
            if best is None or score > best[0]:
                best = (score, i, phrase)
        return best

    def _excerpt(self, doc, hit, max_sentences):
        sents = self._sentences(doc)[hit[1]:hit[1] + max_sentences]
        return tidy(" ".join(sents))

if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        print(f"built {DB}: {build():,} articles")
    else:
        k = Knowledge()
        for q in sys.argv[1:] or ["What is the capital of France?"]:
            print(q, "->", k.answer(q))
