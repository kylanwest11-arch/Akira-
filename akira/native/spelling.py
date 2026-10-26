"""Turn what Live hears into real words.

Live writes letters by sound ("swleader", "elapst", "ten minidset"). This snaps each heard word
to the closest real word, weighing how close the spelling is, how common the word is, and which
word usually follows the previous one; and it splits run-together words ("tenminutes"). The
word list and word pairs come from text Akira already has (Live's transcripts by default).

    speller = Speller.from_texts(lines)
    speller.correct("ias eistdr swleader")
"""
import math
import re
from collections import Counter


def _edits1(w):
    letters = "abcdefghijklmnopqrstuvwxyz'"
    splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    return ({a + b[1:] for a, b in splits if b}
            | {a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1}
            | {a + c + b[1:] for a, b in splits if b for c in letters}
            | {a + c + b for a, b in splits for c in letters})


def _distance(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(b)]


class Speller:
    def __init__(self, counts, pairs):
        self.counts, self.pairs = counts, pairs
        self.total = sum(counts.values())
        self.by_skeleton = {}                     # words grouped by consonant outline, for distant matches
        for w in counts:
            self.by_skeleton.setdefault(self._skeleton(w), []).append(w)

    @classmethod
    def from_texts(cls, texts, min_count=2):
        counts, pairs = Counter(), Counter()
        for t in texts:
            ws = re.findall(r"[a-z']+", t.lower())
            counts.update(ws)
            pairs.update(zip(ws, ws[1:]))
        counts = Counter({w: c for w, c in counts.items() if c >= min_count})
        return cls(counts, pairs)

    @staticmethod
    def _skeleton(w):
        return re.sub(r"[aeiouy']", "", w)[:3]

    def _candidates(self, w):
        if w in self.counts:
            yield w, 0
        near = {c for c in _edits1(w) if c in self.counts}
        for c in near:
            yield c, 1
        if len(w) >= 4:
            for c in self.by_skeleton.get(self._skeleton(w), []):
                if c not in near and abs(len(c) - len(w)) <= 2:
                    d = _distance(w, c)
                    if d <= max(2, len(w) // 3):
                        yield c, d

    def _score(self, cand, dist, prev):
        unigram = math.log((self.counts[cand] + 0.1) / self.total)
        bigram = math.log((self.pairs.get((prev, cand), 0) + 0.05) / (self.counts.get(prev, 0) + 1)) if prev else 0.0
        return -2.2 * dist + 0.35 * unigram + 0.6 * bigram

    def correct(self, text):
        out, prev = [], None
        for w in text.split():
            best = max(self._candidates(w), key=lambda cd: self._score(cd[0], cd[1], prev), default=None)
            if best is None and len(w) >= 6:          # two words run together: "tenminutes"
                splits = [(w[:i], w[i:]) for i in range(2, len(w) - 1)
                          if w[:i] in self.counts and w[i:] in self.counts]
                if splits:
                    a, b = max(splits, key=lambda s: self.counts[s[0]] * self.counts[s[1]])
                    out += [a, b]
                    prev = b
                    continue
            word = best[0] if best else w
            out.append(word)
            prev = word
        return " ".join(out)
