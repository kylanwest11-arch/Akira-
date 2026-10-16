"""Relations & References: Akira's map of how your information connects.

Everything is a node (person, message, meeting, project, file, decision, task,
event, photo) and nodes are joined by typed edges ("sent_by", "discussed_in",
"decided_at", "attached_to", ...). Instead of matching words inside one file,
Akira starts from the things you mention and walks the connections.

    "documents related to the Riverbook keyboard decision we made after that meeting"
      -> decision "Riverbook keyboard layout"  (matched by words)
      -> meeting it was decided at              (1 hop)
      -> files attached to that decision/meeting (1-2 hops)
"""
import re
from collections import defaultdict, deque

WORD = re.compile(r"[a-z0-9]+")
STOP = set("""a an and the of to in on for with about from by at is are was were be been it this that
these those me my we our you your i show find get give related relating what which who where when
did do does made make after before last that other some any all documents document files file please
akira can could would tell""".split())


def stem(w):
    """Fold simple plurals to one internal form, so 'movies' matches 'movie' and 'parties' matches 'party'.

    The form is only compared with other stems, never shown: movies/movie -> movie, parties/party -> partie.
    """
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is", "as", "os")):
        w = w[:-1]
    if len(w) > 3 and w.endswith("y"):
        w = w[:-1] + "ie"
    return w


def words(text):
    return [stem(w) for w in WORD.findall(text.lower()) if w not in STOP and len(w) > 1]


class RelationGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = defaultdict(list)  # id -> [(relation, other_id)]

    def add(self, node_id, kind, title, **props):
        self.nodes[node_id] = {"id": node_id, "kind": kind, "title": title, **props}
        return node_id

    def link(self, a, relation, b):
        self.edges[a].append((relation, b))
        self.edges[b].append((relation, a))

    def text_of(self, node):
        return " ".join(str(v) for k, v in node.items() if k not in ("id",))

    def match(self, query, kinds=None, limit=5):
        """Nodes whose own text best matches the words in query."""
        q = set(words(query))
        if not q:
            return []
        scored = []
        for node in self.nodes.values():
            if kinds and node["kind"] not in kinds:
                continue
            hits = q & set(words(self.text_of(node)))
            if hits:
                scored.append((len(hits) / len(q), node["id"]))
        scored.sort(reverse=True)
        return [(nid, s) for s, nid in scored[:limit]]

    def related(self, query, want=None, depth=3, limit=8):
        """Start from the nodes the query names, walk outwards, rank what we reach.

        A reached node scores by how strongly its seed matched, discounted per hop,
        plus a bonus if it also matches the query itself.
        """
        seeds = self.match(query, limit=4)
        best = {}
        paths = {}
        for seed, seed_score in seeds:
            seen = {seed: 0}
            prev = {seed: None}
            queue = deque([seed])
            while queue:
                cur = queue.popleft()
                if seen[cur] >= depth:
                    continue
                for rel, nxt in self.edges[cur]:
                    if nxt in seen:
                        continue
                    seen[nxt] = seen[cur] + 1
                    prev[nxt] = (cur, rel)
                    queue.append(nxt)
            q = set(words(query))
            for nid, hops in seen.items():
                node = self.nodes[nid]
                if want and node["kind"] not in want:
                    continue
                own = len(q & set(words(self.text_of(node)))) / max(1, len(q))
                score = seed_score * (0.6 ** hops) + 0.5 * own
                if score > best.get(nid, 0):
                    best[nid] = score
                    paths[nid] = self._path(prev, nid)
        ranked = sorted(best.items(), key=lambda kv: -kv[1])[:limit]
        return [{**self.nodes[nid], "score": round(s, 3), "via": paths[nid]} for nid, s in ranked]

    def _path(self, prev, nid):
        steps = []
        while prev.get(nid):
            parent, rel = prev[nid]
            steps.append(f"{self.nodes[parent]['title']} -[{rel}]-> {self.nodes[nid]['title']}")
            nid = parent
        return list(reversed(steps))

    def neighbours(self, node_id):
        return [{"relation": rel, **self.nodes[o]} for rel, o in self.edges[node_id]]
