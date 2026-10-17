"""Akira's internet access, without any API key.

    search("python 3.14 release date")   -> [{"title", "url", "snippet"}, ...]   (DuckDuckGo)
    wikipedia("photosynthesis")          -> {"title", "url", "summary"} or None
    read("https://example.com/page")     -> {"url", "title", "text"}             (a page as plain text)

This is all of Akira's internet access: no API keys and no third-party AI. Results are cached for
ten minutes so a conversation doesn't repeat the same lookups.
"""
import html
import json
import re
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser

UA = "Mozilla/5.0 (compatible; AkiraAssistant/0.1; +https://github.com/kylanwest11-arch/Akira-)"
TIMEOUT = 12
_cache = {}


def _get(url, accept="text/html"):
    hit = _cache.get(url)
    if hit and time.time() - hit[0] < 600:
        return hit[1]
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": accept, "Accept-Language": "en"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        body = r.read(3_000_000).decode(r.headers.get_content_charset() or "utf-8", errors="replace")
    if len(_cache) > 300:
        _cache.clear()
    _cache[url] = (time.time(), body)
    return body


def _text(fragment):
    return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def search(query, n=5):
    """Top web results for a query: DuckDuckGo first, Bing when DuckDuckGo turns the request away."""
    for engine in (_duckduckgo, _bing):
        try:
            results = engine(query, n)
        except Exception:
            results = []
        if results:
            return results
    return []


BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def _bing(query, n):
    url = "https://www.bing.com/search?" + urllib.parse.urlencode({"q": query, "setlang": "en"})
    req = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA, "Accept-Language": "en"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        page = r.read(3_000_000).decode("utf-8", errors="replace")
    results = []
    for chunk in page.split('<li class="b_algo"')[1:]:
        m = re.search(r"<h2[^>]*>\s*<a[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>", chunk, re.S)
        if not m:
            continue
        href, title = html.unescape(m.group(1)), _text(m.group(2))
        u = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("u", [""])[0]
        if u.startswith("a1"):                      # Bing hides the real address, base64-encoded, in u=a1...
            import base64
            try:
                href = base64.urlsafe_b64decode(u[2:] + "=" * (-len(u[2:]) % 4)).decode()
            except Exception:
                pass
        snip = re.search(r"<p[^>]*>(.*?)</p>", chunk, re.S)
        results.append({"title": title, "url": href, "snippet": _text(snip.group(1)) if snip else ""})
        if len(results) >= n:
            break
    return results


def _duckduckgo(query, n):
    url = "https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode({"q": query})
    page = _get(url)
    if "result-link" not in page:
        _cache.pop(url, None)                       # turned away: don't cache the refusal
        return []
    results = []
    for m in re.finditer(r"<a[^>]+href=\"([^\"]+)\"[^>]*class='result-link'>(.*?)</a>(.*?)class='result-snippet'>(.*?)</td>",
                         page, re.S):
        href, title, _, snippet = m.groups()
        target = urllib.parse.parse_qs(urllib.parse.urlparse(html.unescape(href)).query).get("uddg", [href])[0]
        if "duckduckgo.com/y.js" in target or "ad_domain" in target:
            continue
        if target.startswith("//"):
            target = "https:" + target
        results.append({"title": _text(title), "url": target, "snippet": _text(snippet)})
        if len(results) >= n:
            break
    return results


def wikipedia(query, _retry=True):
    """The best-matching Wikipedia article's summary, or None."""
    try:
        return _wikipedia(query)
    except Exception:
        if not _retry:
            raise
        time.sleep(1.0)
        return wikipedia(query, _retry=False)


def _wikipedia(query):
    found = json.loads(_get("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 1}), "application/json"))
    hits = found.get("query", {}).get("search", [])
    if not hits:
        return None
    title = hits[0]["title"]
    s = json.loads(_get("https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title.replace(" ", "_")),
                        "application/json"))
    if s.get("type") == "disambiguation" or not s.get("extract"):
        return None
    return {"title": s.get("title", title), "summary": s["extract"],
            "url": s.get("content_urls", {}).get("desktop", {}).get("page", "https://en.wikipedia.org/wiki/" + title)}


class _Reader(HTMLParser):
    SKIP = {"script", "style", "noscript", "nav", "footer", "header", "svg", "form", "aside"}

    def __init__(self):
        super().__init__()
        self.depth, self.parts, self.title, self._in_title = 0, [], "", False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.depth += 1
        if tag == "title":
            self._in_title = True
        if tag in ("p", "br", "li", "h1", "h2", "h3", "h4", "tr", "div"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.depth:
            self.depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.depth:
            self.parts.append(data)


def read(url, max_chars=6000):
    """A web page as plain text (scripts, menus and footers removed)."""
    if not re.match(r"^https?://", url):
        raise ValueError("Only http and https links can be read.")
    p = _Reader()
    p.feed(_get(url))
    text = re.sub(r"\n\s*\n+", "\n\n", re.sub(r"[ \t]+", " ", "".join(p.parts))).strip()
    return {"url": url, "title": p.title.strip(), "text": text[:max_chars]}


# --- browsing: open a website, see its links, follow them -------------------------------
class _Page(HTMLParser):
    """Collects a page's title, headings, main text and links."""
    SKIP = {"script", "style", "noscript", "svg", "form", "template"}
    CHROME = {"nav", "footer", "header", "aside"}

    def __init__(self, base):
        super().__init__()
        self.base, self.skip, self.chrome = base, 0, 0
        self.title, self.headings, self.parts, self.links = "", [], [], []
        self._in, self._heading, self._link = None, None, None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in self.SKIP:
            self.skip += 1
        if tag in self.CHROME:
            self.chrome += 1
        if tag == "title" and not self.skip and not self.title:   # the page's own title, not an icon's
            self._in = "title"
        if tag in ("h1", "h2", "h3") and not self.skip:
            self._heading = []
        if tag == "a" and a.get("href") and not self.skip:
            self._link = [a["href"], []]
        if tag in ("p", "br", "li", "h1", "h2", "h3", "h4", "tr", "div", "section", "article"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        if tag in self.CHROME and self.chrome:
            self.chrome -= 1
        if tag == "title":
            self._in = None
        if tag in ("h1", "h2", "h3") and self._heading is not None:
            h = " ".join("".join(self._heading).split())
            if 2 < len(h) < 120:
                self.headings.append(h)
            self._heading = None
        if tag == "a" and self._link:
            href, words = self._link
            text = " ".join("".join(words).split())
            self.links.append((href, text))
            self._link = None

    def handle_data(self, data):
        if self._in == "title":
            self.title += data
            return
        if self.skip:
            return
        if self._heading is not None:
            self._heading.append(data)
        if self._link:
            self._link[1].append(data)
        if not self.chrome:
            self.parts.append(data)


def visit(url, max_links=15):
    """Open a page: its title, a readable excerpt, its sections and its links (absolute, de-duplicated)."""
    if not re.match(r"^https?://", url):
        url = "https://" + url
    p = _Page(url)
    p.feed(_get(url))
    text = re.sub(r"\n\s*\n+", "\n\n", re.sub(r"[ \t]+", " ", "".join(p.parts))).strip()
    paragraphs = [t.strip() for t in text.split("\n") if len(t.strip()) > 60]
    links, seen = [], set()
    for href, label in p.links:
        if not label or len(label) < 2 or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        full = urllib.parse.urljoin(url, href).split("#")[0]
        if full in seen or full.rstrip("/") == url.rstrip("/") or not full.startswith("http"):
            continue
        seen.add(full)
        links.append({"text": label[:90], "url": full})
    return {"url": url, "title": " ".join(p.title.split()) or url, "excerpt": "\n\n".join(paragraphs[:4])[:1500],
            "headings": list(dict.fromkeys(p.headings))[:8], "links": links[:max_links], "all_links": len(links)}


def page_markdown(page):
    out = [f"**{page['title']}**  \n{page['url']}"]
    if page["excerpt"]:
        out.append(page["excerpt"])
    else:
        out.append("_This page shows little text without running its scripts, so here are its links._")
    if page["headings"]:
        out.append("**On this page:** " + " · ".join(page["headings"]))
    if page["links"]:
        out.append("**Links:**\n" + "\n".join(f"{i}. [{l['text']}]({l['url']})" for i, l in enumerate(page["links"], 1)))
        out.append(f"Say \"open 1\" to {page['links'][0]['text'][:40]!r}, or any other number, to follow a link.")
    return "\n\n".join(out)


DOMAIN = re.compile(r"\b((?:[a-z0-9-]+\.)+(?:com|org|net|edu|gov|io|ai|co|uk|us|ca|de|fr|jp|info|dev|app|news|tv|me|ly|gg))\b(/[^\s]*)?", re.I)
GO = re.compile(r"\b(go to|goto|open( up)?|visit|browse( to)?|navigate to|take me to|pull up|load|check out|show me)\b", re.I)
FOLLOW = re.compile(r"^\s*(?:please\s+)?(?:open|click|follow|go to|read|visit|show me)?\s*(?:link|result|number|#)?\s*(\d{1,2})\s*[.!]?\s*$", re.I)
SITE_SEARCH = re.compile(r"\bsearch\s+((?:[a-z0-9-]+\.)+[a-z]{2,})\s+for\s+(.+)$", re.I)
LINK_LINE = re.compile(r"^(\d{1,2})\. \[[^\]]*\]\((https?://[^)]+)\)", re.M)


def browse_target(text, history=()):
    """If the message asks to open a website or follow a link Akira listed, return the URL (or a site search)."""
    t = text.strip()
    m = FOLLOW.match(t)
    if m:
        for msg in reversed(list(history)):
            if msg.get("role") == "assistant":
                links = dict(LINK_LINE.findall(msg.get("content", "")))
                if links:
                    return {"url": links.get(m.group(1))} if m.group(1) in links else {"missing": int(m.group(1)), "count": len(links)}
                break
        return None
    m = SITE_SEARCH.search(t)
    if m:
        return {"site_search": (m.group(1), m.group(2).rstrip("?.! "))}
    url = URL.search(t)
    if url:
        return {"url": url.group(0).rstrip(".,)")}
    if GO.search(t):
        d = DOMAIN.search(t)
        if d:
            return {"url": "https://" + d.group(1) + (d.group(2) or "")}
        name = re.sub(r"(?i).*?\b(go to|goto|open( up)?|visit|browse( to)?|navigate to|take me to|pull up|load|check out|show me)\b\s*", "", t)
        name = re.sub(r"(?i)\b(the |website|site|homepage|home page|web ?page|page|please|for me|official)\b", " ", name).strip(" ?.!")
        words = name.lower().split()
        webby = re.search(r"(?i)\b(website|site|homepage|home page|web ?page)\b", t)
        if name and len(words) <= 3 and (webby or not set(words) & PLACES):
            return {"find_site": name}
    return None


# everyday places people "go to" that aren't websites
PLACES = set("""store shop shops school work office bed sleep gym park church doctor dentist bathroom toilet home house mall
beach movies cinema bank hospital class practice store's grocery market restaurant party airport station church
tomorrow today tonight later now""".split())


def browse(target):
    """Carry out a browse_target(): open the page (finding the site first if needed) and describe it."""
    if "missing" in target:
        return f"There's no link {target['missing']}: the last page I showed had {target['count']} links.", []
    if "site_search" in target:
        site, q = target["site_search"]
        results = search(f"site:{site} {q}", 8)
        if not results:
            return f"I couldn't find anything for \"{q}\" on {site}.", []
        return (f"**Results for \"{q}\" on {site}:**\n" + "\n".join(
            f"{i}. [{r['title']}]({r['url']}): {r['snippet']}" for i, r in enumerate(results, 1))
            + "\n\nSay \"open 1\" (or any number) to go to a result."), [r["url"] for r in results]
    url = target.get("url")
    if not url and target.get("find_site"):
        name = target["find_site"]
        results = search(name + " official site", 3)
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        match = next((r for r in results if key and key[:6] in re.sub(r"[^a-z0-9]", "", urllib.parse.urlparse(r["url"]).netloc.lower())), None)
        if match is None:
            return None, []        # not obviously a website: let Akira answer normally
        url = match["url"]
    page = visit(url)
    return page_markdown(page), [page["url"]]


# --- deciding when to go online -------------------------------------------------------
NEEDS_WEB = re.compile(r"\b(search (the )?(web|internet|online)|search for|google|look (it |this |that )?up|online|"
                       r"latest|news|headlines?|today'?s|this (week|month|year)|right now|currently|current|recent(ly)?|"
                       r"price of|how much (is|are|does)|stock|weather|forecast|score|who won|release date|"
                       r"20[2-3]\d|website|https?://)\b", re.I)
PERSONAL = re.compile(r"\b(you|your|yours|yourself|akira|i|i'm|me|my|mine|we|our|us)\b", re.I)   # about us, not the world
QUESTION = re.compile(r"^\s*(akira[, ]+)?(what|who|when|where|which|why|how (much|many|old|tall|far|big|long|does|do|did|is|are)|"
                      r"is|are|was|were|does|did|define|tell me about|explain|describe)\b", re.I)
URL = re.compile(r"https?://[^\s)>\]]+")


def clean_query(text):
    """Turn a spoken-style question into search keywords: "what's the latest news about X?" -> "latest news about X"."""
    q = text.replace("\u2019", "'")
    q = re.sub(r"(?i)^\s*(hey\s+)?akira[,:]?\s*", "", q)
    q = re.sub(r"(?i)^(can you |could you |please )?(search (the )?(web|internet|online) for|search for|google|look up|find out)\s*", "", q)
    q = re.sub(r"(?i)^(please )?(tell me|explain|describe|define)( about)?\s+", "", q)
    q = re.sub(r"(?i)^(what|who|when|where|which|why)('s|'re| is| are| was| were| does| do| did)\s+(the |a |an )?", "", q)
    q = re.sub(r"(?i)^how (much|many|old|tall|far|big|long) (is|are|does|do|was)?\s*(the |a |an )?", r"how \1 ", q)
    return q.strip().rstrip("?.! ")


def answer(text, max_results=5):
    """Look a question up online. Returns (markdown answer, list of sources) or (None, [])."""
    link = URL.search(text)
    if link:
        page = read(link.group(0).rstrip(".,"))
        excerpt = page["text"][:1200].strip()
        return (f"**{page['title'] or page['url']}**\n\n{excerpt}{'…' if len(page['text']) > 1200 else ''}\n\n"
                f"Source: {page['url']}"), [page["url"]]
    q = clean_query(text)
    if not q:
        return None, []
    wiki = None
    if not NEEDS_WEB.search(text):            # timeless facts: an encyclopedia summary reads best
        try:
            wiki = wikipedia(q)
        except Exception:
            wiki = None
    results = []
    try:
        results = search(q, max_results)
    except Exception:
        pass
    if not results and not wiki:
        try:
            wiki = wikipedia(q)                # the web came back empty: the encyclopedia may still know
        except Exception:
            wiki = None
    if not wiki and not results:
        return None, []
    parts, sources = [], []
    if wiki:
        parts.append(f"{wiki['summary']}\n\n— [{wiki['title']} (Wikipedia)]({wiki['url']})")
        sources.append(wiki["url"])
    shown = [r for r in results if r["url"] not in sources][: (3 if wiki else max_results)]
    if shown:
        parts.append(("**More from the web:**" if wiki else "**Here's what I found online:**") + "\n" + "\n".join(
            f"- [{r['title']}]({r['url']}): {r['snippet']}" for r in shown))
        sources += [r["url"] for r in shown]
    return "\n\n".join(parts), sources
