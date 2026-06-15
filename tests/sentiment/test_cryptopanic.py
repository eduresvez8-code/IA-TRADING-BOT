"""Tests de la ingesta de CryptoPanic — sin red (cliente fake)."""

from src.sentiment.cryptopanic import _parse_post, _parse_time, fetch_cryptopanic


def post(url, title="Bitcoin rally continues", t="2025-06-01T12:00:00Z"):
    return {"url": url, "title": title, "published_at": t,
            "source": {"domain": "coindesk.com"}}


class FakeResp:
    def __init__(self, data=None, status=200):
        self._data = data or {}
        self.status_code = status

    def json(self):
        return self._data


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def get(self, url, params=None):
        self.calls.append((url, params))
        return self.responses.pop(0)


# ---------- parsers ----------

def test_parse_post_valido():
    item = _parse_post(post("https://cryptopanic.com/news/1"))
    assert item.title == "Bitcoin rally continues"
    assert item.source == "coindesk.com"
    assert item.published_at.year == 2025 and item.published_at.tzinfo is not None


def test_parse_post_sin_url_o_titulo_es_none():
    assert _parse_post({"title": "x"}) is None
    assert _parse_post({"url": "u"}) is None


def test_parse_time_iso_con_z():
    dt = _parse_time({"published_at": "2025-06-01T12:00:00Z"})
    assert dt.tzinfo is not None and dt.hour == 12


# ---------- paginación ----------

async def test_paginacion_sigue_next_y_deduplica():
    page1 = {"results": [post("u1"), post("u2")], "next": "https://cryptopanic.com/?page=2"}
    page2 = {"results": [post("u2"), post("u3")], "next": None}  # u2 repetida
    fake = FakeClient([FakeResp(page1), FakeResp(page2)])

    items = await fetch_cryptopanic("tok", client=fake, max_pages=5)
    assert len(items) == 3                 # u1, u2, u3 (deduplicada)
    assert len(fake.calls) == 2            # la página sin `next` corta


async def test_backoff_ante_429():
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    fake = FakeClient([
        FakeResp(status=429),
        FakeResp({"results": [post("u1")], "next": None}),
    ])
    items = await fetch_cryptopanic("tok", client=fake, sleep=fake_sleep)
    assert sleeps == [1.0]                  # un reintento, backoff inicial
    assert len(items) == 1
