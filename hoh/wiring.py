"""One place that assembles the optional pieces (truth filter, robust fetcher, semantic embedder) so `learn`, `chat` and `life`
behave the same. Everything degrades gracefully: missing Playwright/sentence-transformers never stops learning."""
import functools
from pathlib import Path


def add_learning_flags(ap):
    g = ap.add_argument_group("learn-from-anywhere")
    g.add_argument("--no-truth", action="store_true", help="skip the truth/corroboration filter (quality filter still applies)")
    g.add_argument("--allow-unverified", action="store_true", help="also learn single-source unverified text (less safe)")
    g.add_argument("--low-trust-host", action="append", default=[], help="host suffix to treat as low-trust (repeatable)")
    g.add_argument("--trusted-host", action="append", default=[], help="extra host suffix to treat as trusted (repeatable)")
    g.add_argument("--no-render", action="store_true", help="never start a headless browser for JavaScript pages")
    g.add_argument("--plain-fetch", action="store_true", help="use the old plain fetcher (no retries/paywall detection)")
    return ap


def build_truth(a, state_dir):
    if getattr(a, "no_truth", False):
        return None
    from .truthfilter import TruthFilter, DEFAULT_POLICY
    pol = {"allow_unverified": bool(getattr(a, "allow_unverified", False)),
           "low_suffixes": list(getattr(a, "low_trust_host", []) or []),
           "trusted_suffixes": DEFAULT_POLICY["trusted_suffixes"] + list(getattr(a, "trusted_host", []) or [])}
    return TruthFilter(Path(state_dir) / "facts.db", pol)


def build_fetcher(a):
    if getattr(a, "plain_fetch", False):
        from .ingest import fetch_url
        return fetch_url
    from .browser import RobustFetcher
    return RobustFetcher(render=not getattr(a, "no_render", False))


def build_ingestor(a, inbox, memory, workspace, state_dir, min_quality=0.6):
    from .ingest import Ingestor, search_urls
    fetcher = build_fetcher(a)
    return Ingestor(inbox, memory, workspace, min_quality=min_quality, fetcher=fetcher,
                    searcher=functools.partial(search_urls, fetcher=fetcher), truth=build_truth(a, state_dir))


def build_memory(path, semantic_path=None):
    """MemoryStore with the best embedder available; reindexes automatically when the embedder changed."""
    from .memory import MemoryStore
    from .semantic import best_available_embedder
    emb = best_available_embedder(semantic_path)
    m = MemoryStore(path, embed_fn=emb)
    if m.needs_reindex():
        m.reindex()
    return m
