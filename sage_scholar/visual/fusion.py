# 功能：使用 RRF 融合文本与视觉检索结果，支持证据级和页级排序。
"""Rank fusion across independent text and image embedding spaces."""
import os


def page_key(item):
    # A page from another version/document must never accumulate this vote.
    return ((item.document_id or item.source_id), item.source_version, item.page) if item.page else ('evidence', item.evidence_id)


def fuse(text, images, limit, *, strategy=None, max_pages=None):
    strategy = strategy or os.getenv('RAG_FUSION_STRATEGY', 'evidence')
    if strategy not in {'evidence', 'page'}:
        raise ValueError('RAG_FUSION_STRATEGY must be evidence or page')
    if not images:
        return text
    if strategy == 'page':
        return _pages(text, images, limit, max_pages or limit)
    scores, records = {}, {}
    for ranking in (text, images):
        for rank, item in enumerate(ranking, 1):
            records[item.evidence_id] = item
            scores[item.evidence_id] = scores.get(item.evidence_id, 0) + 1 / (60 + rank)
    return [records[key] for key in sorted(scores, key=scores.get, reverse=True)[:limit]]


def _pages(text, images, limit, max_pages):
    scores, groups = {}, {}
    for ranking in (text, images):
        seen = set()
        for item in ranking:
            key = page_key(item)
            groups.setdefault(key, {})[item.evidence_id] = item
            if key not in seen:
                seen.add(key)
                scores[key] = scores.get(key, 0) + 1 / (60 + len(seen))
    order = sorted(scores, key=scores.get, reverse=True)
    # Page-diverse representatives first, then complementary original evidence.
    # No new Evidence IDs, rewritten text, or fabricated page-spanning evidence.
    selected = order[:min(max_pages, limit)]
    # Include original pixels when both channels found the page. The remaining
    # slots carry the complementary textual evidence from these same pages.
    representatives = [next((e for e in groups[key].values() if e.metadata.get('kind') == 'visual_source'),
                            next(iter(groups[key].values()))) for key in selected]
    result = list(representatives)
    included = {e.evidence_id for e in representatives}
    for key in selected:
        result.extend(e for e in groups[key].values() if e.evidence_id not in included)
    return result[:limit]
