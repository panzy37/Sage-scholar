"""Prompt-only source locators; full PDF geometry stays in EvidenceStore."""
import json


def compact_location(location):
    keys = {'page', 'page_start', 'page_end', 'element_id', 'chunk_index',
            'char_start', 'char_end', 'offset', 'total_chars', 'table', 'figure'}
    result = {key: value for key, value in (location or {}).items() if key in keys}
    if location and location.get('source_locations'):
        result['source_location_count'] = len(location['source_locations'])
        result['full_location_available'] = True
    return result


def evidence_header(item):
    return (f'[{item.evidence_id}] document={item.document_id} '
            f'section={item.section or "unknown"} page={item.page or "unknown"} '
            f'location={json.dumps(compact_location(item.location), ensure_ascii=False)}')


def text_windows(text, counter, budget):
    """All original characters, bounded by tokens; no dropped/truncated source tail."""
    if budget < 32:
        from .manager import ContextBudgetExceeded
        raise ContextBudgetExceeded('当前请求留给原文的预算不足；请缩小资料范围或当前指令')
    start = 0
    while start < len(text):
        tail = text[start:]
        if counter.count(tail) <= budget:
            yield start, tail
            return
        low, high = 1, len(tail)
        while low < high:
            middle = (low + high + 1) // 2
            if counter.count(tail[:middle]) <= budget:
                low = middle
            else:
                high = middle - 1
        yield start, tail[:low]
        start += low


def evidence_windows(items, counter, budget):
    for item in items:
        header = evidence_header(item)
        for offset, text in text_windows(item.original_text, counter, budget - counter.count(header) - 64):
            yield f'{header} source_offset={offset}\n{text}'


def pack_windows(windows, counter, budget):
    batch = []
    for window in windows:
        if batch and counter.count('\n\n'.join([*batch, window])) > budget:
            yield '\n\n'.join(batch)
            batch = []
        batch.append(window)
    if batch:
        yield '\n\n'.join(batch)
