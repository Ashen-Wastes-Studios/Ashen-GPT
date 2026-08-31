def _search_web_results(query, limit=8):
    """Search no-key HTML endpoints and return resolved source records."""
    if requests is None:
        return [], 'requests is not installed'
    query = str(query or '').strip()
    if not query:
        return [], 'the query is empty'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml',
    }
    endpoints = [
        ('DuckDuckGo', f'https://html.duckduckgo.com/html/?q={quote(query)}'),
        ('DuckDuckGo Lite', f'https://lite.duckduckgo.com/lite/?q={quote(query)}'),
    ]
    errors = []
    for provider, url in endpoints:
        last_status = None
        for attempt in range(3):
            try:
                response = requests.get(url, headers=headers, timeout=10)
                last_status = response.status_code
                if response.status_code == 200:
                    found = _parse_search_results(response.text)
                    if found:
                        return found[:max(1, int(limit))], provider
                    break
                if response.status_code == 202:
                    # DuckDuckGo occasionally returns 202 when it wants the client to
                    # back off briefly. Retry with a small delay before giving up on
                    # this endpoint.
                    import time as _t
                    _t.sleep(1.5)
                    continue
                if response.status_code == 429:
                    # Rate-limited: back off a bit longer and retry once more.
                    import time as _t
                    _t.sleep(3.0)
                    continue
                # Other non-200 codes are unlikely to become usable on retry.
                errors.append(f'{provider} HTTP {response.status_code}')
                break
            except Exception as exc:
                errors.append(f'{provider}: {exc}')
                break
        else:
            if last_status == 202:
                errors.append(f'{provider} returned HTTP 202 after retries')
    return [], '; '.join(errors) or 'no search provider returned results'
