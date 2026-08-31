def _search_web_results(query, limit=8):
    """Search no-key HTML endpoints and return resolved source records."""
    if requests is None:
        return [], 'requests is not installed'
    query = str(query or '').strip()
    if not query:
        return [], 'the query is empty'
    base_headers = {
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
                headers = dict(base_headers)
                response = requests.get(url, headers=headers, timeout=10)
                last_status = response.status_code
                if response.status_code == 200:
                    found = _parse_search_results(response.text)
                    if found:
                        return found[:max(1, int(limit))], provider
                    break
                if response.status_code == 202:
                    import time as _t
                    _t.sleep(1.5)
                    continue
                if response.status_code == 429:
                    import time as _t
                    _t.sleep(3.0)
                    continue
                errors.append(f'{provider} HTTP {response.status_code}')
                break
            except Exception as exc:
                errors.append(f'{provider}: {exc}')
                break
        else:
            if last_status == 202:
                errors.append(f'{provider} returned HTTP 202 after retries')
    return [], '; '.join(errors) or 'no search provider returned results'


def _parse_bing_results(html_text, query):
    """Parse Bing search results from HTML, extracting real external URLs."""
    results = []
    seen = set()
    # Bing wraps external links in redirect URLs like:
    # https://www.bing.com/ck/a?...&u=a1<base64url-encoded-real-url>
    bing_redirect_re = re.compile(r'https://www\.bing\.com/ck/a\?[^"]*u=a1([A-Za-z0-9_\-/+=]+)')
    for m in bing_redirect_re.finditer(html_text or ''):
        encoded = m.group(1)
        try:
            padded = encoded + '=' * (-len(encoded) % 4)
            decoded = base64.urlsafe_b64decode(padded).decode('utf-8', errors='ignore')
            decoded = unquote(decoded).strip()
            if decoded.startswith(('http://', 'https://')):
                url = _normalise_http_url(decoded)
                if url and url not in seen:
                    seen.add(url)
                    results.append({'title': '', 'url': url, 'snippet': ''})
        except Exception:
            continue
    # Also try direct external links in the page (less common with modern Bing)
    direct_re = re.compile(r'href="(https?://[^"]+)"', re.IGNORECASE)
    for m in direct_re.finditer(html_text or ''):
        href = m.group(1)
        if any(x in href for x in ['bing.com', 'microsoft.com', 'go.microsoft', 'msn.com', 'skype.com', 'live.com']):
            continue
        url = _normalise_http_url(href)
        if url and url not in seen:
            seen.add(url)
            results.append({'title': '', 'url': url, 'snippet': ''})
    return results


def _search_bing(query, limit=8):
    """Query Bing search and return resolved result records."""
    if requests is None:
        return [], 'requests is not installed'
    query = str(query or '').strip()
    if not query:
        return [], 'the query is empty'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    url = f'https://www.bing.com/search?q={quote(query)}'
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return [], f'Bing HTTP {response.status_code}'
        found = _parse_bing_results(response.text, query)
        if found:
            return found[:max(1, int(limit))], 'Bing'
        return [], 'Bing returned no parseable results'
    except Exception as exc:
        return [], f'Bing: {exc}'
