#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
Fetch AscendC API documentation from hiascend.com using headless Playwright.

Usage:
    python3 src/scripts/fetch_ascendc_doc.py Exp
    python3 src/scripts/fetch_ascendc_doc.py WholeReduceSum
    python3 src/scripts/fetch_ascendc_doc.py Cast
    python3 src/scripts/fetch_ascendc_doc.py --url <hiascend_url>

How it works:
    1. Navigates to the CANN 9.0.0-beta.2 API catalog page
    2. Extracts all <a> tags with matching text → gets the exact URL
    3. Navigates to that URL → extracts page content

Worker usage: just pass API name, script handles the rest.

ANTI-CHEAT: Only allows hiascend.com. Blocks CANN source repos.
"""
import sys
import re

BLOCKED_PATTERNS = [
    r"gitee\.com/ascend",
    r"github\.com/Ascend",
    r"gitcode\.com/ascend",
    r"workspace/cann",
]

API_CATALOG_URL = (
    "https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/"
    "API/ascendcopapi/atlasascendc_api_07_0003.html"
)


def extract_page_content(page) -> str:
    return page.evaluate("""() => {
        const selectors = ['.doc-content-box', '.article-content', 'article', '.content-right', 'main'];
        for (const sel of selectors) {
            const el = document.querySelector(sel);
            if (el && el.innerText.length > 200) return el.innerText;
        }
        return document.body.innerText;
    }""")


def fetch_api_by_name(api_name: str) -> str:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            print("Loading API catalog...", file=sys.stderr)
            page.goto(API_CATALOG_URL, timeout=20000)
            page.wait_for_timeout(4000)

            # Extract all <a> links with matching text and their hrefs
            links = page.evaluate("""(apiName) => {
                const allLinks = document.querySelectorAll('a');
                const matches = [];
                for (const el of allLinks) {
                    const text = el.textContent.trim();
                    if (text === apiName || text === apiName + '（废弃）' || text === apiName + '(ISASI)') {
                        if (el.href && el.href.includes('atlasascendc_api')) {
                            matches.push({text: text, href: el.href});
                        }
                    }
                }
                return matches;
            }""", api_name)

            if not links:
                # Try partial match
                links = page.evaluate("""(apiName) => {
                    const allLinks = document.querySelectorAll('a');
                    const matches = [];
                    for (const el of allLinks) {
                        const text = el.textContent.trim();
                        if (text.toLowerCase().includes(apiName.toLowerCase()) &&
                            el.href && el.href.includes('atlasascendc_api')) {
                            matches.push({text: text, href: el.href});
                        }
                    }
                    return matches.slice(0, 5);
                }""", api_name)

            if not links:
                return f"ERROR: API '{api_name}' not found in catalog. Check ASCENDC_API_CATALOG.md for valid names."

            # Use first match (SIMD version preferred over Reg version)
            target = links[0]
            if len(links) > 1:
                # Prefer non-Reg version (SIMD basic API)
                for l in links:
                    if 'Reg' not in l['text'] and 'ISASI' not in l['text']:
                        target = l
                        break
                print(f"Multiple matches: {[l['text'] for l in links]}. Using: {target['text']}", file=sys.stderr)

            print(f"Found: {target['text']} → {target['href']}", file=sys.stderr)

            # Navigate to the API detail page
            page.goto(target['href'], timeout=30000)
            page.wait_for_timeout(3000)

            return extract_page_content(page)

        except Exception as e:
            return f"ERROR: {e}"
        finally:
            browser.close()


def fetch_url(url: str) -> str:
    from urllib.parse import urlparse
    parsed = urlparse(url)

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            return f"BLOCKED: URL matches anti-cheat pattern: {pattern}"

    if parsed.hostname not in ("hiascend.com", "www.hiascend.com"):
        return f"BLOCKED: domain {parsed.hostname} not allowed. Only hiascend.com."

    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(url, timeout=15000)
            page.wait_for_timeout(3000)
            return extract_page_content(page)
        except Exception as e:
            return f"ERROR: {e}"
        finally:
            browser.close()


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 fetch_ascendc_doc.py <ApiName>    # e.g. Exp, Cast, WholeReduceSum")
        print("  python3 fetch_ascendc_doc.py --url <url>  # Fetch specific hiascend.com URL")
        sys.exit(1)

    if sys.argv[1] == "--url":
        if len(sys.argv) < 3:
            print("ERROR: --url requires a URL argument")
            sys.exit(1)
        content = fetch_url(sys.argv[2])
    else:
        api_name = " ".join(sys.argv[1:])
        content = fetch_api_by_name(api_name)

    max_chars = 15000
    if len(content) > max_chars:
        print(content[:max_chars])
        print(f"\n... [TRUNCATED at {max_chars} chars, total {len(content)}]")
    else:
        print(content)


if __name__ == "__main__":
    main()
