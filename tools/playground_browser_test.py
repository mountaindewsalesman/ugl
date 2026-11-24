"""End-to-end test of the built playground in REAL browsers (Playwright).

    pip install playwright && python -m playwright install chromium firefox
    python tools/build_playground.py
    python tools/playground_browser_test.py [--site _site] [--browsers chromium firefox]

For each browser: serve the site, load the page, wait for Python to start,
then check that (1) an example sentence renders READ + BACK, (2) typing UGL
decodes live, (3) the readable view matches the library.  Fails on any
'Could not start Python' status or page/worker error.
"""
import argparse
import functools
import http.server
import os
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UGL_IN = "n4 v2 n2344s #3"
EXPECT_READ = "someone has apples three"
EXPECT_BACK = "Someone has 3 apples."


def serve(site):
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    handler = functools.partial(Quiet, directory=site)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/"


def run(browser_name, url, timeout_ms):
    from playwright.sync_api import sync_playwright
    errors = []
    with sync_playwright() as p:
        browser = getattr(p, browser_name).launch()
        page = browser.new_page()
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: m.type == "error" and errors.append(f"console: {m.text}"))
        page.goto(url)
        page.wait_for_function(
            "() => { const s = document.getElementById('status');"
            " return s.classList.contains('done') || s.classList.contains('failed'); }",
            timeout=timeout_ms)
        status = page.inner_text("#status-text")
        if "Could not start" in status:
            browser.close()
            return False, [status] + errors
        # 0. type English -> live encoding in the browser == CPython's encoding
        import json
        gallery = json.loads(page.evaluate("() => fetch('gallery.json').then(r => r.text())"))
        for g in gallery[:6]:
            page.fill("#en", g["en"])
            page.wait_for_function("(want) => document.getElementById('ugl').value.trim() === want",
                                   arg="\n".join(g["ugl"]), timeout=60000)
        print(f"  [{browser_name}] live-encoded {min(6, len(gallery))} typed sentences, identical to CPython")
        page.fill("#en", "")
        # 1. first example button -> results with READ and BACK rows
        page.click("#examples button >> nth=0")
        page.wait_for_selector("#results .back", timeout=30000)
        back = page.inner_text("#results .back")
        # 2. type UGL -> live decode + readable view
        page.fill("#ugl", UGL_IN)
        page.wait_for_function(
            "(want) => [...document.querySelectorAll('#results .readable')].some(e => e.textContent.trim() === want)",
            arg=EXPECT_READ, timeout=30000)
        decoded = page.inner_text("#results .back")
        browser.close()
    problems = list(errors)
    if not back.strip():
        problems.append("example produced no back-translation")
    if decoded.strip() != EXPECT_BACK:
        problems.append(f"typed UGL decoded to {decoded!r}")
    print(f"  [{browser_name}] {status}")
    print(f"  [{browser_name}] example BACK: {back!r}; typed UGL -> {decoded!r}")
    return not problems, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=os.path.join(ROOT, "_site"))
    ap.add_argument("--browsers", nargs="+", default=["chromium", "firefox"])
    ap.add_argument("--timeout", type=int, default=180000, help="ms to wait for Python to start")
    a = ap.parse_args()
    httpd, url = serve(os.path.abspath(a.site))
    ok_all = True
    for b in a.browsers:
        print(f"{b}: loading {url}")
        ok, problems = run(b, url, a.timeout)
        print(f"{b}: {'OK' if ok else 'FAILED'}")
        for pr in problems:
            print("   ", pr)
        ok_all &= ok
    httpd.shutdown()
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
