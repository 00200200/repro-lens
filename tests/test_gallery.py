"""Check generated evidence, build failures, and HTML safety through the build CLI."""

import hashlib
import json
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "examples/framework_checks/cases.json"
SCRIPTS = [{"src": "gallery.js", "defer": None}, {"src": "checker.js", "defer": None}]


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.articles = []
        self.scripts = []
        self.code_ids = []
        self.code = {}
        self.active_code = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "article":
            self.articles.append(attrs)
        if tag == "script":
            self.scripts.append(attrs)
        if tag == "code" and "id" in attrs:
            self.code_ids.append(attrs["id"])
            self.active_code = attrs["id"]
            self.code[self.active_code] = ""

    def handle_endtag(self, tag):
        if tag == "code":
            self.active_code = None

    def handle_data(self, data):
        if self.active_code:
            self.code[self.active_code] += data


def build(tmp_path, cases):
    source = tmp_path / "cases.json"
    source.write_text(json.dumps(cases), encoding="utf-8")
    output = tmp_path / "site"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/build_gallery.py"),
            "--cases",
            str(source),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result, output


def test_gallery_renders_checked_examples_and_local_assets(tmp_path):
    cases = json.loads(CASES.read_text())
    result, output = build(tmp_path, cases)
    assert result.returncode == 0, result.stderr
    text = (output / "index.html").read_text()
    page = Page()
    page.feed(text)
    assert {article["id"] for article in page.articles} == {case["id"] for case in cases}
    assert len(page.code_ids) == len(set(page.code_ids)) == 2 * len(cases) + 1
    for case in cases:
        assert case["expected"][0] in text
        for side in ("before", "after"):
            assert page.code[f"{case['id']}-{side}"] == case[side]
    assert page.scripts == SCRIPTS
    assert (output / "gallery.js").is_file()
    assert (output / "styles.css").is_file()
    assert (output / "mark.svg").is_file()
    assert "{{CARDS}}" not in text
    assert "saved static results" in text
    assert "not proof of repeatability" in text


def test_live_checker_ships_the_analyzer_sources_it_loads(tmp_path):
    result, output = build(tmp_path, json.loads(CASES.read_text()))
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "engine/manifest.json").read_text())
    sources = sorted((ROOT / "src/repro_lens").glob("*.py"))
    assert manifest["files"] == [path.name for path in sources]
    for path in sources:
        copied = (output / "engine/repro_lens" / path.name).read_bytes()
        assert copied == path.read_bytes()
        assert manifest["sha256"][path.name] == hashlib.sha256(copied).hexdigest()
    for name in ("checker.js", "checker-worker.mjs"):
        assert (output / name).read_bytes() == (ROOT / "site" / name).read_bytes()
    worker = (output / "checker-worker.mjs").read_text()
    # Same-origin runtime only: the gallery loads no third-party scripts.
    assert 'from "./pyodide/pyodide.mjs"' in worker
    assert "http" not in worker
    text = (output / "index.html").read_text()
    assert 'id="live-source"' in text and 'id="check"' in text


def test_gallery_rejects_wrong_before_or_after_outcomes(tmp_path):
    case = json.loads(CASES.read_text())[0]
    case["after"] = case["before"]
    result, output = build(tmp_path, [case])
    assert result.returncode != 0
    assert "Scanner results disagree" in result.stderr
    assert not (output / "index.html").exists()


def test_gallery_escapes_example_text_without_expanding_template_tokens(tmp_path):
    case = json.loads(CASES.read_text())[0]
    case["name"] = '<script>alert("example")</script> {{COUNT}}'
    case["before"] += '\ntext = "<img src=x onerror=alert(1)>"\n'
    result, output = build(tmp_path, [case])
    assert result.returncode == 0, result.stderr
    text = (output / "index.html").read_text()
    page = Page()
    page.feed(text)
    assert page.scripts == SCRIPTS
    assert "&lt;script&gt;alert(&quot;example&quot;)&lt;/script&gt; {{COUNT}}" in text
    assert "&lt;img src=x onerror=alert(1)&gt;" in text
    assert "<img src=x" not in text


def test_gallery_rejects_duplicate_deep_link_ids(tmp_path):
    case = json.loads(CASES.read_text())[0]
    result, output = build(tmp_path, [case, case])
    assert result.returncode != 0
    assert "duplicate example id" in result.stderr
    assert not (output / "index.html").exists()


def test_highlighted_code_preserves_multiline_unicode_tabs_and_html_text(tmp_path):
    case = json.loads(CASES.read_text())[0]
    extra = (
        '\ntext = """<b>zażółć</b>\u2028separator\n{{COUNT}} & <script>"""\n'
        "if True:\n\tvalue = 2.5  # <tag>\n"
    )
    case["before"] += extra
    case["after"] += extra
    result, output = build(tmp_path, [case])
    assert result.returncode == 0, result.stderr
    text = (output / "index.html").read_text()
    page = Page()
    page.feed(text)
    for side in ("before", "after"):
        assert page.code[f"{case['id']}-{side}"] == case[side]
    assert page.scripts == SCRIPTS
    assert "<b>zażółć</b>" not in text
