"""Check generated evidence, build failures, and HTML safety through the build CLI."""

import json
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "examples/framework_checks/cases.json"


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.articles = []
        self.scripts = []
        self.code_ids = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "article":
            self.articles.append(attrs)
        if tag == "script":
            self.scripts.append(attrs)
        if tag == "code" and "id" in attrs:
            self.code_ids.append(attrs["id"])


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
    assert len(page.code_ids) == len(set(page.code_ids)) == 2 * len(cases)
    for case in cases:
        assert case["expected"][0] in text
    assert page.scripts == [{"src": "gallery.js", "defer": None}]
    assert (output / "gallery.js").is_file()
    assert (output / "styles.css").is_file()
    assert "{{CARDS}}" not in text
    assert "saved static results" in text


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
    assert page.scripts == [{"src": "gallery.js", "defer": None}]
    assert "&lt;script&gt;alert(&quot;example&quot;)&lt;/script&gt; {{COUNT}}" in text
    assert "&lt;img src=x onerror=alert(1)&gt;" in text
    assert "<img src=x" not in text


def test_gallery_rejects_duplicate_deep_link_ids(tmp_path):
    case = json.loads(CASES.read_text())[0]
    result, output = build(tmp_path, [case, case])
    assert result.returncode != 0
    assert "duplicate example id" in result.stderr
    assert not (output / "index.html").exists()
