"""Build a static, offline-readable gallery using the shared Repro Lens analyzer."""

import argparse
import hashlib
import html
import io
import json
import keyword
import re
import shutil
import tokenize
from pathlib import Path

from repro_lens.analysis import analyze

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/00200200/repro-lens"


def highlight(source: str) -> str:
    """Color Python tokens while preserving the exact, HTML-escaped source text."""
    offsets = [0]
    # Match StringIO.readline: Unicode separators inside strings are not new lines.
    for line in source.split("\n")[:-1]:
        offsets.append(offsets[-1] + len(line) + 1)
    classes = {
        tokenize.STRING: "string",
        tokenize.NUMBER: "number",
        tokenize.COMMENT: "comment",
    }
    result = []
    cursor = 0
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        style = classes.get(token.type)
        if token.type == tokenize.NAME and keyword.iskeyword(token.string):
            style = "keyword"
        if not style:
            continue
        start = offsets[token.start[0] - 1] + token.start[1]
        end = offsets[token.end[0] - 1] + token.end[1]
        result.append(html.escape(source[cursor:start]))
        result.append(f'<span class="tok-{style}">{html.escape(source[start:end])}</span>')
        cursor = end
    result.append(html.escape(source[cursor:]))
    return "".join(result)


def build_gallery(cases_path: Path, output: Path) -> int:
    raw = cases_path.read_bytes()
    cases = json.loads(raw)
    if not isinstance(cases, list) or not cases:
        raise ValueError("The gallery needs a nonempty list of examples.")
    cards = []
    case_links = []
    identifiers = set()
    frameworks = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each example must be an object.")
        for key in ("id", "name", "framework", "question", "explanation", "before", "after"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError(f"Each example needs a nonempty {key} string.")
        identifier = case["id"]
        if not re.fullmatch(r"[a-z][a-z0-9-]*", identifier) or identifier in identifiers:
            raise ValueError(f"Invalid or duplicate example id: {identifier}")
        identifiers.add(identifier)
        frameworks.add(case["framework"])
        before, _ = analyze(case["before"], f"{identifier}/before.py")
        after, _ = analyze(case["after"], f"{identifier}/after.py")
        expected = case.get("expected")
        if (
            not isinstance(expected, list)
            or len(expected) != 2
            or [(finding.code, finding.severity) for finding in before] != [tuple(expected)]
            or after
        ):
            raise ValueError(f"Scanner results disagree with the advertised example: {identifier}")
        finding = before[0]
        if finding.severity not in ("warning", "review"):
            raise ValueError(f"Unsupported gallery severity: {finding.severity}")
        escape = html.escape
        panels = []
        for side, label in (("before", "Before"), ("after", "One explicit alternative")):
            diagnostic = (
                f"{finding.code} · {finding.severity} · line {finding.line}: {finding.message}"
                if side == "before"
                else "No covered finding in this snippet. This is not proof of repeatability."
            )
            panels.append(
                f'<section class="code-panel {side}" aria-label="{label}">'
                f'<div class="code-heading"><h3>{label}</h3>'
                f'<button type="button" class="copy" hidden data-copy="{identifier}-{side}" '
                f'aria-label="Copy {side} code for {escape(case["name"])}">Copy code</button></div>'
                f'<pre tabindex="0"><code id="{identifier}-{side}">'
                f"{highlight(case[side])}</code></pre>"
                f'<p class="diagnostic">{escape(diagnostic)}</p></section>'
            )
        cards.append(
            f'<article class="example" id="{identifier}" '
            f'data-framework="{escape(case["framework"])}" '
            f'data-severity="{finding.severity}">'
            f'<div class="example-heading"><div><span class="framework">'
            f"{escape(case['framework'])}</span><h2>{escape(case['name'])}</h2></div>"
            f'<a class="badge {finding.severity}" href="{REPO}/blob/main/docs/rules.md">'
            f"{finding.code} · {finding.severity}</a></div>"
            f'<p class="question">{escape(case["question"])}</p>'
            f'<div class="code-pair">{"".join(panels)}</div>'
            f'<p class="explanation">{escape(case["explanation"])}</p>'
            f'<a class="case-link" href="#{identifier}">Link to this example <span '
            f'aria-hidden="true">↗</span></a></article>'
        )
        case_links.append(
            f'<a href="#{identifier}" data-case="{identifier}">'
            f'<span class="nav-name">{escape(case["name"])}</span>'
            f'<span class="nav-framework">{escape(case["framework"])}</span>'
            f'<span class="nav-rule">{finding.code}</span></a>'
        )
    options = "".join(
        f'<option value="{html.escape(name)}">{html.escape(name)}</option>'
        for name in sorted(frameworks, key=str.casefold)
    )
    engine_hash = hashlib.sha256()
    for path in sorted((ROOT / "src/repro_lens").rglob("*.py")):
        engine_hash.update(path.relative_to(ROOT).as_posix().encode() + b"\0" + path.read_bytes())
    replacements = {
        "{{CARDS}}": "\n".join(cards),
        "{{CASE_LINKS}}": "\n".join(case_links),
        "{{OPTIONS}}": options,
        "{{COUNT}}": str(len(cases)),
        "{{FRAMEWORK_COUNT}}": str(len(frameworks)),
        "{{CASES_HASH}}": hashlib.sha256(raw).hexdigest(),
        "{{ENGINE_HASH}}": engine_hash.hexdigest(),
    }
    template = (ROOT / "site/index.html").read_text(encoding="utf-8")
    # Substitute once so example text that resembles a template token stays literal.
    rendered = re.sub(r"\{\{[A-Z_]+\}\}", lambda match: replacements[match[0]], template)
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.html").write_text(rendered, encoding="utf-8")
    for name in ("styles.css", "gallery.js", "checker.js", "checker-worker.mjs", "mark.svg"):
        shutil.copyfile(ROOT / "site" / name, output / name)
    # The live checker loads these exact analyzer sources in the browser.
    engine = output / "engine/repro_lens"
    engine.mkdir(parents=True, exist_ok=True)
    sources = sorted((ROOT / "src/repro_lens").glob("*.py"))
    for path in sources:
        shutil.copyfile(path, engine / path.name)
    manifest = {
        "files": [path.name for path in sources],
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
    }
    (output / "engine/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / ".nojekyll").touch()
    return len(cases)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "examples/framework_checks/cases.json")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/gallery")
    args = parser.parse_args()
    try:
        count = build_gallery(args.cases, args.output)
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"Built {count} checked examples in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
