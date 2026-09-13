# Browser example gallery

The public gallery at <https://00200200.github.io/repro-lens/> lets visitors browse
before/after snippets, filter by framework or finding, copy code and link a case.
The case navigator shows one selected example at a time, with syntax-colored code
whose copied text remains identical to the versioned source. On mobile the case
list scrolls horizontally. Direct links and browser history restore the chosen case.
It is a static page. It does not run Python, accept source uploads or claim to scan
code in the browser. All examples remain readable without JavaScript; search and
copy controls are progressively enabled.

The hero lets visitors switch between the three scripted edits in the
[agent review demo](../examples/agent_review/README.md). It presents that demo's
expected outcomes, already checked by CI: a refactor matches, a changed threshold
differs, and changing the tolerance changes the comparison contract. These are
eight-sample synthetic fixture results, not a live run, model benchmark or agent
evaluation. All three outcomes remain readable without JavaScript.

## Build and preview

From the repository root:

```bash
uv sync --locked
uv run python tools/build_gallery.py
python3 -m http.server 8765 --bind 127.0.0.1 --directory dist/gallery
```

Open <http://127.0.0.1:8765/>. The generated directory also works as a local static
artifact, with code selection as a fallback when clipboard access is unavailable.
There are no external scripts, fonts, analytics, API keys or ML dependencies.

## Add a case

Edit `examples/framework_checks/cases.json`. Each entry has a stable lowercase `id`,
`name`, `framework`, `question`, `explanation`, before/after source and an expected
`[rule, severity]` for the before snippet. Choose an id that can remain a shared
link. Explain the scientific or runtime tradeoff in the alternative; use `seed` as
a placeholder for a project owner's decision, not a universal seed value.

The builder uses `repro_lens.analysis.analyze`, the same analyzer as the CLI. It
fails if the before finding differs from the advertised result or the alternative
still has a finding. Both snippets are parsed as text, never executed. The gallery
records hashes of the example data and checker sources. The generated HTML is
escaped, including code and contributor-authored descriptions.

Run `uv run python examples/framework_checks/demo.py`, the gallery build, and the
repository tests and Ruff checks. Review desktop/mobile layout and exercise search,
combined filters, empty results, reset, code/command copying, scenario switching,
browser back/forward and a direct case link when changing
the interface. Build output belongs in ignored `dist/`, not Git.

## Publishing

The Checks workflow builds the gallery for PRs and pushes. On a successful push to
`main`, deployment waits for the core suite, XGBoost replay and gallery build, then
publishes the Pages artifact. GitHub Pages must use the GitHub Actions build source.
Only the deployment job receives Pages write and OIDC permissions. PRs cannot
deploy. GitHub Pages serves this project's public educational content for free;
the project offers community support and does not collect payments.
