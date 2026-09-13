# Browser example gallery

The public gallery at <https://00200200.github.io/repro-lens/> lets visitors browse
before/after snippets, filter by framework or finding, copy code and link a case.
The case navigator shows one selected example at a time, with syntax-colored code
whose copied text remains identical to the versioned source. On mobile the case
list scrolls horizontally. Direct links and browser history restore the chosen case.
The gallery results are saved at build time. All examples remain readable without
JavaScript; search and copy controls are progressively enabled.

## Live check

The **Check your code** section runs `repro_lens.analysis.analyze` on pasted source
in the visitor's browser. Nothing is uploaded: a module worker loads
[Pyodide](https://pyodide.org/) and the analyzer files from the same site only when
someone presses **Check code** (about 13 MB before compression). The source is
parsed as text and never executed. Findings show the rule, severity, message,
suggestion and a button that selects the line; justified suppressions are counted.
A share link stores up to 8,000 characters of code in the URL fragment, which the
browser does not send to the server; opening it loads the code but does not start a
check or download Python.

`tools/build_gallery.py` copies `src/repro_lens/*.py` into `engine/` with SHA-256
hashes in `engine/manifest.json`. `tools/fetch_pyodide.py` downloads the pinned npm
package, verifies its integrity hash, and extracts only the five runtime files.
`tools/smoke_live_checker.mjs` loads that runtime in Node, checks the engine hashes
and confirms known findings. Live results cover the same selected APIs as the CLI;
notebooks, wrappers and runtime seeding are not analyzed, and a clean result is not
proof of repeatability. If the worker cannot be created, reports an error, or does not
answer within 120 seconds for the first check (which includes the download) or 20
seconds afterwards, the worker is stopped, the controls are restored and the page
suggests the CLI; the next check starts a new worker. `tests/checker_state.test.mjs`
covers these paths with a fake page and Worker (`node --test`, also run by pytest
when Node.js is installed). It is a JavaScript state test, not a Safari or Firefox run.

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
uv run python tools/fetch_pyodide.py
node tools/smoke_live_checker.mjs dist/gallery
python3 -m http.server 8765 --bind 127.0.0.1 --directory dist/gallery
```

Open <http://127.0.0.1:8765/>. Without `fetch_pyodide.py` the gallery still works and
the live check reports that it could not start. Serve the directory over HTTP for the
live check; module workers do not load from `file://`. Code selection is the fallback
when clipboard access is unavailable. There are no third-party scripts, fonts,
analytics, API keys or ML dependencies; Pyodide is served from the site itself.

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
the interface. For the live check, also try an unseeded and a seeded snippet, a syntax
error, a suppression, the line button, Cmd/Ctrl+Enter and a share link. Build output belongs in ignored `dist/`, not Git.

## Publishing

The Checks workflow builds the gallery for PRs and pushes. On a successful push to
`main`, deployment waits for the core suite, XGBoost replay and gallery build, then
publishes the Pages artifact. GitHub Pages must use the GitHub Actions build source.
Only the deployment job receives Pages write and OIDC permissions. PRs cannot
deploy. GitHub Pages serves this project's public educational content for free;
the project offers community support and does not collect payments.

The gallery job runs `tools/fetch_pyodide.py` and
`node tools/smoke_live_checker.mjs dist/gallery` after building the gallery and before
uploading the Pages artifact. A runtime integrity failure or unexpected engine
result fails the job and prevents deployment.
