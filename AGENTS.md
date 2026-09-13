# Working on Repro Lens

Keep one implementation of each rule in `src/repro_lens`; CLI, hooks and skills use it.
The checker must not import project code, execute notebook cells or start training.
Keep `review` findings separate from confirmed policy violations. Do not call a clean
static scan proof of reproducibility, or a two-run match proof of scientific validity.

Add positive and negative behavioral cases when changing a detection rule. Preserve
aliases, shadowed identifiers, configured exclusions and justified suppressions.
Never autofix a scientific choice by inserting seed 42 or relaxing a tolerance.

The template is a working user project: validate a generated copy, not just its text.
Keep generated data/models and local evidence out of source control except tiny
deliberately versioned fixtures. Document new limits alongside new capabilities.

Use `uv run pytest`, `uv run ruff check .` and `uv run ruff format --check .`.

## Community support

Keep Repro Lens free and community-supported. Do not add paid pilots, consulting
offers, pricing, donation links, GitHub Sponsors configuration or payment forms.
The maintainer does not want to collect money through this project.

## Product presentation

The maintainer expects a polished visual design and a useful first experience.
Show a concrete problem, understandable evidence and a clear way to try the tool.
Preserve readable code, keyboard access, mobile layouts and shareable example links.
Review actual desktop and mobile rendering when changing the site. Prefer focused,
working interactions over decorative controls or a long undifferentiated feature list.
Claims of SOTA, superiority or adoption need comparative evidence; presentation alone
does not establish them. Keep demo outcomes and live execution clearly distinguished.
