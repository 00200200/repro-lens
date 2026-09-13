# One engine, four entry points

The Python package owns static rules, policy loading, report schemas and experiment replay. The pre-commit hook invokes the CLI. The Agent Skill invokes the same package through a small launcher. The scikit-learn template exercises the report contract in a runnable project.

These components share a repository because a change to the verification contract needs corresponding tests, documentation and template changes. The template can become a separate project if it develops its own release cycle and maintainers. It can already be generated into an independent directory.

## Existing work

- [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) provides an established project layout.
- [dslinter](https://github.com/SERG-Delft/dslinter) includes ML-specific checks, including randomness.
- [mllint](https://bvobart.github.io/mllint/docs/) checks project quality and produces reports.
- [UTMIST research-skills](https://github.com/UTMIST/research-skills) includes repository reproducibility audits.
- [experiment-agent](https://github.com/Imbad0202/experiment-agent) covers experiment execution and interpretation.
- [OMDS](https://github.com/spkc83/omds) combines CLIs, skills and methodology checks.

This documentation review informed the scope; it is not a comparative accuracy benchmark. Repro Lens focuses on a dependency-light checker, explicit uncertainty, and retained evidence from local replay. False-positive rates on independent projects still need measurement.

## Boundaries

Static screening never imports target code or runs training. Replay is an explicit command and is not a sandbox. The agent interprets evidence and works within the user's requested scope. Experiment policy lives in pyproject.toml; there is no mandatory folder layout for existing projects.

The current AST implementation deliberately stops short of full Python data-flow analysis. Improving lexical scope handling, notebook locations and reviewed framework rules are follow-up work. Every expansion needs passing examples as well as violations.
