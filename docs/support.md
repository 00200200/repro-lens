# Support Repro Lens and discuss a paid pilot

Repro Lens is MIT-licensed and free to use, including in commercial projects.
Bug reports and contributions are welcome through the usual GitHub issues and PRs.

## Need help reviewing an ML experiment?

If your team needs to know whether a refactor or an agent's edit changed experiment
outputs, you can [enquire about a paid pilot](https://github.com/00200200/repro-lens/issues/new?template=paid-pilot.yml).
Start with one experiment and a concrete decision you need to make.

A proposed pilot can cover:

| Work | Deliverable to agree before starting |
| --- | --- |
| Review one Python training entry point | A short report separating known static risks, unresolved settings and checks requiring execution |
| Define a replay contract | Declared inputs, metrics, artifact hashes, environment details and tolerances chosen with the project owner |
| Review a refactor against a baseline | Saved `verify` reports and a `compare` result explaining changed outputs or why the runs cannot be compared |
| Set up an advisory CI check | A reviewed workflow and handover notes describing which findings block a change and which require human review |

See the [runnable before/after example](../examples/agent_review/) for the evidence
the tool can produce today, and the [framework coverage](frameworks.md) for supported
APIs. A clean scan or two matching runs does not establish scientific validity.

The enquiry is an initial scoping conversation. Feasibility, deliverables, price,
timing and access arrangements are agreed before any paid work starts. There is
currently no hosted Repro Lens service or subscription checkout.

### How to enquire

Use the [pilot enquiry form](https://github.com/00200200/repro-lens/issues/new?template=paid-pilot.yml)
to describe the framework, the recurring problem and the result you need. GitHub
issues are public: describe private projects at a high level and keep source code,
datasets, credentials and personal contact details out of the form. An appropriate
private channel can be agreed before sharing restricted material.

Execution, if needed, is scoped to an agreed environment and compute budget.
The static checker does not run your project; `verify` explicitly runs the configured
command. The [verification contract](verification.md) explains what is recorded.

## Sponsorship

The repository's [funding configuration](../.github/FUNDING.yml) is prepared for
GitHub Sponsors account `00200200`. Sponsorship payments are not available yet;
the maintainer's Sponsors profile still needs activation. This page will be updated
with a working payment link once it is live.

Sponsorship would support maintenance, regression cases, framework coverage and
practical examples. A paid pilot has a separately agreed scope; sponsorship does
not purchase a particular fix, review outcome or support deadline.
