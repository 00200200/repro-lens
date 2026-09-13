# Seven framework checks, before and after

From the repository root, with Python 3.11+ and uv:

```bash
uv run --no-dev python examples/framework_checks/demo.py
```

The demo scans 14 small source strings in [cases.json](cases.json), covering seven
checks across XGBoost, LightGBM, PyTorch, TensorFlow and Lightning. It asserts the
expected rule and severity for each initial example and no finding for its revised
example. A changed or missing detection makes the demo exit nonzero. The actual
findings are retained in `.repro-lens/framework-demo.json`.

No ML framework, data download, model training or API key is needed. These snippets
are deliberately incomplete experiment fragments; `seed`, data and other inputs
stand for the real project's choices. They are parsed as text, never executed.

The examples show choices to review, not automatic scientific fixes. Changing an
updater or disabling a performance feature may affect an experiment; capture a
baseline and validate the chosen change. A passing static example does not prove
repeatability. R105/R106/R109/R110 are advisory because settings may be controlled
elsewhere. See [coverage and primary references](../../docs/frameworks.md).
