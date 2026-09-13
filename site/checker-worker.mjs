// Runs the Repro Lens analyzer with Pyodide in a worker. Source text never leaves the page.
import { loadPyodide } from "./pyodide/pyodide.mjs";

const ready = (async () => {
  const pyodide = await loadPyodide({ indexURL: new URL("./pyodide/", import.meta.url).href });
  const manifest = await (await fetch(new URL("./engine/manifest.json", import.meta.url))).json();
  pyodide.FS.mkdirTree("/engine/repro_lens");
  for (const name of manifest.files) {
    const response = await fetch(new URL(`./engine/repro_lens/${name}`, import.meta.url));
    if (!response.ok) throw new Error(`Could not load checker file ${name}`);
    pyodide.FS.writeFile(`/engine/repro_lens/${name}`, await response.text());
  }
  pyodide.runPython(`
import json, sys
sys.path.insert(0, "/engine")
from repro_lens.analysis import analyze

def check_source(source):
    active, suppressed = analyze(source, "your_code.py")
    return json.dumps({
        "findings": [finding.to_dict() for finding in active],
        "suppressed": [finding.to_dict() for finding in suppressed],
    })
`);
  return pyodide.globals.get("check_source");
})();

self.addEventListener("message", async event => {
  try {
    const checkSource = await ready;
    self.postMessage({ id: event.data.id, result: JSON.parse(checkSource(event.data.source)) });
  } catch (error) {
    self.postMessage({ id: event.data.id, error: String(error?.message || error) });
  }
});
