// Load the built gallery's Pyodide runtime and analyzer files, as the browser worker does,
// and check that a known snippet produces the CLI's findings. Usage:
//   node tools/smoke_live_checker.mjs dist/gallery
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const site = resolve(process.argv[2] ?? "dist/gallery");
const { loadPyodide } = await import(pathToFileURL(`${site}/pyodide/pyodide.mjs`).href);
const pyodide = await loadPyodide({ indexURL: `${site}/pyodide/` });
const manifest = JSON.parse(readFileSync(`${site}/engine/manifest.json`, "utf8"));
pyodide.FS.mkdirTree("/engine/repro_lens");
for (const name of manifest.files) {
  const data = readFileSync(`${site}/engine/repro_lens/${name}`);
  if (createHash("sha256").update(data).digest("hex") !== manifest.sha256[name]) {
    throw new Error(`Engine file does not match its manifest hash: ${name}`);
  }
  pyodide.FS.writeFile(`/engine/repro_lens/${name}`, data);
}
pyodide.globals.set(
  "source",
  "import numpy as np\nfrom sklearn.model_selection import train_test_split\n"
    + "train_test_split(X, y)\nnp.random.default_rng()\ndef broken(:\n",
);
pyodide.globals.set("valid", "import random\nrandom.Random()\n");
const result = JSON.parse(pyodide.runPython(`
import json, sys
sys.path.insert(0, "/engine")
from repro_lens.analysis import analyze
json.dumps({
    "python": sys.version.split()[0],
    "invalid": [f.code for f in analyze(source, "your_code.py")[0]],
    "valid": [(f.code, f.line) for f in analyze(valid, "your_code.py")[0]],
})
`));
const expected = JSON.stringify({ invalid: ["S902"], valid: [["R103", 2]] });
const actual = JSON.stringify({ invalid: result.invalid, valid: result.valid });
if (actual !== expected) throw new Error(`Unexpected live checker result: ${actual}`);
console.log(`Live checker works with Python ${result.python} in Pyodide: ${actual}`);
