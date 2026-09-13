"use strict";

(() => {
  const source = document.querySelector("#live-source");
  if (!source) return;
  const run = document.querySelector("#live-run");
  const share = document.querySelector("#live-share");
  const sample = document.querySelector("#live-sample");
  const status = document.querySelector("#live-status");
  const output = document.querySelector("#live-output");
  const results = document.querySelector("#live-results");
  const example = source.value;
  const MAX_SOURCE = 200_000;
  const MAX_SHARE = 8_000;
  let worker;
  let checks = 0;
  let pending;

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function setStatus(text) {
    status.textContent = text;
  }

  function startWorker() {
    worker = new Worker("checker-worker.mjs", { type: "module" });
    worker.addEventListener("message", event => pending?.(event.data));
    worker.addEventListener("error", event => {
      event.preventDefault();
      pending?.({ error: "The live checker could not start in this browser or build." });
      worker.terminate();
      worker = undefined;
    });
  }

  function check(text) {
    if (!worker) startWorker();
    const id = ++checks;
    return new Promise(resolve => {
      pending = data => {
        if (data.id === undefined || data.id === id) resolve(data);
      };
      worker.postMessage({ id, source: text });
    });
  }

  function selectLine(line) {
    const lines = source.value.split("\n");
    const start = lines.slice(0, line - 1).reduce((total, text) => total + text.length + 1, 0);
    source.focus();
    source.setSelectionRange(start, start + (lines[line - 1] || "").length);
    const lineHeight = parseFloat(getComputedStyle(source).lineHeight) || 24;
    source.scrollTop = Math.max(0, (line - 3) * lineHeight);
  }

  function render(result) {
    const findings = result.findings;
    const counts = { warning: 0, review: 0, error: 0 };
    for (const finding of findings) counts[finding.severity] = (counts[finding.severity] || 0) + 1;
    output.replaceChildren();
    const summary = element("div", "live-summary");
    if (findings.length) {
      const parts = Object.entries(counts)
        .filter(([, total]) => total)
        .map(([severity, total]) => `${total} ${severity}${total === 1 ? "" : "s"}`);
      summary.append(element("strong", "", `${findings.length} finding${findings.length === 1 ? "" : "s"}`));
      summary.append(element("span", "", parts.join(" · ")));
    } else {
      summary.append(element("strong", "", "No covered finding"));
      summary.append(element("span", "", "This is not proof that the experiment is repeatable."));
    }
    output.append(summary);
    if (findings.length) {
      const list = element("ol", "finding-list");
      for (const finding of findings) {
        const item = element("li", "finding");
        const top = element("div", "finding-top");
        top.append(element("span", `badge ${finding.severity}`, `${finding.code} · ${finding.severity}`));
        const jump = element("button", "line-button", `Line ${finding.line}, column ${finding.column}`);
        jump.type = "button";
        jump.setAttribute("aria-label", `Select line ${finding.line} in the editor`);
        jump.addEventListener("click", () => selectLine(finding.line));
        top.append(jump);
        item.append(top, element("p", "", finding.message), element("p", "suggestion", finding.suggestion));
        list.append(item);
      }
      output.append(list);
    }
    if (result.suppressed.length) {
      const total = result.suppressed.length;
      output.append(element("p", "live-note", `${total} finding${total === 1 ? " is" : "s are"} suppressed with a justification.`));
    }
    const rules = element("a", "live-note", "What the rules mean ↗");
    rules.href = "https://github.com/00200200/repro-lens/blob/main/docs/rules.md";
    output.append(rules);
  }

  async function runCheck() {
    const text = source.value;
    if (!text.trim()) {
      setStatus("Paste some Python first.");
      return;
    }
    if (text.length > MAX_SOURCE) {
      setStatus("That file is too large for the live check. Use the CLI instead.");
      return;
    }
    const first = !worker;
    run.disabled = true;
    results.setAttribute("aria-busy", "true");
    setStatus(first ? "Loading Python in your browser. This happens once per visit…" : "Checking…");
    const started = performance.now();
    const data = await check(text);
    results.setAttribute("aria-busy", "false");
    run.disabled = false;
    if (data.error) {
      output.replaceChildren();
      setStatus(`${data.error} You can still install the CLI and run it locally.`);
      return;
    }
    render(data.result);
    const elapsed = performance.now() - started;
    const took = elapsed < 1000 ? `${Math.max(1, Math.round(elapsed))} ms` : `${(elapsed / 1000).toFixed(1)} s`;
    setStatus(first ? `Checked in ${took}, including loading Python.` : `Checked in ${took}.`);
  }

  function encode(text) {
    const bytes = new TextEncoder().encode(text);
    let binary = "";
    for (const byte of bytes) binary += String.fromCharCode(byte);
    return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
  }

  function decode(value) {
    const binary = atob(value.replaceAll("-", "+").replaceAll("_", "/"));
    return new TextDecoder("utf-8", { fatal: true }).decode(Uint8Array.from(binary, char => char.charCodeAt(0)));
  }

  async function copyShareLink() {
    if (source.value.length > MAX_SHARE) {
      setStatus(`Share links hold up to ${MAX_SHARE.toLocaleString()} characters. Shorten the snippet first.`);
      return;
    }
    // The code travels in the URL fragment, which browsers do not send to the server.
    const url = `${location.origin}${location.pathname}#check=${encode(source.value)}`;
    history.replaceState(null, "", url);
    try {
      await navigator.clipboard.writeText(url);
      setStatus("Share link copied. Anyone who opens it sees this code, not results.");
    } catch {
      setStatus("Your browser blocked the clipboard. Copy the link from the address bar.");
    }
  }

  if (location.hash.startsWith("#check=")) {
    try {
      source.value = decode(location.hash.slice("#check=".length));
      setStatus("Shared code loaded. Press Check code to scan it in your browser.");
      document.querySelector("#check").scrollIntoView({ block: "start" });
    } catch {
      setStatus("This share link is damaged, so the example is shown instead.");
    }
  }

  document.querySelector(".live-actions").hidden = false;
  sample.hidden = false;
  run.addEventListener("click", runCheck);
  share.addEventListener("click", copyShareLink);
  sample.addEventListener("click", () => {
    source.value = example;
    output.replaceChildren();
    setStatus("Example restored. Press Check code.");
    source.focus();
  });
  source.addEventListener("keydown", event => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      if (!run.disabled) runCheck();
    }
  });
})();
