// State tests for site/checker.js in a Node VM with a fake page and Worker.
// Run with: node --test tests/checker_state.test.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const script = readFileSync(new URL("../site/checker.js", import.meta.url), "utf8");

function fakeElement(extra = {}) {
  const listeners = {};
  return Object.assign(
    {
      hidden: true,
      disabled: false,
      value: "",
      textContent: "",
      children: [],
      attributes: {},
      addEventListener(type, listener) {
        (listeners[type] ??= []).push(listener);
      },
      dispatch(type, event = {}) {
        return Promise.all((listeners[type] ?? []).map(listener => listener({ preventDefault() {}, ...event })));
      },
      setAttribute(name, value) {
        this.attributes[name] = String(value);
      },
      getAttribute(name) {
        return this.attributes[name];
      },
      replaceChildren(...nodes) {
        this.children = nodes;
      },
      append(...nodes) {
        this.children.push(...nodes);
      },
      focus() {},
      setSelectionRange() {},
      scrollIntoView() {},
    },
    extra,
  );
}

function page(WorkerImplementation) {
  const elements = {
    "#live-source": fakeElement({ value: "import random\nrandom.Random()\n" }),
    "#live-run": fakeElement(),
    "#live-share": fakeElement(),
    "#live-sample": fakeElement(),
    "#live-status": fakeElement(),
    "#live-output": fakeElement(),
    "#live-results": fakeElement({ attributes: { "aria-busy": "false" } }),
    "#check": fakeElement(),
    ".live-actions": fakeElement(),
  };
  const timers = [];
  vm.runInNewContext(script, {
    document: { querySelector: selector => elements[selector] ?? null, createElement: () => fakeElement() },
    Worker: WorkerImplementation,
    performance: { now: () => 0 },
    location: { hash: "", origin: "https://example.test", pathname: "/" },
    history: { replaceState() {} },
    navigator: {},
    getComputedStyle: () => ({ lineHeight: "24px" }),
    TextEncoder,
    TextDecoder,
    btoa,
    atob,
    setTimeout: (callback, ms) => timers.push({ callback, ms, cleared: false }),
    clearTimeout: id => {
      if (timers[id - 1]) timers[id - 1].cleared = true;
    },
  });
  return {
    run: elements["#live-run"],
    status: elements["#live-status"],
    results: elements["#live-results"],
    output: elements["#live-output"],
    timers,
    click: () => elements["#live-run"].dispatch("click"),
  };
}

class FakeWorker {
  static instances = [];

  constructor() {
    this.listeners = {};
    this.messages = [];
    this.terminated = false;
    FakeWorker.instances.push(this);
  }

  addEventListener(type, listener) {
    this.listeners[type] = listener;
  }

  postMessage(message) {
    this.messages.push(message);
    this.onPost?.(message);
  }

  terminate() {
    this.terminated = true;
  }
}

function assertControlsRestored(view) {
  assert.equal(view.run.disabled, false);
  assert.equal(view.results.getAttribute("aria-busy"), "false");
  assert.match(view.status.textContent, /install the CLI/);
}

test("a Worker constructor that throws restores the controls", async () => {
  const view = page(
    class {
      constructor() {
        throw new Error("Worker construction blocked");
      }
    },
  );
  await view.click();
  assertControlsRestored(view);
  assert.match(view.status.textContent, /could not start/);
  assert.ok(view.timers.every(timer => timer.cleared));
});

test("an asynchronous worker error restores the controls and stops the worker", async () => {
  FakeWorker.instances = [];
  class FailingWorker extends FakeWorker {
    onPost() {
      queueMicrotask(() => this.listeners.error({ preventDefault() {} }));
    }
  }
  const view = page(FailingWorker);
  await view.click();
  assertControlsRestored(view);
  assert.match(view.status.textContent, /could not start/);
  assert.equal(FakeWorker.instances[0].terminated, true);
});

test("a stalled runtime times out, then the next check starts a new worker", async () => {
  FakeWorker.instances = [];
  const view = page(FakeWorker);
  const clicked = view.click();
  assert.equal(view.run.disabled, true);
  assert.equal(view.results.getAttribute("aria-busy"), "true");
  const [timer] = view.timers;
  assert.equal(timer.ms, 120_000);
  timer.callback();
  await clicked;
  assertControlsRestored(view);
  assert.match(view.status.textContent, /took too long/);
  assert.equal(FakeWorker.instances[0].terminated, true);

  const retried = view.click();
  assert.equal(FakeWorker.instances.length, 2);
  assert.equal(view.timers[1].ms, 120_000);
  view.timers[1].callback();
  await retried;
});

test("a successful check renders findings and uses the shorter timeout afterwards", async () => {
  FakeWorker.instances = [];
  class ReplyingWorker extends FakeWorker {
    onPost({ id }) {
      const finding = {
        code: "R103",
        severity: "warning",
        line: 2,
        column: 1,
        message: "random.Random has no explicit non-None seed.",
        suggestion: "Pass the experiment's seed/RNG.",
      };
      queueMicrotask(() => this.listeners.message({ data: { id, result: { findings: [finding], suppressed: [] } } }));
    }
  }
  const view = page(ReplyingWorker);
  await view.click();
  assert.equal(view.run.disabled, false);
  assert.equal(view.results.getAttribute("aria-busy"), "false");
  assert.match(view.status.textContent, /Checked in .*including loading Python/);
  assert.ok(view.output.children.length > 0);
  await view.click();
  assert.equal(FakeWorker.instances.length, 1);
  assert.deepEqual(view.timers.map(timer => [timer.ms, timer.cleared]), [[120_000, true], [20_000, true]]);
});

test("an unexpected result restores the controls", async () => {
  class MalformedWorker extends FakeWorker {
    onPost({ id }) {
      queueMicrotask(() => this.listeners.message({ data: { id, result: {} } }));
    }
  }
  const view = page(MalformedWorker);
  await view.click();
  assertControlsRestored(view);
  assert.match(view.status.textContent, /live check failed/);
});
