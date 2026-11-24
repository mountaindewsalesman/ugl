// Runs the real ugl package in Pyodide, off the main thread.
// This is a MODULE worker (new Worker(url, {type: "module"})): Pyodide 0.28+
// only supports module workers and is loaded as an ES module.
// Messages in:  {id, cmd: "analyze" | "encode", text}
// Messages out: {type: "progress", label, loaded?, total?}
//               {type: "ready", info}
//               {type: "result", id, data} | {type: "error", id?, message}
//
// Pyodide is served from this site (tools/build_playground.py downloads this
// exact version into pyodide/), so nothing is loaded cross-origin.  It finds
// its runtime files next to pyodide.mjs on its own.
import { loadPyodide } from "./pyodide/pyodide.mjs";

const PYODIDE_VERSION = "314.0.7";
const LIB = "/home/pyodide/lib";
const here = (path) => new URL(path, self.location.href).href;
const progress = (label, loaded, total) => self.postMessage({ type: "progress", label, loaded, total });

async function fetchBytes(url, label, expected) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${url}: HTTP ${resp.status}`);
  const total = Number(resp.headers.get("content-length")) || expected || 0;
  if (!resp.body || !resp.body.getReader) return new Uint8Array(await resp.arrayBuffer());
  const reader = resp.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    progress(label, loaded, total);
  }
  const out = new Uint8Array(loaded);
  let at = 0;
  for (const c of chunks) { out.set(c, at); at += c.length; }
  return out;
}

async function boot() {
  progress("Loading manifest");
  const manifest = await (await fetch(here("manifest.json"))).json();

  // Start the 10 MB download right away, in parallel with Pyodide itself.
  const bundle = fetchBytes(here(manifest.bundle), "Downloading UGL code + dictionary", manifest.bundle_bytes);
  const glueSrc = fetch(here("playground.py")).then((r) => r.text());

  progress("Starting Python (Pyodide " + PYODIDE_VERSION + ")");
  const py = await loadPyodide();
  progress("Installing numpy + lemminflect");
  await py.loadPackage(["numpy", ...manifest.wheels.map(here)], { messageCallback: () => {} });

  const zip = await bundle;
  progress("Unpacking UGL");
  py.FS.mkdirTree(LIB);
  py.unpackArchive(zip, "zip", { extractDir: LIB });
  py.FS.writeFile(LIB + "/playground.py", await glueSrc);
  py.runPython(`import sys\nif ${JSON.stringify(LIB)} not in sys.path: sys.path.insert(0, ${JSON.stringify(LIB)})`);

  progress("Loading dictionary " + manifest.dictionary);
  const glue = py.pyimport("playground");
  const info = JSON.parse(glue.init());
  info.pyodide = PYODIDE_VERSION;
  info.python = py.runPython("import sys; sys.version.split()[0]");
  return { glue, info };
}

const ready = boot();
ready.then(
  ({ info }) => self.postMessage({ type: "ready", info }),
  (err) => self.postMessage({ type: "error", message: "Startup failed: " + (err && err.message || err) }),
);

self.onmessage = async (ev) => {
  const { id, cmd, text } = ev.data;
  try {
    const { glue } = await ready;
    if (cmd !== "analyze" && cmd !== "encode") throw new Error("unknown command " + cmd);
    self.postMessage({ type: "result", id, data: JSON.parse(glue[cmd](text)) });
  } catch (err) {
    self.postMessage({ type: "error", id, message: String(err && err.message || err) });
  }
};
