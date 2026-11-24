// Smoke test for the playground: imports the built site's worker.js UNCHANGED
// as an ES module under Node (exactly as a browser module worker does), so it
// loads the site's own self-hosted pyodide/pyodide.mjs + runtime + numpy, then
// decodes every gallery entry through the worker's message protocol.
//
//   python tools/build_playground.py
//   python -m http.server -d _site 8000
//   node tools/playground_smoke.mjs http://localhost:8000/ smoke.json _site
//
// smoke.json gets the Pyodide-side results; then
//   python tools/build_playground.py --compare smoke.json
// checks them against CPython running the same files.
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [,, base, outPath = "playground-smoke.json", siteDir = "_site"] = process.argv;
const pending = new Map();
let nextId = 1, readyRes, readyRej;
const readyP = new Promise((r, j) => { readyRes = r; readyRej = j; });
// what a worker sees as `self`; fetches go to the served site over HTTP
globalThis.self = {
  location: { href: base + "worker.js" },
  postMessage(m) {
    if (m.type === "progress") {
      if (!m.total || m.loaded === m.total) console.log("progress:", m.label, m.loaded ?? "", m.total ?? "");
    } else if (m.type === "ready") { console.log("ready", JSON.stringify(m.info)); readyRes(); }
    else if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
    else { console.log("message", JSON.stringify(m)); if (m.type === "error") readyRej(m); }
  },
};
(async () => {
  const t0 = Date.now();
  await import(pathToFileURL(path.resolve(siteDir, "worker.js")).href);
  const ask = (text, cmd = "analyze") => new Promise((res) => {
    const id = nextId++; pending.set(id, res); self.onmessage({ data: { id, cmd, text } });
  });
  await readyP;
  console.log("boot ms", Date.now() - t0);
  const gallery = await (await fetch(base + "gallery.json")).json();
  const out = { gallery: [] };
  const t1 = Date.now();
  for (const g of gallery) {
    // live encoding in Pyodide must give exactly what CPython gave at build time
    const e = await ask(g.en, "encode");
    if (e.type !== "result") throw new Error(g.en + ": " + e.message);
    if (JSON.stringify(e.data.ugl) !== JSON.stringify(g.ugl))
      throw new Error(`encode differs for ${JSON.stringify(g.en)}:\n pyodide ${JSON.stringify(e.data.ugl)}\n cpython ${JSON.stringify(g.ugl)}`);
    const m = await ask(g.ugl.join("\n"));
    if (m.type !== "result") throw new Error(g.en + ": " + m.message);
    out.gallery.push({ en: g.en, ugl: g.ugl, ids: m.data.ids, sentences: m.data.sentences });
  }
  console.log("encoded + decoded", gallery.length, "entries in ms", Date.now() - t1, "(encodings identical to CPython)");
  out.bad = (await ask("n1 v444p n2344t\nnot_a_token v1\nn1 \\v20")).data;
  out.explain = (await ask("n988st v9 'John'y #9 n?blorft nv444t a6w \\v9")).data;
  fs.writeFileSync(outPath, JSON.stringify(out, null, 1));
  console.log("wrote", outPath);
  process.exit(0);
})().catch((e) => { console.error("FAIL", e); process.exit(1); });
