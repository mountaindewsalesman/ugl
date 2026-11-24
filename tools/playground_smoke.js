// Smoke test for the playground: runs site/worker.js UNCHANGED under Node with
// the real Pyodide (npm package, same version as the page), against a built
// site served over HTTP, and decodes every gallery entry.
//
//   python tools/build_playground.py
//   python -m http.server -d _site 8000
//   npm install pyodide@314.0.7        (in any folder; set NODE_PATH to it)
//   node tools/playground_smoke.js http://localhost:8000/ smoke.json
//
// smoke.json gets the Pyodide-side results; then
//   python tools/build_playground.py --compare smoke.json
// checks them against CPython running the same files.
const fs = require("fs"), vm = require("vm");
const [,, base, outPath = "playground-smoke.json"] = process.argv;
const { loadPyodide } = require("pyodide");
const pending = new Map();
let nextId = 1, readyRes, readyRej;
const readyP = new Promise((r, j) => { readyRes = r; readyRej = j; });
const self = {
  location: { href: base + "worker.js" },
  postMessage(m) {
    if (m.type === "progress") {
      if (!m.total || m.loaded === m.total) console.log("progress:", m.label, m.loaded ?? "", m.total ?? "");
    } else if (m.type === "ready") { console.log("ready", JSON.stringify(m.info)); readyRes(); }
    else if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
    else { console.log("message", JSON.stringify(m)); if (m.type === "error") readyRej(m); }
  },
};
const ctx = {
  self, fetch, URL, console, Uint8Array, JSON, Number, String, Error, Promise, setTimeout,
  // the page loads pyodide.js from the CDN; here the npm package of the same version
  importScripts: () => { ctx.loadPyodide = (opts) => loadPyodide({ ...opts, indexURL: undefined }); },
};
vm.createContext(ctx);
(async () => {
  const src = await (await fetch(base + "worker.js")).text();   // the served file
  vm.runInContext(src, ctx, { filename: "worker.js" });
  const ask = (text) => new Promise((res) => {
    const id = nextId++; pending.set(id, res); ctx.self.onmessage({ data: { id, cmd: "analyze", text } });
  });
  const t0 = Date.now();
  await readyP;
  console.log("boot ms", Date.now() - t0);
  const gallery = await (await fetch(base + "gallery.json")).json();
  const out = { gallery: [] };
  const t1 = Date.now();
  for (const g of gallery) {
    const m = await ask(g.ugl.join("\n"));
    if (m.type !== "result") throw new Error(g.en + ": " + m.message);
    out.gallery.push({ en: g.en, ugl: g.ugl, ids: m.data.ids, sentences: m.data.sentences });
  }
  console.log("decoded", gallery.length, "entries in ms", Date.now() - t1);
  out.bad = (await ask("n1 v444p n2344t\nnot_a_token v1\nn1 \\v20")).data;
  out.explain = (await ask("n988st v9 'John'y #9 n?blorft nv444t a6w \\v9")).data;
  fs.writeFileSync(outPath, JSON.stringify(out, null, 1));
  console.log("wrote", outPath);
  process.exit(0);
})().catch((e) => { console.error("FAIL", e); process.exit(1); });
