// Playground UI.  All UGL logic runs in worker.js (the real ugl package in
// Pyodide); this file only renders what it returns.
"use strict";

const $ = (id) => document.getElementById(id);

function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid === undefined || kid === null || kid === false) continue;
    e.append(kid instanceof Node ? kid : String(kid));
  }
  return e;
}

function debounce(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

const norm = (s) => s.toLowerCase().replace(/\s+/g, " ").trim();

// ------------------------------------------------------------ worker
const state = { gallery: [], manifest: null, ready: false, failed: false, seq: 0, shown: 0 };
const worker = new Worker("worker.js", { type: "module" });
const waiting = new Map();

worker.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === "progress") return showProgress(m.label, m.loaded, m.total);
  if (m.type === "ready") return onReady(m.info);
  if (m.id && waiting.has(m.id)) {
    const { resolve, reject } = waiting.get(m.id);
    waiting.delete(m.id);
    return m.type === "result" ? resolve(m.data) : reject(new Error(m.message));
  }
  if (m.type === "error") onFatal(m.message);
};
worker.onerror = (e) => onFatal(e.message || "the Python worker could not start");

function ask(text, cmd = "analyze") {
  const id = ++state.seq;
  return new Promise((resolve, reject) => {
    waiting.set(id, { resolve, reject });
    worker.postMessage({ id, cmd, text });
  }).then((data) => ({ id, data }));
}

// ------------------------------------------------------------ status
function showProgress(label, loaded, total) {
  $("status-text").textContent = label;
  const fill = $("bar-fill");
  if (total) {
    fill.classList.remove("indeterminate");
    fill.style.width = Math.min(100, (100 * loaded) / total).toFixed(1) + "%";
    $("status-num").textContent = `${(loaded / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`;
  } else {
    fill.classList.add("indeterminate");
    fill.style.width = "";
    $("status-num").textContent = "";
  }
}

function onReady(info) {
  state.ready = true;
  const s = $("status");
  s.classList.add("done");
  $("status-text").textContent =
    `Ready: ugl ${info.ugl_version}, dictionary ${info.dictionary_version}, ` +
    `${info.vocab_size.toLocaleString()} token ids, Python ${info.python} (Pyodide ${info.pyodide})`;
  $("status-num").textContent = "";
  if ($("en").value.trim()) onEnglish(); else analyze();
}

function onFatal(msg) {
  state.failed = true;
  const s = $("status");
  s.classList.add("failed");
  $("status-text").textContent = "Could not start Python: " + msg;
  $("status-num").textContent = "";
  $("bar-fill").classList.remove("indeterminate");
}

// ------------------------------------------------------------ inputs
function loadExample(g) {
  $("en").value = g.en;
  onEnglish();
}

function setUGL(text) {
  $("ugl").value = text;
  analyze();
}

let encodeSeq = 0;
async function onEnglish() {
  const text = $("en").value;
  const note = $("en-note");
  if (!text.trim()) { note.hidden = true; return; }
  if (!state.ready) { note.hidden = false; note.textContent = "Waiting for Python to load..."; return; }
  const mine = ++encodeSeq;
  note.hidden = false;
  note.textContent = "Encoding...";
  try {
    const { data } = await ask(text, "encode");
    if (mine !== encodeSeq) return;            // a newer edit is on its way
    note.textContent = data.unknown.length ? `Not in the dictionary (kept as literals): ${data.unknown.join(", ")}` : "";
    note.hidden = !data.unknown.length;
    setUGL(data.ugl.join("\n"));
  } catch (err) {
    if (mine === encodeSeq) { note.hidden = false; note.textContent = "Could not encode: " + (err.message || err); }
  }
}

async function analyze() {
  const text = $("ugl").value;
  if (!text.trim()) {
    $("results").replaceChildren();
    $("ids").replaceChildren(el("p", { class: "hint" }, "No UGL yet."));
    return;
  }
  if (!state.ready) {
    if (!state.failed) $("results").replaceChildren(el("p", { class: "hint" }, "Waiting for Python to load..."));
    return;
  }
  try {
    const { id, data } = await ask(text);
    if (id < state.shown) return;          // a newer result is already on screen
    state.shown = id;
    renderResults(data);
    renderIds(data);
  } catch (err) {
    $("results").replaceChildren(el("p", { class: "error" }, String(err.message || err)));
  }
}

// ------------------------------------------------------------ rendering
function chipClass(t) {
  if (t.kind === "word") return "k-" + t.cls;
  if (t.kind === "close") return "k-close";
  if (t.kind === "bad") return "k-bad";
  return "k-lit";
}

function renderResults(data) {
  const cards = data.sentences.map((s, i) => {
    const inspector = el("div", { class: "inspector" }, el("p", { class: "hint" }, "Hover or tap a token to explain it."));
    const chips = s.tokens.map((t, j) => {
      const b = el("button", { type: "button", class: "chip " + chipClass(t), "aria-label": `${t.token}: ${t.lemma || t.class}` }, t.token);
      const show = () => {
        for (const c of b.parentNode.children) c.classList.toggle("sel", c === b);
        inspector.replaceChildren(explain(t, s.ids && s.ids[j]));
      };
      b.addEventListener("mouseenter", show);
      b.addEventListener("focus", show);
      b.addEventListener("click", show);
      return b;
    });
    const rows = [el("div", { class: "row" }, el("span", { class: "tag" }, "UGL"), el("div", { class: "chips tokens" }, chips))];
    if (s.error) {
      rows.push(el("div", { class: "row" }, el("span", { class: "tag" }, "ERROR"), el("span", { class: "error" }, s.error)));
    } else {
      rows.push(el("div", { class: "row" }, el("span", { class: "tag" }, "READ"), el("span", { class: "readable" }, s.readable)));
      rows.push(el("div", { class: "row" }, el("span", { class: "tag" }, "GLOSS"), el("span", { class: "mono gloss" }, s.gloss)));
      rows.push(el("div", { class: "row" }, el("span", { class: "tag" }, "BACK"),
        s.back_error ? el("span", { class: "error" }, "decode error: " + s.back_error) : el("span", { class: "back" }, s.back)));
    }
    return el("article", { class: "card" }, data.sentences.length > 1 ? el("div", { class: "num" }, `Sentence ${i + 1}`) : null, rows, inspector);
  });
  $("results").replaceChildren(...cards);
}

function explain(t, ids) {
  const dl = el("dl");
  const add = (k, v) => { if (v !== undefined && v !== null && v !== "") dl.append(el("dt", {}, k), el("dd", {}, v)); };
  if (t.kind === "word") {
    add("meaning", t.lemma);
    add("concept", t.id + (t.reserved ? ` (${t.reserved})` : ""));
    add("sense", t.sense);
    add("gloss", t.gloss);
    if (t.freq !== undefined) add("corpus freq.", Math.round(t.freq).toLocaleString());
    if (t.missing) add("note", "not in this dictionary");
  } else if (t.kind === "close") {
    add("closes", `${t.closes.id} (${t.closes.lemma})`);
  } else if (t.kind !== "bad") {
    add("text", t.lemma);
  }
  if (t.mods && t.mods.length) {
    add("modifiers", el("ul", { class: "mods" }, t.mods.map((m) => el("li", {}, el("code", {}, m.letter), " " + m.meaning))));
  }
  if (ids) add("ids", el("span", { class: "mono" }, ids.ids.map((n, k) => `${n} ${ids.pieces[k]}`).join(" | ")));
  return el("div", {}, el("p", { class: "insp-head" }, el("code", {}, t.token), " " + t.class), dl);
}

function renderIds(data) {
  const blocks = data.sentences.filter((s) => s.ids).map((s) =>
    el("div", { class: "idline" }, s.ids.map((g) =>
      el("span", { class: "idgroup", title: g.pieces.join(" ") },
        el("span", { class: "idtok" }, g.token), el("span", { class: "idnums" }, g.ids.join(" "))))));
  const full = el("code", { id: "ids-full" }, JSON.stringify(data.ids));
  $("ids").replaceChildren(...blocks,
    el("p", { class: "hint" }, `All sentences with <bos>, <sep> and <eos>: ${data.ids.length} ids`),
    el("pre", { class: "cmd" }, full, el("button", { class: "copy", type: "button", "data-copy": "ids-full" }, "Copy")));
}

// ------------------------------------------------------------ setup
document.addEventListener("click", async (ev) => {
  const b = ev.target.closest("button.copy");
  if (!b) return;
  try {
    await navigator.clipboard.writeText($(b.dataset.copy).textContent);
    b.textContent = "Copied";
  } catch {
    b.textContent = "Select + copy";
  }
  setTimeout(() => { b.textContent = "Copy"; }, 1500);
});

$("en").addEventListener("input", debounce(onEnglish, 250));
const analyzeSoon = debounce(analyze, 200);
$("ugl").addEventListener("input", () => {
  encodeSeq++;            // the user is editing UGL directly: drop any pending English encode
  analyzeSoon();
});

async function init() {
  const [manifest, gallery] = await Promise.all([
    fetch("manifest.json").then((r) => r.json()),
    fetch("gallery.json").then((r) => r.json()),
  ]);
  state.manifest = manifest;
  state.gallery = gallery;
  $("gallery-count").textContent = gallery.length;
  $("examples").replaceChildren(...gallery.slice(0, 8).map((g) =>
    el("button", { type: "button", class: "chip example", onclick: () => loadExample(g) }, g.en)));
  $("gallery-list").replaceChildren(...gallery.map((g) =>
    el("li", {}, el("button", { type: "button", class: "linkish", onclick: () => loadExample(g) },
      el("span", {}, g.en), el("code", {}, g.ugl.join("  /  "))))));
  const commit = manifest.commit ? manifest.commit.slice(0, 7) : "local";
  const commitLink = manifest.commit
    ? el("a", { href: `https://github.com/mountaindewsalesman/ugl/commit/${manifest.commit}` }, commit) : commit;
  $("footer").replaceChildren(el("p", {},
    "Built ", manifest.built, " from ", commitLink, ". Parser ", manifest.gallery.parser,
    ", ", manifest.gallery.sense_choice, ". ",
    el("a", { href: "https://github.com/mountaindewsalesman/ugl" }, "Source on GitHub"), "."));
  loadExample(gallery[0]);
}
init().catch((err) => onFatal("could not load the site data (" + err.message + ")"));
