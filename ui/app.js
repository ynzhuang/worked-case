"use strict";
/* Single-page client over the API.
 *
 * Four panels matching the pipeline: source text beside the extracted event
 * object, the phenotype definition as controls bound to the YAML, retrieval
 * with a visible assertion filter, and the agent behind an approval gate.
 *
 * No build step and no external assets: the whole thing has to run with the
 * network cable pulled.
 */

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

async function api(path, options) {
  const response = await fetch(path, options);
  let body = null;
  try { body = await response.json(); } catch (_) { body = null; }
  return { ok: response.ok, status: response.status, body };
}

const state = {
  definitions: [],
  current: null,      // the definition version selected
  frozen: null,       // the frozen baseline it is compared against
  candidate: null,
  question: null,
};

/* ---------------------------------------------------------------- tabs */
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    $(tab.dataset.panel).classList.add("active");
  });
});

/* ------------------------------------------------------------ 1. source */

/** Render text with every span highlighted, nesting resolved by splitting
 *  at every boundary so overlapping spans never lose a character. */
function renderWithSpans(text, spans) {
  const container = el("div");
  const local = spans.filter((s) => s.end > s.start && s.end <= text.length);
  if (!local.length) { container.textContent = text; return container; }

  const cuts = new Set([0, text.length]);
  local.forEach((s) => { cuts.add(s.start); cuts.add(s.end); });
  const points = [...cuts].sort((a, b) => a - b);

  for (let i = 0; i < points.length - 1; i++) {
    const [from, to] = [points[i], points[i + 1]];
    const chunk = text.slice(from, to);
    if (!chunk) continue;
    const covering = local.filter((s) => s.start <= from && s.end >= to);
    if (!covering.length) { container.appendChild(document.createTextNode(chunk)); continue; }
    const mark = el("mark", "f-" + covering[0].field, chunk);
    mark.title = covering
      .map((s) => `${s.field} = ${s.extracted_value}`)
      .join("\n");
    container.appendChild(mark);
  }
  return container;
}

const NULLABLE = [
  "coded_term", "coded_term_version", "verbatim_term", "onset_date",
  "onset_offset_days", "anchor_event", "anchor_date", "severity",
  "relatedness", "action_taken", "rechallenge", "outcome",
];

function renderEvent(event) {
  const box = el("div", "evt");
  const head = el("div", "evt-head",
    `${event.concept_id} · assertion: ${event.assertion}`);
  box.appendChild(head);

  const table = el("table");
  const spanFields = new Set(event.evidence.map((s) => s.field));

  const row = (key, value, opts = {}) => {
    const tr = el("tr", value === null || value === undefined ||
      (Array.isArray(value) && !value.length) || value === false ? "null" : "");
    tr.appendChild(el("td", "k", key));
    const td = el("td", "v");
    td.textContent = Array.isArray(value)
      ? (value.length ? value.join(", ") : "—")
      : (value === null || value === undefined ? "—" : String(value));
    if (opts.needsSpan && !spanFields.has(opts.needsSpan)) {
      const warn = el("span", "nospan", "  ⚠ no span");
      td.appendChild(warn);
    }
    tr.appendChild(td);
    table.appendChild(tr);
  };

  row("coded term", event.coded_term, { needsSpan: event.coded_term ? "coded_term" : null });
  row("dictionary version", event.coded_term_version);
  row("verbatim term", event.verbatim_term);
  row("identified by", event.concept_match_kinds);
  row("assertion", event.assertion, { needsSpan: "assertion" });
  row("symptoms", event.symptoms.map((s) => s.symptom));
  row("labs", event.labs.map((l) =>
    `${l.test} ${l.value} ${l.unit}` +
    (l.canonical_value !== null ? ` = ${l.canonical_value} ${l.canonical_unit}` : "")));
  row("onset date", event.onset_date);
  row("onset offset (days)", event.onset_offset_days,
    { needsSpan: event.onset_offset_days !== null ? "onset_offset_days" : null });
  row("anchor", event.anchor_event);
  row("severity (intensity)", event.severity,
    { needsSpan: event.severity ? "severity" : null });
  row("seriousness (regulatory)", event.seriousness);
  row("relatedness", event.relatedness);
  row("action taken", event.action_taken);
  row("rechallenge", event.rechallenge);
  row("rescue treatment", event.rescue_treatment);
  row("outcome", event.outcome);
  row("spans", `${event.evidence.length}`);
  row("extractor", event.extractor_version);
  box.appendChild(table);
  return box;
}

async function loadDocList() {
  const study = $("doc-study").value;
  const { body } = await api(`/api/documents?limit=400${study ? "&study=" + study : ""}`);
  const list = $("doc-list");
  list.innerHTML = "";
  (body.documents || []).forEach((doc) => {
    const item = el("div", "doc-item", doc.doc_id);
    item.addEventListener("click", () => {
      document.querySelectorAll(".doc-item").forEach((d) => d.classList.remove("sel"));
      item.classList.add("sel");
      loadDoc(doc.doc_id);
    });
    list.appendChild(item);
  });
  if (body.documents && body.documents.length) list.firstChild.click();
}

async function loadDoc(docId) {
  const { body } = await api(`/api/documents/${encodeURIComponent(docId)}`);
  const spans = [];
  (body.events || []).forEach((e) =>
    e.evidence.forEach((s) => { if (s.doc_id === docId) spans.push(s); }));
  const target = $("doc-text");
  target.innerHTML = "";
  target.appendChild(renderWithSpans(body.text, spans));

  $("event-count").textContent = `${(body.events || []).length} object(s)`;
  const view = $("event-view");
  view.innerHTML = "";
  if (!body.events || !body.events.length) {
    view.appendChild(el("p", "hint",
      "No event object. Nothing in this record raised a catalogued concept — " +
      "an ungated abbreviation, for instance, is deliberately not an event."));
    return;
  }
  body.events.forEach((e) => view.appendChild(renderEvent(e)));
}

/* -------------------------------------------------------- 2. definition */

function findLabValue(definition) {
  let found = null;
  const walk = (node) => {
    if (found !== null || !node) return;
    if (Array.isArray(node)) { node.forEach(walk); return; }
    if (typeof node !== "object") return;
    for (const [key, value] of Object.entries(node)) {
      if (key === "lab" && value && typeof value === "object") { found = value; return; }
      walk(value);
    }
  };
  definition.evidence_rules.forEach((rule) => walk(rule.when));
  return found;
}

function fillDefinitionControls(definition) {
  $("def-badge").textContent = `v${definition.version}`;
  const status = $("def-status");
  status.textContent = definition.status;
  status.className = "badge status" + (definition.status === "draft" ? " draft" : "");
  $("def-label").textContent = definition.label + " — " + (definition.description || "");
  $("c-concept").value = definition.concept.primary;
  $("c-require").value = definition.assertion.require.join(", ");
  $("c-review").value = definition.assertion.route_to_review.join(", ") || "—";
  $("c-exclude").value = definition.assertion.exclude.join(", ") || "—";
  const lab = findLabValue(definition);
  $("c-glucose").value = lab ? lab.value : "";
  $("c-glucose").disabled = !lab;
  $("c-wmin").value = definition.window ? definition.window.min : "";
  $("c-wmax").value = definition.window ? definition.window.max : "";
  $("c-unresolved").value = definition.window ? definition.window.on_unresolved_onset : "review";
}

function collectChanges() {
  const base = state.current;
  const changes = {};
  const lab = findLabValue(base);
  const glucose = parseFloat($("c-glucose").value);
  if (lab && !Number.isNaN(glucose) && glucose !== lab.value) {
    changes[`evidence_rules.${labRuleId(base)}.lab.value`] = glucose;
  }
  if (base.window) {
    const wmin = parseInt($("c-wmin").value, 10);
    const wmax = parseInt($("c-wmax").value, 10);
    if (!Number.isNaN(wmin) && wmin !== base.window.min) changes["window.min"] = wmin;
    if (!Number.isNaN(wmax) && wmax !== base.window.max) changes["window.max"] = wmax;
    const unresolved = $("c-unresolved").value;
    if (unresolved !== base.window.on_unresolved_onset) {
      changes["window.on_unresolved_onset"] = unresolved;
    }
  }
  return changes;
}

function labRuleId(definition) {
  for (const rule of definition.evidence_rules) {
    let hit = false;
    const walk = (node) => {
      if (hit || !node) return;
      if (Array.isArray(node)) { node.forEach(walk); return; }
      if (typeof node !== "object") return;
      for (const [key, value] of Object.entries(node)) {
        if (key === "lab") { hit = true; return; }
        walk(value);
      }
    };
    walk(rule.when);
    if (hit) return rule.id;
  }
  return definition.evidence_rules[0].id;
}

function renderDiff(changes) {
  const box = $("def-diff");
  box.innerHTML = "";
  const keys = Object.keys(changes);
  if (!keys.length) { box.textContent = "No changes from the selected version."; return; }
  keys.sort().forEach((key) => {
    box.appendChild(el("div", "chg", `${key}: → ${JSON.stringify(changes[key])}`));
  });
}

async function loadDefinitions() {
  const { body } = await api("/definitions");
  state.definitions = body.definitions;
  const select = $("def-select");
  select.innerHTML = "";
  body.definitions.forEach((d) => {
    const option = el("option", null, `${d.key} (${d.status})`);
    option.value = d.key;
    select.appendChild(option);
  });
  const frozen = body.definitions.find((d) => d.status === "frozen") || body.definitions[0];
  state.frozen = frozen;
  select.value = frozen.key;
  selectDefinition(frozen.key);
}

function selectDefinition(key) {
  const entry = state.definitions.find((d) => d.key === key);
  if (!entry) return;
  state.current = entry.body;
  state.currentEntry = entry;
  fillDefinitionControls(entry.body);
  renderDiff({});
  $("candidate-yaml").textContent = "";
  $("btn-download").disabled = true;
  state.candidate = null;
}

$("def-select").addEventListener("change", (e) => selectDefinition(e.target.value));
["c-glucose", "c-wmin", "c-wmax", "c-unresolved"].forEach((id) =>
  $(id).addEventListener("input", () => renderDiff(collectChanges())));
$("btn-reset").addEventListener("click", () => selectDefinition($("def-select").value));

$("btn-candidate").addEventListener("click", async () => {
  const changes = collectChanges();
  if (!Object.keys(changes).length) {
    $("candidate-note").textContent =
      "Nothing has been changed, so there is no candidate to build.";
    return;
  }
  const { ok, body } = await api("/definitions/candidate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      definition_id: state.current.id,
      base_version: state.current.version,
      changes,
    }),
  });
  if (!ok) { $("candidate-note").textContent = (body && body.detail) || "candidate rejected"; return; }
  state.candidate = body;
  $("candidate-note").textContent = body.note;
  $("candidate-yaml").textContent = body.yaml;
  $("btn-download").disabled = false;
});

$("btn-download").addEventListener("click", () => {
  if (!state.candidate) return;
  const blob = new Blob([state.candidate.yaml], { type: "text/yaml" });
  const link = el("a");
  link.href = URL.createObjectURL(blob);
  link.download = state.candidate.filename;
  link.click();
  URL.revokeObjectURL(link.href);
});

$("btn-evaluate").addEventListener("click", async () => {
  const target = $("def-eval");
  target.textContent = "evaluating…";
  const { ok, body } = await api("/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      definition_id: state.current.id,
      version: state.current.version,
      allow_draft: state.current.status === "draft",
    }),
  });
  target.innerHTML = "";
  if (!ok) { target.appendChild(el("p", "err", (body && body.detail) || "evaluation failed")); return; }
  target.appendChild(el("p", "hint",
    `run ${body.run_id} · results ${body.results_hash} · definition hash ${body.definition_hash}`));
  const counts = el("div", "summary");
  Object.entries(body.counts_by_verdict).forEach(([k, v]) => {
    const stat = el("div", "stat");
    stat.appendChild(el("div", "n", String(v)));
    stat.appendChild(el("div", "l", k));
    counts.appendChild(stat);
  });
  target.appendChild(counts);

  const table = el("table", "grid");
  const head = el("tr");
  ["subject", "verdict", "state", "rule", "reason"].forEach((h) =>
    head.appendChild(el("th", null, h)));
  table.appendChild(head);
  body.assignments.filter((a) => a.verdict !== "excluded").slice(0, 25).forEach((a) => {
    const tr = el("tr");
    tr.appendChild(el("td", null, a.subject_id));
    tr.appendChild(el("td", null, a.verdict));
    tr.appendChild(el("td", null, a.evidence_state));
    tr.appendChild(el("td", null, a.matched_rule_id || "—"));
    tr.appendChild(el("td", "reason", a.reason));
    table.appendChild(tr);
  });
  target.appendChild(table);
});

/* --------------------------------------------------------- 3. retrieval */

function selected(id) {
  return [...$(id).selectedOptions].map((o) => o.value);
}

$("btn-clear-assertion").addEventListener("click", () => {
  [...$("r-assertion").options].forEach((o) => (o.selected = false));
  runRetrieval();
});
$("btn-retrieve").addEventListener("click", runRetrieval);

async function runRetrieval() {
  const params = new URLSearchParams();
  params.set("concept", $("r-concept").value);
  selected("r-assertion").forEach((v) => params.append("assertion", v));
  selected("r-state").forEach((v) => params.append("evidence_state", v));
  const wmin = $("r-wmin").value, wmax = $("r-wmax").value;
  if (wmin !== "" && wmax !== "") { params.set("window_min", wmin); params.set("window_max", wmax); }
  params.set("mode", $("r-mode").value);
  params.set("top_k", $("r-topk").value || "20");

  const { ok, body } = await api("/retrieve?" + params.toString());
  const summary = $("r-summary");
  const results = $("r-results");
  summary.innerHTML = ""; results.innerHTML = "";
  if (!ok) { results.appendChild(el("p", "err", (body && body.detail) || "retrieval failed")); return; }

  const filterOn = (body.filters.assertion || []).length > 0;
  const stats = [
    ["records", body.count, ""],
    ["assertion filter", filterOn ? (body.filters.assertion || []).join(", ") : "OFF", ""],
    ["records asserting absence", body.negation_false_positives,
      body.negation_false_positives ? "alert" : "ok"],
    ["negation FP rate", body.negation_false_positive_rate.toFixed(4),
      body.negation_false_positives ? "alert" : "ok"],
    ["concept surface forms", body.expanded_terms.length, ""],
  ];
  stats.forEach(([label, value, tone]) => {
    const stat = el("div", "stat" + (tone ? " " + tone : ""));
    stat.appendChild(el("div", "n", String(value)));
    stat.appendChild(el("div", "l", label));
    summary.appendChild(stat);
  });
  (body.notes || []).forEach((n) => summary.appendChild(el("div", "l", n)));

  body.records.forEach((r) => {
    const card = el("div", "rec");
    const meta = el("div", "meta");
    meta.appendChild(el("span", null, r.subject_id));
    meta.appendChild(el("span", "pill " + r.assertion, r.assertion));
    if (r.evidence_state) meta.appendChild(el("span", "pill", r.evidence_state));
    meta.appendChild(el("span", null, `offset ${r.onset_offset_days ?? "—"}d`));
    meta.appendChild(el("span", null, r.study_id));
    if (r.coded_term) meta.appendChild(el("span", null, `coded: ${r.coded_term}`));
    card.appendChild(meta);
    card.appendChild(el("div", "snip", r.snippet));
    results.appendChild(card);
  });
  if (!body.records.length) {
    results.appendChild(el("p", "hint", "No records matched those filters."));
  }
}

/* ------------------------------------------------------------- 4. agent */

document.querySelectorAll(".examples .link").forEach((button) =>
  button.addEventListener("click", () => {
    $("a-question").value = button.dataset.q;
    $("btn-compile").click();
  }));

$("btn-compile").addEventListener("click", async () => {
  const question = $("a-question").value.trim();
  state.question = question;
  $("a-result").innerHTML = "";
  $("a-approve").checked = false;
  $("btn-run").disabled = true;

  const { body } = await api("/agent/compile", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, backend: "deterministic" }),
  });

  const clarifyBox = $("a-clarification");
  const specWrap = $("a-spec-wrap");
  if (!body) { $("a-result").appendChild(el("p", "err", "compile failed")); return; }
  if (body.needs_clarification) {
    specWrap.classList.add("hidden");
    clarifyBox.classList.remove("hidden");
    clarifyBox.innerHTML = "";
    clarifyBox.appendChild(el("h3", null, "Clarification needed — nothing was executed"));
    clarifyBox.appendChild(el("p", null, "Ambiguity: " + body.clarification.ambiguity));
    clarifyBox.appendChild(el("p", null, "Effect: " + body.clarification.effect));
    if (body.clarification.options.length) {
      const list = el("ul");
      body.clarification.options.forEach((o) => list.appendChild(el("li", null, o)));
      clarifyBox.appendChild(list);
    }
    return;
  }
  clarifyBox.classList.add("hidden");
  specWrap.classList.remove("hidden");
  $("a-spec").textContent = JSON.stringify(body.spec, null, 2);
});

$("a-approve").addEventListener("change", (e) => {
  $("btn-run").disabled = !e.target.checked;
});

$("btn-run").addEventListener("click", async () => {
  const target = $("a-result");
  target.textContent = "executing…";
  const { ok, body } = await api("/agent/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: state.question, approved: true }),
  });
  target.innerHTML = "";
  if (!ok) { target.appendChild(el("p", "err", (body && body.detail) || "execution failed")); return; }

  target.appendChild(el("h2", null, "Evidence package"));
  const stats = el("div", "summary");
  [
    ["primary cases", body.summary.primary_case_count, "ok"],
    ["review set", body.summary.review_set_count, ""],
    ["subjects", body.summary.subjects, ""],
    ["negation FP rate", body.retrieval.negation_false_positive_rate.toFixed(4), "ok"],
  ].forEach(([label, value, tone]) => {
    const stat = el("div", "stat" + (tone ? " " + tone : ""));
    stat.appendChild(el("div", "n", String(value)));
    stat.appendChild(el("div", "l", label));
    stats.appendChild(stat);
  });
  target.appendChild(stats);

  target.appendChild(el("p", "hint",
    `definition ${body.definition.id}.v${body.definition.version} (${body.definition.status}) ` +
    `hash ${body.definition.hash} · extractor ${body.extractor_version} · ` +
    `snapshot ${body.snapshot_id} · run ${body.run_id}`));

  const byState = el("table", "grid");
  const head = el("tr");
  ["evidence state", "subjects"].forEach((h) => head.appendChild(el("th", null, h)));
  byState.appendChild(head);
  Object.entries(body.summary.counts_by_state).forEach(([k, v]) => {
    const tr = el("tr");
    tr.appendChild(el("td", null, k));
    tr.appendChild(el("td", null, String(v)));
    byState.appendChild(tr);
  });
  target.appendChild(byState);

  target.appendChild(el("h2", null, "Contributing spans"));
  const spans = el("table", "grid");
  const spanHead = el("tr");
  ["subject", "field", "rule", "text"].forEach((h) => spanHead.appendChild(el("th", null, h)));
  spans.appendChild(spanHead);
  body.contributing_spans.slice(0, 12).forEach((s) => {
    const tr = el("tr");
    tr.appendChild(el("td", null, s.subject_id));
    tr.appendChild(el("td", null, s.field));
    tr.appendChild(el("td", null, s.rule || "—"));
    tr.appendChild(el("td", "reason", s.text));
    spans.appendChild(tr);
  });
  target.appendChild(spans);

  target.appendChild(el("h2", null, "Limitations"));
  const limits = el("ul");
  body.limitations.forEach((l) => limits.appendChild(el("li", null, l)));
  target.appendChild(limits);
});

/* ---------------------------------------------------------------- boot */

(async function boot() {
  const { ok, body } = await api("/api/summary");
  if (!ok) {
    $("corpus-line").textContent =
      (body && body.detail) || "No corpus. Run `aelayer generate` then `aelayer extract`.";
    return;
  }
  $("corpus-line").textContent =
    `${body.studies.length} studies · ${body.subjects} subjects · ` +
    `${body.ae_records} AE records · extractor ${body.extractor_version} · ` +
    `snapshot ${body.snapshot_id}`;

  const studySelect = $("doc-study");
  body.studies.forEach((s) => {
    const option = el("option", null, s);
    option.value = s;
    studySelect.appendChild(option);
  });
  studySelect.addEventListener("change", loadDocList);

  const conceptSelect = $("r-concept");
  body.concepts.forEach((c) => {
    const option = el("option", null, c);
    option.value = c;
    conceptSelect.appendChild(option);
  });
  // Open on the concept the shipped definition is about, not whichever
  // concept happens to sort first.
  if (body.concepts.includes("HYPOGLYCEMIA")) conceptSelect.value = "HYPOGLYCEMIA";
  Object.keys(body.concept_groups || {}).forEach((g) => {
    const option = el("option", null, `${g} (group)`);
    option.value = g;
    conceptSelect.appendChild(option);
  });

  await loadDocList();
  await loadDefinitions();
  await runRetrieval();
})();
