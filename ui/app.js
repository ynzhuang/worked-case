/* Adverse event evidence layer — single page UI.
 *
 * Every panel reads the same API the CLI drives, so the two cannot disagree
 * about which definition version produced a number. Nothing is computed here:
 * the browser renders what the server says and never derives a count of its own.
 */
'use strict';

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};
const esc = (s) => String(s ?? '').replace(/[&<>"]/g,
  (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));

async function api(path, options) {
  const response = await fetch(path, options);
  let body = null;
  try { body = await response.json(); } catch (_) { body = null; }
  if (!response.ok) {
    const error = new Error((body && (body.detail || body.error)) || response.statusText);
    error.status = response.status;
    error.body = body;
    throw error;
  }
  return body;
}

const post = (path, payload) => api(path, {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify(payload),
});

function query(params) {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined || value === '' || value === false) continue;
    if (Array.isArray(value)) value.forEach((v) => search.append(key, v));
    else search.append(key, value);
  }
  return search.toString();
}

const selected = (node) => Array.from(node.selectedOptions).map((o) => o.value);

function stat(container, value, label, tone) {
  const box = el('div', 'stat' + (tone ? ' ' + tone : ''));
  box.appendChild(el('div', 'n', String(value)));
  box.appendChild(el('div', 'l', label));
  container.appendChild(box);
}

function table(container, headers, rows) {
  const node = el('table', 'grid');
  node.appendChild(el('tr'));
  headers.forEach((h) => {
    const th = el('th', null, h);
    node.firstChild.appendChild(th);
  });
  rows.forEach((row) => {
    const tr = el('tr');
    row.forEach((cell, index) => {
      const td = el('td', index === row.length - 1 ? 'reason' : null,
        cell === null || cell === undefined ? '—' : String(cell));
      tr.appendChild(td);
    });
    node.appendChild(tr);
  });
  container.innerHTML = '';
  container.appendChild(node);
}

function fail(node, error) {
  node.innerHTML = '';
  node.appendChild(el('p', 'err', error.message || String(error)));
}

document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
    tab.classList.add('active');
    $(tab.dataset.panel).classList.add('active');
  });
});

/* ------------------------------------------------------------- summary */

let SUMMARY = null;

async function loadSummary() {
  SUMMARY = await api('/api/summary');
  $('corpus-line').textContent =
    `${Object.keys(SUMMARY.profiles).length} profiles · ${SUMMARY.subjects} subjects · ` +
    `${SUMMARY.records} records · ${SUMMARY.episodes} episodes · ` +
    `normalizer ${SUMMARY.normalizer_version} · extractor ${SUMMARY.extractor_version} ` +
    `(${SUMMARY.extraction_backend}) · snapshot ${SUMMARY.snapshot_id}`;

  const box = $('route-summary');
  box.innerHTML = '';
  for (const [route, count] of Object.entries(SUMMARY.location_routes)) {
    const tone = ['direct', 'normalized', 'extracted'].includes(route) ? 'ok' : null;
    stat(box, count, route.replace(/_/g, ' '), tone);
  }

  table($('profile-table'),
    ['profile', 'term style', 'location home', 'pattern home', 'dictionary',
     'both routes?'],
    Object.entries(SUMMARY.profiles).map(([id, body]) => [
      id, body.reported_term_style, body.location_home.join(', '),
      body.pattern_home.join(', '), body.dictionary_version,
      body.carries_both_location ? 'yes — silver standard' : 'no',
    ]));

  const select = $('rec-profile');
  Object.keys(SUMMARY.profiles).forEach((id) => {
    const option = el('option', null, id);
    option.value = id;
    select.appendChild(option);
  });
}

/* ------------------------------------------------------- 1. records */

async function loadRecords() {
  const list = $('rec-list');
  list.innerHTML = 'loading…';
  const body = await api('/api/records?' + query({
    profile: $('rec-profile').value,
    method: $('rec-method').value,
    limit: 200,
  }));
  list.innerHTML = '';
  if (!body.records.length) {
    list.appendChild(el('p', 'hint', 'No records match.'));
    return;
  }
  body.records.forEach((record) => {
    const item = el('div', 'doc-item');
    item.textContent =
      `${record.source_record_id}\n${record.profile} · ` +
      `${record.location || '—'} (${record.location_method || record.location_availability})`;
    item.addEventListener('click', () => {
      list.querySelectorAll('.doc-item').forEach((n) => n.classList.remove('sel'));
      item.classList.add('sel');
      showRecord(record.source_record_id);
    });
    list.appendChild(item);
  });
  list.firstChild.classList.add('sel');
  showRecord(body.records[0].source_record_id);
}

function highlight(text, spans) {
  const marks = spans
    .filter((s) => typeof s.start === 'number' && s.end > s.start)
    .sort((a, b) => a.start - b.start);
  let out = '';
  let cursor = 0;
  marks.forEach((span) => {
    if (span.start < cursor) return;
    out += esc(text.slice(cursor, span.start));
    out += `<mark title="${esc(span.field)}: ${esc(span.extracted_value)}">` +
           `${esc(text.slice(span.start, span.end))}</mark>`;
    cursor = span.end;
  });
  return out + esc(text.slice(cursor));
}

async function showRecord(recordId) {
  const view = $('rec-view');
  const source = $('rec-source');
  view.innerHTML = 'loading…';
  let body;
  try { body = await api(`/api/records/${encodeURIComponent(recordId)}`); }
  catch (error) { fail(view, error); return; }

  $('rec-badge').textContent = body.provenance_complete
    ? 'every populated attribute has a span' : 'PROVENANCE DEFECT';

  source.innerHTML = '';
  const term = body.record.reported_term.value || '';
  const termSpans = Object.values(body.attributes)
    .flatMap((a) => a.evidence)
    .filter((s) => s.kind === 'text' && s.doc_id.endsWith(':AETERM'));
  const termBlock = el('div');
  termBlock.appendChild(el('div', 'hint', 'AETERM — the investigator\'s own words'));
  const termText = el('div');
  termText.innerHTML = highlight(term, termSpans);
  termBlock.appendChild(termText);
  source.appendChild(termBlock);

  (body.supplemental || []).forEach((row) => {
    const block = el('div');
    block.appendChild(el('div', 'hint',
      `SUPPAE.${row.QNAM} — a sponsor-defined qualifier`));
    block.appendChild(el('div', null, `${row.QVAL}  (${row.QLABEL})`));
    source.appendChild(block);
  });

  (body.documents || []).forEach((document_) => {
    const spans = Object.values(body.attributes)
      .flatMap((a) => a.evidence)
      .filter((s) => s.doc_id === document_.doc_id);
    const block = el('div');
    block.appendChild(el('div', 'hint', `${document_.doc_id} — a linked comment`));
    const text = el('div');
    text.innerHTML = highlight(document_.text, spans);
    block.appendChild(text);
    source.appendChild(block);
  });

  view.innerHTML = '';
  const card = el('div', 'evt');
  card.appendChild(el('div', 'evt-head',
    `${body.record.source_record_id} · ${body.record.profile}`));
  const grid = el('table');
  Object.entries(body.attributes).forEach(([name, attribute]) => {
    const row = el('tr');
    if (attribute.value === null) row.className = 'null';
    row.appendChild(el('td', 'k', name));
    const value = el('td', 'v');
    value.appendChild(document.createTextNode(
      (attribute.value === null ? '—' : String(attribute.value)) + '  '));
    value.appendChild(el('span',
      'pill ' + (attribute.availability === 'collected' ? 'present' : 'uncertain'),
      attribute.availability));
    if (attribute.method) {
      value.appendChild(document.createTextNode(' '));
      value.appendChild(el('span', 'pill', attribute.method));
    }
    if (attribute.source_variable) {
      value.appendChild(document.createTextNode(' '));
      value.appendChild(el('span', 'pill', attribute.source_variable));
    }
    if (attribute.prior_availability
        && attribute.prior_availability !== attribute.availability) {
      value.appendChild(document.createTextNode(' '));
      value.appendChild(el('span', 'pill', `CRF: ${attribute.prior_availability}`));
    }
    if (attribute.note) value.appendChild(el('div', 'hint', attribute.note));
    if (attribute.value !== null && !attribute.evidence.length) {
      value.appendChild(el('div', 'nospan', 'no span — defect'));
    }
    row.appendChild(value);
    grid.appendChild(row);
  });
  card.appendChild(grid);
  view.appendChild(card);
}

/* -------------------------------------------------------- 2. silver */

async function loadSilver() {
  const box = $('silver-summary');
  box.innerHTML = 'loading…';
  let body;
  try { body = await api('/eval/silver?limit=40'); }
  catch (error) { fail(box, error); return; }
  const overall = body.overall;
  box.innerHTML = '';
  stat(box, overall.precision.toFixed(3), 'precision');
  stat(box, overall.recall.toFixed(3), 'recall');
  stat(box, overall.coverage.toFixed(3), 'coverage');
  stat(box, overall.abstention_rate.toFixed(3), 'abstention rate');
  stat(box, overall.normalized_agreement.toFixed(3), 'normalized agreement');
  stat(box, overall.disagreements, 'disagreements',
    overall.disagreements ? 'alert' : 'ok');

  table($('silver-style'),
    ['style', 'precision', 'recall', 'coverage', 'abstention', 'answered'],
    Object.entries(body.by_reported_term_style).map(([style, m]) => [
      style, m.precision.toFixed(3), m.recall.toFixed(3), m.coverage.toFixed(3),
      m.abstention_rate.toFixed(3), m.answered,
    ]));

  const queue = $('silver-queue');
  queue.innerHTML = '';
  $('queue-badge').textContent = `${body.adjudication_queue.length} shown`;
  body.adjudication_queue.forEach((row) => {
    const card = el('div', 'rec');
    const meta = el('div', 'meta');
    meta.appendChild(el('span', null, row.source_record_id));
    meta.appendChild(el('span', 'pill ' +
      (row.agreement === 'agree' ? 'present' : 'absent'), row.agreement));
    meta.appendChild(el('span', null, `structured: ${row.structured_value ?? '—'}`));
    meta.appendChild(el('span', null, `extracted: ${row.extracted_value ?? '—'}`));
    card.appendChild(meta);
    card.appendChild(el('div', 'snip', row.text));
    card.appendChild(el('div', 'hint', row.queue_reason));
    queue.appendChild(card);
  });
}

/* ----------------------------------------------------- 3. phenotype */

let DEFINITIONS = [];

async function loadDefinitions() {
  const body = await api('/definitions');
  DEFINITIONS = body.definitions;
  const select = $('def-select');
  select.innerHTML = '';
  DEFINITIONS.forEach((definition) => {
    const option = el('option', null,
      `${definition.key} (${definition.status}) — accepts ${definition.accept_methods.join(', ')}`);
    option.value = definition.key;
    select.appendChild(option);
  });
  select.value = DEFINITIONS[0].key;
}

const currentDefinition = () =>
  DEFINITIONS.find((d) => d.key === $('def-select').value);

async function evaluateDefinition() {
  const definition = currentDefinition();
  const summary = $('verdict-summary');
  summary.innerHTML = 'evaluating…';
  let body;
  try {
    body = await post('/evaluate', {
      definition_id: definition.id, version: definition.version,
      allow_draft: definition.status === 'draft',
    });
  } catch (error) { fail(summary, error); return; }

  const manifest = body.manifest;
  summary.innerHTML = '';
  Object.entries(manifest.counts_by_verdict).forEach(([verdict, count]) => {
    stat(summary, count, verdict,
      verdict === 'case' ? 'ok' : (verdict === 'not_ascertainable' ? 'alert' : null));
  });
  Object.entries(manifest.attribute_methods).forEach(([method, count]) => {
    stat(summary, count, `via ${method}`);
  });

  table($('verdict-table'),
    ['episode', 'verdict', 'location', 'route', 'why'],
    body.assignments
      .filter((a) => a.verdict !== 'not_case')
      .slice(0, 30)
      .map((a) => [
        a.episode_id, a.verdict,
        (a.findings.find((f) => f.name === 'location') || {}).value ?? '—',
        Object.entries(a.attribute_sources).map(([k, v]) => `${k}=${v}`).join(' ') || '—',
        a.reason,
      ]));
}

async function loadAblation() {
  const view = $('ablation-view');
  view.innerHTML = 'running…';
  let body;
  try { body = await api('/eval/ablation'); }
  catch (error) { fail(view, error); return; }
  view.innerHTML = '';
  const box = el('div', 'summary');
  stat(box, body.cases_structured_only, 'cases, structured only');
  stat(box, body.cases_with_text, 'cases, with text', 'ok');
  stat(box, body.cases_only_findable_through_text, 'only findable through text', 'ok');
  stat(box, `${(body.fraction_only_findable_through_text * 100).toFixed(1)}%`,
    'of all cases');
  stat(box, body.not_ascertainable_resolved_by_text, 'unascertainable resolved by text');
  view.appendChild(box);
  view.appendChild(el('p', 'hint', body.note));
}

async function compareDefinitions() {
  const definition = currentDefinition();
  const view = $('compare-view');
  view.innerHTML = 'comparing…';
  let body;
  try {
    body = await api(`/definitions/${definition.id}/compare?` + query({
      left: $('cmp-left').value, right: $('cmp-right').value,
      scope: $('cmp-scope').value,
    }));
  } catch (error) { fail(view, error); return; }
  view.innerHTML = '';
  view.appendChild(el('p', 'hint', body.summary));
  table(view, ['episode', 'left', 'right', 'why it moved'],
    body.discordant.slice(0, 15).map((entry) => [
      entry.episode_id, entry.verdict_a, entry.verdict_b, entry.reason_b,
    ]));
}

/* ----------------------------------------------------- 4. retrieval */

async function search() {
  const summary = $('r-summary');
  const results = $('r-results');
  summary.innerHTML = '';
  results.innerHTML = 'searching…';
  const precise = $('r-path').value === 'precise';
  let body;
  try {
    body = precise
      ? await api('/retrieve?' + query({
          concept: 'RASH', region: $('r-region').value,
          method: selected($('r-method')), verdict: selected($('r-verdict')),
          top_k: $('r-topk').value,
        }))
      : await api('/discover?' + query({
          unnormalized_only: $('r-unnormalized').checked,
          mode: 'hybrid', top_k: $('r-topk').value,
        }));
  } catch (error) { fail(results, error); return; }

  stat(summary, body.count, precise ? 'episodes' : 'mentions');
  stat(summary, body.usable_as_cohort ? 'yes' : 'no', 'usable as a cohort',
    body.usable_as_cohort ? 'ok' : 'alert');
  if (!precise) {
    stat(summary, body.unnormalized_count, 'not covered by any catalogue value');
  }

  results.innerHTML = '';
  if (body.cohort_note) results.appendChild(el('p', 'hint', body.cohort_note));
  (body.notes || []).forEach((note) => results.appendChild(el('p', 'hint', note)));

  (precise ? body.episodes : body.mentions).forEach((row) => {
    const card = el('div', 'rec');
    const meta = el('div', 'meta');
    if (precise) {
      meta.appendChild(el('span', null, row.episode_id));
      meta.appendChild(el('span', 'pill', String(row.verdict)));
      meta.appendChild(el('span', null, `${row.location} via ${row.location_method}`));
      meta.appendChild(el('span', null, row.location_source || ''));
      meta.appendChild(el('span', null, `offset ${row.onset_offset_days}`));
      card.appendChild(meta);
      card.appendChild(el('div', 'snip', row.reported_terms));
    } else {
      meta.appendChild(el('span', null, row.subject_id));
      meta.appendChild(el('span', 'pill', row.attribute));
      meta.appendChild(el('span', 'pill ' + (row.normalized ? 'present' : 'absent'),
        row.normalized ? 'normalized' : 'not in any catalogue'));
      meta.appendChild(el('span', null, row.source_variable || ''));
      meta.appendChild(el('span', 'pill', 'candidate'));
      card.appendChild(meta);
      card.appendChild(el('div', 'snip', `${row.surface} — ${row.sentence}`));
    }
    results.appendChild(card);
  });
}

/* --------------------------------------------------------- 5. agent */

function renderClarification(clarification) {
  const box = $('a-clarification');
  box.classList.remove('hidden');
  box.innerHTML = '';
  box.appendChild(el('h3', null, 'Clarification needed before anything is run'));
  box.appendChild(el('p', null, clarification.ambiguity));
  box.appendChild(el('p', null, clarification.effect));
  const list = el('ul');
  clarification.options.forEach((o) => list.appendChild(el('li', null, o)));
  box.appendChild(list);
  box.appendChild(el('p', 'hint',
    'No specification was compiled and no number was produced.'));
}

async function compileOnly() {
  $('a-clarification').classList.add('hidden');
  $('a-result').innerHTML = '';
  let body;
  try { body = await post('/agent/compile', {question: $('a-question').value}); }
  catch (error) {
    if (error.body && error.body.clarification) {
      $('a-spec-wrap').classList.add('hidden');
      renderClarification(error.body.clarification);
      return;
    }
    fail($('a-result'), error);
    return;
  }
  $('a-spec-badge').textContent = 'not executed';
  $('a-spec-wrap').classList.remove('hidden');
  $('a-spec').textContent = JSON.stringify(body.spec, null, 2);
}

async function ask() {
  const result = $('a-result');
  $('a-clarification').classList.add('hidden');
  result.innerHTML = 'running…';
  let body;
  try { body = await post('/agent/ask', {question: $('a-question').value}); }
  catch (error) {
    result.innerHTML = '';
    if (error.body && error.body.clarification) {
      $('a-spec-wrap').classList.add('hidden');
      renderClarification(error.body.clarification);
      return;
    }
    fail(result, error);
    return;
  }

  $('a-spec-badge').textContent = 'executed';
  $('a-spec-wrap').classList.remove('hidden');
  $('a-spec').textContent = JSON.stringify(body.spec, null, 2);

  result.innerHTML = '';
  const box = el('div', 'summary');
  Object.entries(body.cohort.counts_by_verdict).forEach(([verdict, count]) => {
    stat(box, count, verdict,
      verdict === 'case' ? 'ok' : (verdict === 'not_ascertainable' ? 'alert' : null));
  });
  Object.entries(body.cohort.attribute_methods).forEach(([method, count]) => {
    stat(box, count, `via ${method}`);
  });
  result.appendChild(box);
  result.appendChild(el('p', 'hint', body.cohort.not_ascertainable_note));
  result.appendChild(el('p', 'hint',
    `definition ${body.definition.id}.v${body.definition.version} ` +
    `(${body.definition.status}) hash ${body.definition.hash} · ` +
    `manifest ${body.manifest_id} · results ${body.results_hash} · ` +
    `tools: ${body.tools_called.join(', ')}`));
  const limits = el('ul');
  body.limitations.forEach((l) => limits.appendChild(el('li', 'hint', l)));
  result.appendChild(el('h2', null, 'Limitations'));
  result.appendChild(limits);

  try {
    const trace = await api(`/trace/${body.manifest_id}`);
    $('a-trace').textContent = trace.rendered;
  } catch (error) {
    $('a-trace').textContent = `trace unavailable: ${error.message}`;
  }
}

async function loadTools() {
  const body = await api('/agent/tools');
  table($('tool-table'), ['tool', 'permission', 'writes source records', 'what it does'],
    body.tools.map((t) => [
      t.name, t.permission, t.writes_source_records ? 'yes' : 'no', t.description,
    ]));
}

/* ------------------------------------------------------------- wiring */

$('rec-profile').addEventListener('change', loadRecords);
$('rec-method').addEventListener('change', loadRecords);
$('btn-evaluate').addEventListener('click', evaluateDefinition);
$('btn-ablation').addEventListener('click', loadAblation);
$('btn-compare').addEventListener('click', compareDefinitions);
$('btn-search').addEventListener('click', search);
$('btn-compile').addEventListener('click', compileOnly);
$('btn-ask').addEventListener('click', ask);
document.querySelectorAll('.examples .link').forEach((button) => {
  button.addEventListener('click', () => {
    $('a-question').value = button.dataset.q;
    compileOnly();
  });
});

(async function start() {
  try {
    await loadSummary();
    await loadRecords();
    await loadDefinitions();
    await loadSilver();
    await loadTools();
    await search();
  } catch (error) {
    $('corpus-line').textContent =
      'Could not load: ' + (error.message || error) +
      ' — run `make demo` first, then reload.';
  }
})();
