/* Adverse event evidence layer — single page UI.
 *
 * Every panel reads the same API the CLI drives, so the two cannot disagree
 * about which definition version produced a number. Nothing is computed here:
 * the browser renders what the server says and never derives a count of its
 * own.
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
    const detail = (body && (body.detail || body.error)) || response.statusText;
    const error = new Error(detail);
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
    if (value === null || value === undefined || value === '') continue;
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

function fail(node, error) {
  node.innerHTML = '';
  node.appendChild(el('p', 'err', error.message || String(error)));
}

/* ---------------------------------------------------------------- tabs */

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
    `${SUMMARY.studies.length} studies · ${SUMMARY.subjects} subjects · ` +
    `${SUMMARY.records} canonical records · ${SUMMARY.episodes} episodes · ` +
    `normalizer ${SUMMARY.normalizer_version} · extractor ` +
    `${SUMMARY.extractor_version} (${SUMMARY.extraction_backend}) · ` +
    `snapshot ${SUMMARY.snapshot_id}`;

  for (const id of ['rec-study', 'ep-study']) {
    const node = $(id);
    SUMMARY.studies.forEach((study) => {
      const semantics = SUMMARY.study_semantics[study] || {};
      const option = el('option', null,
        `${study} · ${semantics.representation || '?'} · ${semantics.dictionary_version || ''}`);
      option.value = study;
      node.appendChild(option);
    });
  }
  const concepts = SUMMARY.concepts;
  concepts.forEach((concept) => {
    const a = el('option', null, concept); a.value = concept; $('ep-concept').appendChild(a);
    const b = el('option', null, concept); b.value = concept; $('r-concept').appendChild(b);
  });
  Object.keys(SUMMARY.concept_groups || {}).forEach((group) => {
    const option = el('option', null, `group: ${group}`);
    option.value = group;
    $('r-concept').appendChild(option);
  });
  $('r-concept').value = concepts.includes('HYPOGLYCEMIA') ? 'HYPOGLYCEMIA' : concepts[0];

  const box = $('state-summary');
  box.innerHTML = '';
  for (const [state, count] of Object.entries(SUMMARY.collection_states)) {
    stat(box, count, state.replace(/_/g, ' '), state === 'collected' ? 'ok' : null);
  }
}

/* ------------------------------------------------------- 1. records */

let RECORDS = [];

async function loadRecords() {
  const list = $('rec-list');
  list.innerHTML = 'loading…';
  const body = await api('/api/records?' + query({
    study: $('rec-study').value,
    collection_state: $('rec-state').value,
    limit: 200,
  }));
  RECORDS = body.records;
  list.innerHTML = '';
  if (!RECORDS.length) { list.appendChild(el('p', 'hint', 'No records match.')); return; }
  RECORDS.forEach((record) => {
    const item = el('div', 'doc-item',
      `${record.source_record_id}\n${record.study_id} · ${record.verbatim_term || '(no term)'}`);
    item.addEventListener('click', () => {
      list.querySelectorAll('.doc-item').forEach((n) => n.classList.remove('sel'));
      item.classList.add('sel');
      showRecord(record.source_record_id);
    });
    list.appendChild(item);
  });
  showRecord(RECORDS[0].source_record_id);
  list.firstChild.classList.add('sel');
}

function highlight(text, spans) {
  const marks = spans
    .filter((s) => s.doc_id && typeof s.start === 'number' && s.end > s.start)
    .sort((a, b) => a.start - b.start || b.end - a.end);
  let out = '';
  let cursor = 0;
  marks.forEach((span) => {
    if (span.start < cursor) return;
    out += esc(text.slice(cursor, span.start));
    out += `<mark class="f-${esc(span.field)}" title="${esc(span.field)}: ` +
           `${esc(span.extracted_value)}">${esc(text.slice(span.start, span.end))}</mark>`;
    cursor = span.end;
  });
  out += esc(text.slice(cursor));
  return out;
}

async function showRecord(recordId) {
  const view = $('rec-view');
  const textbox = $('rec-text');
  view.innerHTML = 'loading…';
  let body;
  try { body = await api(`/api/records/${encodeURIComponent(recordId)}`); }
  catch (error) { fail(view, error); return; }

  $('rec-badge').textContent = body.provenance_complete
    ? 'every populated field has a span' : 'PROVENANCE DEFECT';

  if (body.narrative) {
    const spans = [];
    Object.values(body.fields).forEach((field) => {
      field.spans.forEach((span) => {
        if (span.doc_id === body.narrative.doc_id) spans.push(span);
      });
    });
    textbox.innerHTML = highlight(body.narrative.text, spans);
  } else {
    textbox.textContent =
      'This record has no narrative. Everything on it came from structured fields.';
  }

  view.innerHTML = '';
  const card = el('div', 'evt');
  card.appendChild(el('div', 'evt-head',
    `${body.record.source_record_id} · ${body.record.study_id} · ` +
    `${body.record.subject_id}`));
  const table = el('table');
  Object.entries(body.fields).forEach(([name, field]) => {
    const row = el('tr');
    if (field.value === null || field.value === undefined) row.className = 'null';
    row.appendChild(el('td', 'k', name));
    const value = el('td', 'v');
    const text = field.value === null || field.value === undefined
      ? '—' : String(field.value);
    value.appendChild(document.createTextNode(text + '  '));
    const state = el('span', 'pill ' + (field.collection_state === 'collected'
      ? 'present' : 'uncertain'), field.collection_state);
    value.appendChild(state);
    if (field.source === 'text') {
      value.appendChild(document.createTextNode(' '));
      value.appendChild(el('span', 'pill', 'from narrative'));
    }
    if (field.structured_state !== field.collection_state) {
      value.appendChild(document.createTextNode(' '));
      value.appendChild(el('span', 'pill', `CRF: ${field.structured_state}`));
    }
    if (field.note) value.appendChild(el('div', 'hint', field.note));
    if (field.value !== null && field.value !== undefined && !field.spans.length) {
      value.appendChild(el('div', 'nospan', 'no span — defect'));
    }
    row.appendChild(value);
    table.appendChild(row);
  });
  card.appendChild(table);
  view.appendChild(card);
}

/* ------------------------------------------------------ 2. episodes */

async function loadEpisodes() {
  const list = $('ep-list');
  list.innerHTML = 'loading…';
  const body = await api('/api/episodes?' + query({
    study: $('ep-study').value,
    concept: $('ep-concept').value,
    review_only: $('ep-review').checked ? 'true' : '',
    limit: 200,
  }));
  $('ep-count').textContent = `${body.count} total`;
  list.innerHTML = '';
  if (!body.episodes.length) { list.appendChild(el('p', 'hint', 'No episodes match.')); return; }
  body.episodes.forEach((episode) => {
    const item = el('div', 'doc-item');
    item.textContent =
      `${episode.episode_id}\n${episode.linkage_rule} · ` +
      `${episode.source_record_ids.length} record(s)` +
      (episode.linkage_review_required ? ' · FLAGGED' : '');
    item.addEventListener('click', () => {
      list.querySelectorAll('.doc-item').forEach((n) => n.classList.remove('sel'));
      item.classList.add('sel');
      showEpisode(episode.episode_id);
    });
    list.appendChild(item);
  });
  list.firstChild.classList.add('sel');
  showEpisode(body.episodes[0].episode_id);
}

async function showEpisode(episodeId) {
  const view = $('ep-view');
  view.innerHTML = 'loading…';
  let body;
  try { body = await api(`/api/episodes/${encodeURIComponent(episodeId)}`); }
  catch (error) { fail(view, error); return; }
  const episode = body.episode;
  view.innerHTML = '';

  const head = el('div', 'evt');
  head.appendChild(el('div', 'evt-head', episode.episode_id));
  const table = el('table');
  const rows = [
    ['concept', episode.standardized_concept],
    ['linkage rule', episode.linkage_rule],
    ['linkage confidence', episode.linkage_confidence],
    ['flagged for review', episode.linkage_review_required ? 'yes' : 'no'],
    ['linkage note', episode.linkage_note || '—'],
    ['records', episode.source_record_ids.join(', ')],
    ['discovery candidate', episode.candidate ? 'yes' : 'no'],
  ];
  rows.forEach(([key, value]) => {
    const row = el('tr');
    row.appendChild(el('td', 'k', key));
    row.appendChild(el('td', 'v', String(value ?? '—')));
    table.appendChild(row);
  });
  head.appendChild(table);
  view.appendChild(head);

  view.appendChild(el('h2', null, 'Source records, unmodified'));
  body.records.forEach((record) => {
    const card = el('div', 'rec');
    card.appendChild(el('div', 'meta',
      `${record.source_record_id} · ${record.source_form_id} · ` +
      `${record.coded_term?.value || record.verbatim_term?.value || '(no term)'}`));
    const states = Object.entries(body.field_states)
      .map(([field, state]) => `${field}=${state}`).join('  ');
    card.appendChild(el('div', 'snip', states));
    view.appendChild(card);
  });
}

/* ---------------------------------------------------- 3. definitions */

let DEFINITIONS = [];
let CANDIDATE = null;

async function loadDefinitions() {
  const body = await api('/definitions');
  DEFINITIONS = body.definitions;
  const select = $('def-select');
  select.innerHTML = '';
  DEFINITIONS.forEach((definition) => {
    const option = el('option', null, `${definition.key} (${definition.status})`);
    option.value = definition.key;
    select.appendChild(option);
  });
  select.value = DEFINITIONS[0].key;
  showDefinition();
}

const currentDefinition = () =>
  DEFINITIONS.find((d) => d.key === $('def-select').value);

function findPredicate(condition, name) {
  if (Array.isArray(condition)) {
    for (const item of condition) {
      const found = findPredicate(item, name);
      if (found) return found;
    }
    return null;
  }
  if (!condition || typeof condition !== 'object') return null;
  for (const [key, value] of Object.entries(condition)) {
    if (key === name && value && typeof value === 'object') return value;
    const found = findPredicate(value, name);
    if (found) return found;
  }
  return null;
}

function showDefinition() {
  const definition = currentDefinition();
  if (!definition) return;
  const body = definition.body;
  $('def-badge').textContent = definition.hash;
  const status = $('def-status');
  status.textContent = definition.status;
  status.className = 'badge status' + (definition.status === 'draft' ? ' draft' : '');
  $('def-label').textContent = definition.description || definition.label;
  $('c-concept').value = body.concept.primary;
  $('c-absent').value = (body.missingness.treat_as_absent || []).join(', ') || '(nothing)';
  $('c-review').value = (body.missingness.route_to_review || []).join(', ');
  $('c-linkage').value = body.episode.require_linkage_confidence;

  let threshold = null;
  (body.evidence_rules || []).forEach((rule) => {
    threshold = threshold || findPredicate(rule.when, 'lab');
  });
  $('c-glucose').value = threshold ? threshold.value : '';
  $('c-wmin').value = body.window ? body.window.min : '';
  $('c-wmax').value = body.window ? body.window.max : '';
  $('c-unresolved').value = body.window ? body.window.on_unresolved_onset : 'review';
}

async function buildCandidate() {
  const definition = currentDefinition();
  const body = definition.body;
  const changes = {};
  let threshold = null;
  (body.evidence_rules || []).forEach((rule) => {
    threshold = threshold || findPredicate(rule.when, 'lab');
  });
  const glucose = Number($('c-glucose').value);
  if (threshold && glucose && glucose !== threshold.value) {
    changes['evidence_rules.supported.lab.value'] = glucose;
  }
  if (body.window) {
    const min = Number($('c-wmin').value);
    const max = Number($('c-wmax').value);
    if (min !== body.window.min) changes['window.min'] = min;
    if (max !== body.window.max) changes['window.max'] = max;
    if ($('c-unresolved').value !== body.window.on_unresolved_onset) {
      changes['window.on_unresolved_onset'] = $('c-unresolved').value;
    }
  }
  const note = $('candidate-note');
  if (!Object.keys(changes).length) {
    note.textContent = 'Nothing was changed, so there is no candidate to build.';
    $('candidate-yaml').textContent = '';
    $('btn-download').disabled = true;
    return;
  }
  try {
    CANDIDATE = await post('/definitions/candidate', {
      definition_id: definition.id,
      base_version: definition.version,
      changes,
    });
  } catch (error) { note.textContent = error.message; return; }
  note.textContent =
    `${CANDIDATE.note} Applied: ${CANDIDATE.applied_changes.join('; ')}`;
  $('candidate-yaml').textContent = CANDIDATE.yaml;
  $('btn-download').disabled = false;
}

async function evaluateDefinition() {
  const definition = currentDefinition();
  const target = $('def-eval');
  target.innerHTML = 'evaluating…';
  let body;
  try {
    body = await post('/evaluate', {
      definition_id: definition.id,
      version: definition.version,
      allow_draft: definition.status === 'draft',
    });
  } catch (error) { fail(target, error); return; }
  const manifest = body.manifest;
  target.innerHTML = '';
  const box = el('div', 'summary');
  Object.entries(manifest.counts_by_verdict).forEach(([verdict, count]) => {
    stat(box, count, verdict, verdict === 'case' ? 'ok' : null);
  });
  target.appendChild(box);
  target.appendChild(el('p', 'hint',
    `definition ${manifest.definition_hash} · normalizer ` +
    `${manifest.normalizer_version} · extractor ${manifest.extractor_version} · ` +
    `snapshot ${manifest.data_snapshot_id} · manifest ${manifest.manifest_id} · ` +
    `results ${manifest.results_hash}`));
  target.appendChild(el('p', 'hint',
    `${body.review_set.length} episode(s) in the review set, reported ` +
    `separately rather than counted or discarded.`));

  const table = el('table', 'grid');
  table.innerHTML =
    '<tr><th>episode</th><th>verdict</th><th>state</th><th>rule</th><th>reason</th></tr>';
  body.assignments
    .filter((a) => a.verdict !== 'excluded')
    .slice(0, 25)
    .forEach((assignment) => {
      const row = el('tr');
      row.appendChild(el('td', null, assignment.episode_id));
      row.appendChild(el('td', null, assignment.verdict));
      row.appendChild(el('td', null, assignment.evidence_state));
      row.appendChild(el('td', null, assignment.matched_rule_id || '—'));
      row.appendChild(el('td', 'reason', assignment.reason));
      table.appendChild(row);
    });
  target.appendChild(table);
}

async function compareDefinitions() {
  const definition = currentDefinition();
  const target = $('def-compare');
  target.innerHTML = 'comparing…';
  let body;
  try {
    body = await api(`/definitions/${definition.id}/compare?` + query({
      left: $('cmp-left').value,
      right: $('cmp-right').value,
      scope: $('cmp-scope').value,
    }));
  } catch (error) { fail(target, error); return; }
  target.innerHTML = '';
  target.appendChild(el('p', 'hint', body.summary));
  const table = el('table', 'grid');
  table.innerHTML =
    '<tr><th>episode</th><th>left</th><th>right</th><th>why it moved</th></tr>';
  body.discordant.slice(0, 20).forEach((entry) => {
    const row = el('tr');
    row.appendChild(el('td', null, entry.episode_id));
    row.appendChild(el('td', null, entry.verdict_a));
    row.appendChild(el('td', null, entry.verdict_b));
    row.appendChild(el('td', 'reason', entry.reason_b));
    table.appendChild(row);
  });
  target.appendChild(table);
}

/* ------------------------------------------------------ 4. retrieval */

async function search() {
  const summary = $('r-summary');
  const results = $('r-results');
  summary.innerHTML = '';
  results.innerHTML = 'searching…';
  const path = $('r-path').value;
  const concept = $('r-concept').value;
  const isGroup = concept.startsWith('group: ');
  const params = {
    concept: isGroup ? '' : concept,
    group: isGroup ? concept.slice(7) : '',
    assertion: selected($('r-assertion')),
    top_k: $('r-topk').value,
  };

  let body;
  try {
    if (path === 'precise') {
      body = await api('/retrieve?' + query({
        ...params,
        verdict: selected($('r-verdict')),
        representation: selected($('r-representation')),
        window_min: $('r-wmin').value,
        window_max: $('r-wmax').value,
      }));
    } else {
      body = await api('/discover?' + query({...params, mode: $('r-mode').value}));
    }
  } catch (error) { fail(results, error); return; }

  stat(summary, body.count, path === 'precise' ? 'episodes' : 'mentions');
  stat(summary, body.negation_false_positives, 'asserting absence',
    body.negation_false_positives ? 'alert' : 'ok');
  stat(summary, body.negation_false_positive_rate.toFixed(4), 'negation FP rate');
  stat(summary, body.usable_as_cohort ? 'yes' : 'no', 'usable as a cohort',
    body.usable_as_cohort ? 'ok' : 'alert');

  results.innerHTML = '';
  if (body.cohort_note) results.appendChild(el('p', 'hint', body.cohort_note));
  (body.notes || []).forEach((note) => results.appendChild(el('p', 'hint', note)));

  const rows = path === 'precise' ? body.records : body.mentions;
  rows.forEach((row) => {
    const card = el('div', 'rec');
    const meta = el('div', 'meta');
    if (path === 'precise') {
      meta.appendChild(el('span', null, row.episode_id));
      meta.appendChild(el('span', null, row.study_id + ' · ' + row.representation));
      meta.appendChild(el('span', 'pill', String(row.verdict)));
      meta.appendChild(el('span', 'pill', String(row.evidence_state)));
      meta.appendChild(el('span', null, `onset offset ${row.onset_offset_days}`));
      if (row.linkage_review_required) meta.appendChild(el('span', 'pill absent', 'linkage flagged'));
      card.appendChild(meta);
      card.appendChild(el('div', 'snip', row.snippet || ''));
    } else {
      meta.appendChild(el('span', null, row.subject_id));
      meta.appendChild(el('span', 'pill ' + row.assertion, row.assertion));
      meta.appendChild(el('span', null, row.match_kind));
      if (row.cue) meta.appendChild(el('span', null, `cue: ${row.cue}`));
      meta.appendChild(el('span', 'pill', 'candidate'));
      card.appendChild(meta);
      card.appendChild(el('div', 'snip', row.sentence));
    }
    results.appendChild(card);
  });
}

/* ---------------------------------------------------------- 5. agent */

function renderClarification(clarification) {
  const box = $('a-clarification');
  box.classList.remove('hidden');
  box.innerHTML = '';
  box.appendChild(el('h3', null, 'Clarification needed before anything is run'));
  box.appendChild(el('p', null, clarification.ambiguity));
  box.appendChild(el('p', null, clarification.effect));
  const list = el('ul');
  clarification.options.forEach((option) => list.appendChild(el('li', null, option)));
  box.appendChild(list);
  box.appendChild(el('p', 'hint',
    'No specification was compiled and no number was produced.'));
}

async function compileOnly() {
  $('a-clarification').classList.add('hidden');
  $('a-result').innerHTML = '';
  $('a-trace').textContent = 'Ask a question to produce a trace.';
  let body;
  try {
    body = await post('/agent/compile', {question: $('a-question').value});
  } catch (error) {
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
  try {
    body = await post('/agent/ask', {question: $('a-question').value});
  } catch (error) {
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
  stat(box, body.summary.primary_case_count, 'primary cases', 'ok');
  stat(box, body.summary.review_set_count, 'review set (reported separately)');
  stat(box, body.summary.linkage_flagged, 'linkage flagged');
  stat(box, body.statistics.incidence_proportion, 'incidence proportion');
  result.appendChild(box);
  result.appendChild(el('p', 'hint',
    `definition ${body.definition.id}.v${body.definition.version} ` +
    `(${body.definition.status}) hash ${body.definition.hash} · normalizer ` +
    `${body.versions.normalizer_version} · extractor ` +
    `${body.versions.extractor_version} · manifest ${body.manifest_id} · ` +
    `results ${body.results_hash}`));
  result.appendChild(el('p', 'hint', body.statistics.caveat));
  const limits = el('ul');
  body.limitations.forEach((limitation) => limits.appendChild(el('li', 'hint', limitation)));
  result.appendChild(el('h2', null, 'Limitations'));
  result.appendChild(limits);

  try {
    const trace = await api(`/trace/${body.manifest_id}`);
    $('a-trace').textContent = trace.rendered;
  } catch (error) {
    $('a-trace').textContent = `trace unavailable: ${error.message}`;
  }
  loadKnowledge();
}

async function loadKnowledge() {
  const target = $('k-status');
  try {
    const body = await api('/knowledge/status');
    target.innerHTML = '';
    const box = el('div', 'summary');
    stat(box, body.manifests, 'recorded executions');
    stat(box, body.definitions.length, 'definitions');
    stat(box, body.snapshots.length, 'snapshots');
    target.appendChild(box);
    target.appendChild(el('p', 'hint', body.note));
    if (body.definitions_used.length) {
      target.appendChild(el('p', 'hint', 'used in runs: ' + body.definitions_used.join(', ')));
    }
  } catch (error) { fail(target, error); }
}

/* ------------------------------------------------------------- wiring */

$('rec-study').addEventListener('change', loadRecords);
$('rec-state').addEventListener('change', loadRecords);
$('btn-episodes').addEventListener('click', () => loadEpisodes().catch(
  (error) => fail($('ep-list'), error)));
$('def-select').addEventListener('change', showDefinition);
$('btn-candidate').addEventListener('click', buildCandidate);
$('btn-evaluate').addEventListener('click', evaluateDefinition);
$('btn-compare').addEventListener('click', compareDefinitions);
$('btn-reset').addEventListener('click', showDefinition);
$('btn-download').addEventListener('click', () => {
  if (!CANDIDATE) return;
  const blob = new Blob([CANDIDATE.yaml], {type: 'text/yaml'});
  const link = el('a');
  link.href = URL.createObjectURL(blob);
  link.download = CANDIDATE.filename;
  link.click();
  URL.revokeObjectURL(link.href);
});
$('btn-retrieve').addEventListener('click', search);
$('btn-clear-assertion').addEventListener('click', () => {
  Array.from($('r-assertion').options).forEach((o) => { o.selected = false; });
  search();
});
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
    await loadEpisodes();
    await loadKnowledge();
    await search();
  } catch (error) {
    $('corpus-line').textContent =
      'Could not load: ' + (error.message || error) +
      ' — run `make demo` first, then reload.';
  }
})();
