/* Adverse event evidence layer — single page UI.
 *
 * Every panel reads the same API the CLI drives, so the two cannot disagree
 * about which definition version produced a number. Nothing is computed here:
 * the browser renders what the server says and never derives a count of its own.
 *
 * Two things this page refuses to do, because doing them would misrepresent the
 * system it is showing:
 *
 *   - it never offers a single "missing" filter. Assertion and availability are
 *     separate controls everywhere they appear.
 *   - it never renders a silver number without its two caveats, and never
 *     renders the ablation's stage table without the decision above it.
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
const fixed = (n, d) => (n === null || n === undefined ? '—' : Number(n).toFixed(d));

function stat(container, value, label, tone) {
  const box = el('div', 'stat' + (tone ? ' ' + tone : ''));
  box.appendChild(el('div', 'n', String(value)));
  box.appendChild(el('div', 'l', label));
  container.appendChild(box);
}

function table(container, headers, rows) {
  const node = el('table', 'grid');
  const head = el('tr');
  headers.forEach((h) => head.appendChild(el('th', null, h)));
  node.appendChild(head);
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

function section(container, title, headers, rows) {
  container.innerHTML = '';
  if (title) container.appendChild(el('h2', null, title));
  const holder = el('div');
  container.appendChild(holder);
  table(holder, headers, rows);
}

function fail(node, error) {
  node.innerHTML = '';
  node.appendChild(el('p', 'err', error.message || String(error)));
}

/* ---------------------------------------------------------------- tabs --- */

document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
    tab.classList.add('active');
    $(tab.dataset.panel).classList.add('active');
  });
});

/* ------------------------------------------------------------- summary --- */

let SUMMARY = null;

async function loadSummary() {
  SUMMARY = await api('/api/summary');
  $('corpus-line').textContent =
    `${SUMMARY.studies} studies · ${SUMMARY.subjects} subjects · ` +
    `${SUMMARY.records} source records · normalizer ${SUMMARY.normalizer_version} · ` +
    `extractor ${SUMMARY.extractor_version} (${SUMMARY.extraction_backend}) · ` +
    `snapshot ${SUMMARY.snapshot_id}`;

  const routes = $('route-summary');
  routes.innerHTML = '';
  const assertions = SUMMARY.assertions || {};
  stat(routes, assertions.present || 0, 'present');
  stat(routes, assertions.absent || 0, 'absent — documented negatives', 'good');
  stat(routes, assertions.uncertain || 0, 'uncertain');
  stat(routes, assertions.silent || 0, 'silent — nobody said anything', 'warn');
  const extraction = SUMMARY.extraction || {};
  stat(routes, extraction.recovered || 0, 'recovered from text');
  stat(routes, `${((extraction.abstention_rate || 0) * 100).toFixed(0)}%`,
    'abstention rate — a valid answer');

  const profiles = Object.entries(SUMMARY.profiles || {}).sort();
  table($('profile-table'),
    ['profile', 'study', 'term style', 'modifier home', 'dictionary',
     'supportability', 'note'],
    profiles.map(([name, body]) => [
      name, body.study_id, body.reported_term_style,
      (body.modifier_homes || []).join(', '),
      body.dictionary_version, body.supportability, body.note,
    ]));

  section($('reconciliation-table'),
    'Dictionary version reconciliation — mechanical, never a model',
    ['outcome', 'records', 'what it means'],
    [
      ['unchanged', (SUMMARY.reconciliation || {}).unchanged || 0,
       'the code is identical under the target version'],
      ['remapped_mechanically',
       (SUMMARY.reconciliation || {}).remapped_mechanically || 0,
       'the concept persists across versions, so a 1:1 map applies; the ' +
       'original code is preserved beside it'],
      ['flagged_for_review',
       (SUMMARY.reconciliation || {}).flagged_for_review || 0,
       'the concept has no code under the target version. A human decides ' +
       'what it becomes — no model ever recodes it'],
    ]);

  const profileSelect = $('rec-profile');
  profiles.forEach(([name]) => {
    const option = el('option', null, name);
    option.value = name;
    profileSelect.appendChild(option);
  });

  const defs = await api('/definitions');
  ['def-select', 'abl-select'].forEach((id) => {
    const select = $(id);
    select.innerHTML = '';
    defs.definitions.forEach((d) => {
      const option = el('option', null, `${d.key} — ${d.label}`);
      option.value = `${d.id}:${d.version}`;
      select.appendChild(option);
    });
    const latest = defs.definitions.filter((d) => d.id === 'cutaneous_mucosal');
    if (latest.length) {
      select.value = `${latest[latest.length - 1].id}:${latest[latest.length - 1].version}`;
    }
  });
}

/* --------------------------------------------------- 1. records & routes --- */

async function loadRecords() {
  const params = {
    profile: $('rec-profile').value,
    assertion: $('rec-assertion').value,
    availability: $('rec-availability').value,
    method: $('rec-method').value,
    limit: 200,
  };
  const body = await api('/api/records?' + query(params));
  const list = $('rec-list');
  list.innerHTML = '';
  if (!body.records.length) {
    list.appendChild(el('p', 'hint', 'No record matches those two filters.'));
    return;
  }
  body.records.forEach((row) => {
    const item = el('button', 'rec');
    item.appendChild(el('div', 'rec-id', row.source_record_id));
    const assertion = row.modifier_assertion || 'silent';
    item.appendChild(el('div', 'rec-meta',
      `${row.profile} · ${assertion} · ${row.modifier_availability}` +
      (row.modifier_method ? ` · ${row.modifier_method}` : '')));
    item.addEventListener('click', () => showRecord(row.record_id));
    list.appendChild(item);
  });
}

function highlight(text, spans) {
  if (!spans.length) return esc(text);
  const ordered = spans.slice().sort((a, b) => a.start - b.start);
  let out = '';
  let cursor = 0;
  ordered.forEach((span) => {
    if (span.start < cursor) return;
    out += esc(text.slice(cursor, span.start));
    out += `<mark class="f-assertion">${esc(text.slice(span.start, span.end))}</mark>`;
    cursor = span.end;
  });
  return out + esc(text.slice(cursor));
}

async function showRecord(recordId) {
  const body = await api('/api/records/' + encodeURIComponent(recordId));
  const badge = $('rec-badge');
  badge.textContent = body.profile;
  badge.className = 'badge';

  const source = $('rec-source');
  source.innerHTML = '';
  const modifier = body.attributes.mucosal_involvement || {};
  const spans = (modifier.evidence || []).filter((s) => s.kind === 'text');

  const term = body.attributes.reported_term || {};
  const termBlock = el('div', 'doc-item');
  termBlock.appendChild(el('div', 'doc-head', 'AETERM — the investigator’s own words'));
  const termText = el('div');
  termText.innerHTML = highlight(String(term.value || ''),
    spans.filter((s) => s.doc_id.endsWith('AETERM')));
  termBlock.appendChild(termText);
  source.appendChild(termBlock);

  (body.documents || []).forEach((doc) => {
    const block = el('div', 'doc-item');
    block.appendChild(el('div', 'doc-head', `${doc.kind} — ${doc.doc_id}`));
    const text = el('div');
    text.innerHTML = highlight(doc.text, spans.filter((s) => s.doc_id === doc.doc_id));
    block.appendChild(text);
    source.appendChild(block);
  });

  const coded = body.coded_event;
  if (coded) {
    const block = el('div', 'doc-item');
    block.appendChild(el('div', 'doc-head', 'AEDECOD — the coded term, never rewritten'));
    block.appendChild(el('div', null,
      `${coded.code} (${coded.dictionary_version}) → ` +
      `${coded.reconciliation}` +
      (coded.reconciled_to && coded.reconciled_to !== coded.code
        ? ` → ${coded.reconciled_to}` : '')));
    if (coded.note) block.appendChild(el('div', 'hint', coded.note));
    source.appendChild(block);
  }

  const rows = Object.entries(body.attributes).map(([name, a]) => [
    name,
    a.value === null || a.value === undefined ? '—' : String(a.value),
    a.assertion || '—',
    a.availability,
    a.method || '—',
    a.source_variable || '—',
    a.confidence === null || a.confidence === undefined ? '—' : fixed(a.confidence, 2),
    a.note || '',
  ]);
  section($('rec-view'), null,
    ['attribute', 'value', 'assertion', 'availability', 'method', 'variable',
     'conf', 'note'],
    rows);
}

['rec-profile', 'rec-assertion', 'rec-availability', 'rec-method'].forEach((id) => {
  $(id).addEventListener('change', () => loadRecords().catch((e) => fail($('rec-list'), e)));
});

/* ------------------------------------------------------ 2. the decision --- */

async function runAblation() {
  const [id, version] = $('abl-select').value.split(':');
  const decision = $('ablation-decision');
  decision.textContent = 'running…';
  try {
    const body = await api(`/eval/ablation?${query({definition_id: id, version})}`);
    decision.innerHTML = '';
    decision.appendChild(el('strong', null, 'DECISION: '));
    decision.appendChild(document.createTextNode(body.decision));

    section($('ablation-stages'), 'Stages, cumulative',
      ['stage', 'evaluated', 'ascertained', 'asc. fraction', 'cases',
       'correct', 'wrong', 'precision', 'recall'],
      body.stages.map((s) => [
        s.stage, s.n_evaluated, s.n_ascertained, fixed(s.ascertainable_fraction, 3),
        s.n_case, s.n_case_correct, s.n_case_incorrect,
        fixed(s.precision, 3), fixed(s.recall, 3),
      ]));

    const increments = $('ablation-increments');
    increments.innerHTML = '';
    increments.appendChild(el('h2', null, 'What each stage bought'));
    body.increments.forEach((inc) => {
      const pane = el('div', 'pane');
      pane.appendChild(el('h2', null, `${inc.from_stage} → ${inc.to_stage}`));
      const list = el('ul', 'hint');
      inc.reasons.forEach((reason) => list.appendChild(el('li', null, reason)));
      pane.appendChild(list);
      const verdict = el('div', 'notice' + (inc.material ? '' : ' warn'));
      verdict.textContent = inc.decision;
      pane.appendChild(verdict);
      increments.appendChild(pane);
    });
    increments.appendChild(el('p', 'hint', body.note));

    table($('ablation-criteria'), ['criterion', 'threshold'],
      Object.entries(body.materiality_criteria).sort());
  } catch (error) {
    fail(decision, error);
  }
}

$('btn-ablation').addEventListener('click', runAblation);

/* ------------------------------------------------------ 3. silver ------- */

async function loadSilver() {
  const caveats = await api('/eval/caveats');
  const holder = $('silver-caveats');
  holder.innerHTML = '';
  caveats.silver.forEach((text) => {
    const box = el('div', 'notice warn');
    box.textContent = text;
    holder.appendChild(box);
  });

  const body = await api('/eval/silver');
  const summary = $('silver-summary');
  summary.innerHTML = '';
  const overall = body.overall;
  stat(summary, overall.eligible_records, 'eligible records');
  stat(summary, fixed(overall.precision, 3), 'precision');
  stat(summary, fixed(overall.recall, 3), 'recall');
  stat(summary, fixed(overall.coverage, 3), 'coverage');
  stat(summary, fixed(overall.abstention_rate, 3), 'abstention rate');
  stat(summary, overall.disagreements, 'disagreements',
    overall.disagreements ? 'warn' : 'good');

  table($('silver-assertion'),
    ['assertion', 'n', 'answered', 'correct', 'recall', 'precision'],
    Object.entries(body.by_assertion).map(([name, row]) => [
      name, row.n, row.answered, row.correct,
      fixed(row.recall, 3), fixed(row.precision, 3),
    ]));

  const calibration = $('silver-calibration');
  calibration.innerHTML = '';
  const head = el('div', 'summary');
  stat(head, fixed(body.calibration.brier_score, 4), 'Brier score');
  stat(head, fixed(body.calibration.expected_calibration_error, 4),
    'expected calibration error');
  calibration.appendChild(head);
  const grid = el('div');
  calibration.appendChild(grid);
  table(grid, ['confidence bin', 'n', 'mean confidence', 'observed', 'gap'],
    body.calibration.reliability.map((r) => [
      r.bin, r.n, fixed(r.mean_confidence, 3),
      fixed(r.observed_accuracy, 3), fixed(r.gap, 3),
    ]));
  calibration.appendChild(el('p', 'hint', body.calibration.note));

  const queue = body.adjudication_queue || [];
  $('queue-badge').textContent = `${queue.length} rows`;
  const list = $('silver-queue');
  list.innerHTML = '';
  queue.forEach((row) => {
    const item = el('div', 'evt');
    item.appendChild(el('div', 'evt-head',
      `${row.source_record_id} · ${row.profile}`));
    item.appendChild(el('div', null,
      `structured says ${row.structured_assertion || '—'}; ` +
      `extraction says ${row.extracted_assertion || 'abstained'}` +
      (row.extracted_confidence !== null && row.extracted_confidence !== undefined
        ? ` at ${fixed(row.extracted_confidence, 2)}` : '')));
    if (row.text) item.appendChild(el('div', 'hint', row.text));
    item.appendChild(el('div', 'pill', row.queue_reason));
    list.appendChild(item);
  });
}

/* --------------------------------------------- 4. verdicts & denominators --- */

async function evaluate() {
  const [id, version] = $('def-select').value.split(':');
  const summary = $('verdict-summary');
  summary.innerHTML = 'evaluating…';
  try {
    const body = await post('/evaluate', {
      definition_id: id, version: Number(version), save: true,
    });
    const counts = body.manifest.counts_by_verdict;
    summary.innerHTML = '';
    stat(summary, counts.case || 0, 'case', 'good');
    stat(summary, counts.non_case || 0, 'non_case — evaluated negatives');
    stat(summary, counts.review || 0, 'review');
    stat(summary, counts.not_ascertainable || 0, 'not ascertainable', 'warn');

    table($('denominator-table'),
      ['study', 'profile', 'total', 'case', 'non_case', 'review', 'not asc.',
       'asc. fraction', 'incidence'],
      body.denominators.map((d) => [
        d.study_id, d.profile, d.n_total, d.n_case, d.n_non_case, d.n_review,
        d.n_not_ascertainable, fixed(d.ascertainable_fraction, 3),
        d.incidence_within_ascertainable === null
          ? '—' : fixed(d.incidence_within_ascertainable, 3),
      ]));
    $('denominator-note').textContent = body.denominator_note;

    table($('verdict-table'),
      ['record', 'verdict', 'route', 'reason'],
      body.assignments.slice(0, 40).map((a) => [
        a.record_id, a.verdict,
        Object.values(a.attribute_methods || {}).join(', '),
        a.reason,
      ]));
  } catch (error) {
    fail(summary, error);
  }
}

async function compareVersions() {
  const [id] = $('def-select').value.split(':');
  const view = $('compare-view');
  view.innerHTML = 'comparing…';
  try {
    const body = await api(`/definitions/${id}/compare?` + query({
      left: $('cmp-left').value,
      right: $('cmp-right').value,
      scope: $('cmp-scope').value,
    }));
    view.innerHTML = '';
    view.appendChild(el('p', null, body.summary));
    const grid = el('div');
    view.appendChild(grid);
    table(grid, ['record', 'left', 'right', 'why it moved'],
      body.discordant.slice(0, 20).map((d) => [
        d.record_id, d.verdict_a, d.verdict_b, d.reason_b,
      ]));
  } catch (error) {
    fail(view, error);
  }
}

$('btn-evaluate').addEventListener('click', evaluate);
$('btn-compare').addEventListener('click', compareVersions);

/* ------------------------------------------------- 5. retrieval ---------- */

async function search() {
  const summary = $('r-summary');
  const results = $('r-results');
  summary.innerHTML = 'searching…';
  results.innerHTML = '';
  try {
    if ($('r-path').value === 'discovery') {
      const body = await api('/discover?' + query({
        text: $('r-text').value, top_k: $('r-topk').value,
      }));
      summary.innerHTML = '';
      stat(summary, body.count, 'mentions');
      stat(summary, 'all candidates', 'not cohort-eligible', 'warn');
      const holder = el('div');
      results.innerHTML = '';
      results.appendChild(el('p', 'hint', body.cohort_note));
      results.appendChild(holder);
      table(holder, ['subject', 'assertion', 'value', 'surface', 'rule'],
        body.mentions.map((m) => [
          m.subject_id, m.assertion, m.value, m.surface, m.rule,
        ]));
      return;
    }
    const body = await api('/retrieve?' + query({
      assertion: selected($('r-assertion')),
      availability: selected($('r-availability')),
      method: selected($('r-method')),
      verdict: selected($('r-verdict')),
      top_k: $('r-topk').value,
    }));
    summary.innerHTML = '';
    stat(summary, body.count, 'records');
    stat(summary, body.usable_as_cohort ? 'yes' : 'no', 'usable as a cohort',
      body.usable_as_cohort ? 'good' : 'warn');
    results.innerHTML = '';
    (body.notes || []).forEach((note) => results.appendChild(el('p', 'hint', note)));
    const holder = el('div');
    results.appendChild(holder);
    table(holder,
      ['record', 'verdict', 'assertion', 'availability', 'method', 'variable',
       'code'],
      body.records.map((r) => [
        r.record_id, r.verdict, r.assertion, r.availability, r.method,
        r.source_variable, r.code,
      ]));
  } catch (error) {
    fail(summary, error);
  }
}

$('btn-search').addEventListener('click', search);

/* ------------------------------------------------------ 6. ask & trace --- */

function renderConflict(conflict) {
  const box = $('a-clarification');
  box.classList.remove('hidden');
  box.innerHTML = '';
  box.appendChild(el('h2', null, 'Not run — the question conflicts with the definition it names'));
  box.appendChild(el('p', null, conflict.conflict));
  if (conflict.bound_definition) {
    box.appendChild(el('p', 'hint', `bound to ${conflict.bound_definition}`));
  }
  box.appendChild(el('p', null, conflict.effect));
  const list = el('ul');
  (conflict.options || []).forEach((option) => list.appendChild(el('li', null, option)));
  box.appendChild(list);
  box.appendChild(el('p', 'hint',
    'The agent does not override a bound definition to accommodate a question. ' +
    'A different rule is a new version, not a parameter. Nothing was computed.'));
}

function renderSupport(support) {
  const holder = $('a-support');
  holder.innerHTML = '';
  Object.entries(support || {}).forEach(([modifier, screen]) => {
    const pane = el('div', 'pane');
    pane.appendChild(el('h2', null, `Supportability screen — ${modifier}`));
    pane.appendChild(el('p', 'hint', screen.note));
    const grid = el('div');
    pane.appendChild(grid);
    table(grid, ['study', 'profile', 'status', 'reason'],
      (screen.studies || []).map((s) => [
        s.study_id, s.profile, s.status, s.reason,
      ]));
    holder.appendChild(pane);
  });
}

async function ask(executeIt) {
  const question = $('a-question').value;
  $('a-clarification').classList.add('hidden');
  $('a-spec-wrap').classList.add('hidden');
  $('a-result').innerHTML = '';
  $('a-support').innerHTML = '';
  $('a-trace').textContent = 'running…';
  try {
    const body = await post(executeIt ? '/agent/ask' : '/agent/compile',
      {question, save: executeIt});
    if (!executeIt) {
      $('a-spec-wrap').classList.remove('hidden');
      $('a-spec-badge').textContent = 'not executed';
      $('a-spec').textContent = JSON.stringify(body.spec, null, 2);
      $('a-trace').textContent = 'Compiled only — nothing was executed.';
      return;
    }
    $('a-spec-wrap').classList.remove('hidden');
    $('a-spec-badge').textContent = 'executed';
    $('a-spec-badge').className = 'badge';
    $('a-spec').textContent = JSON.stringify(body.spec, null, 2);
    renderSupport(body.supportability);

    const result = $('a-result');
    const summary = el('div', 'summary');
    const counts = body.cohort.counts_by_verdict;
    stat(summary, counts.case || 0, 'case', 'good');
    stat(summary, counts.non_case || 0, 'non_case');
    stat(summary, counts.review || 0, 'review');
    stat(summary, counts.not_ascertainable || 0, 'not ascertainable', 'warn');
    stat(summary, fixed(body.cohort.overall.ascertainable_fraction, 3),
      'ascertainable fraction');
    stat(summary, body.cohort.overall.incidence_within_ascertainable === null
      ? '—' : fixed(body.cohort.overall.incidence_within_ascertainable, 3),
      'incidence within ascertainable');
    result.appendChild(summary);
    result.appendChild(el('p', 'hint', body.cohort.denominator_note));
    result.appendChild(el('p', 'hint',
      `tools called: ${(body.tools_called || []).join(', ')} · manifest ` +
      `${body.manifest_id} · results ${body.results_hash}`));
    const limits = el('ul', 'hint');
    (body.limitations || []).forEach((limit) => limits.appendChild(el('li', null, limit)));
    result.appendChild(limits);

    $('a-trace').textContent = body.trace
      ? renderTraceText(body.trace)
      : 'no trace was produced';
  } catch (error) {
    if (error.status === 409 && error.body && error.body.conflict) {
      renderConflict(error.body.conflict);
      $('a-trace').textContent =
        'Nothing was executed, so there is no number and no trace.';
      return;
    }
    fail($('a-result'), error);
    $('a-trace').textContent = '';
  }
}

const INDENT = {
  result: 0, analysis: 1, cohort: 2, definition: 3, attribute: 4,
  record: 5, span: 6,
};

function renderTraceText(trace) {
  const lines = [];
  trace.links.forEach((link) => {
    const pad = '  '.repeat(INDENT[link.level] ?? 0);
    lines.push(`${pad}${link.level.padEnd(10)} ${link.identifier}`);
    if (link.detail) lines.push(`${pad}${''.padEnd(10)} ${link.detail}`);
  });
  if (!trace.complete) {
    lines.push('');
    lines.push(`INCOMPLETE: the chain breaks at ${trace.broken_at}. ` +
      'A number that cannot be traced to source is not reportable.');
  }
  return lines.join('\n');
}

$('btn-ask').addEventListener('click', () => ask(true));
$('btn-compile').addEventListener('click', () => ask(false));
document.querySelectorAll('.examples .link').forEach((button) => {
  button.addEventListener('click', () => {
    $('a-question').value = button.dataset.q;
    ask(true);
  });
});

async function loadTools() {
  const body = await api('/agent/tools');
  const holder = $('tool-table');
  holder.innerHTML = '';
  holder.appendChild(el('p', 'hint', body.note));
  const grid = el('div');
  holder.appendChild(grid);
  table(grid, ['tool', 'permission', 'writes source records', 'description'],
    body.tools.map((t) => [
      t.name, t.permission, t.writes_source_records ? 'YES' : 'no',
      t.description,
    ]));
}

/* -------------------------------------------------------------- start --- */

(async function start() {
  try {
    await loadSummary();
    await loadRecords();
    await loadSilver();
    await loadTools();
    await runAblation();
  } catch (error) {
    $('corpus-line').textContent =
      `could not load: ${error.message}. Run \`aelayer generate\` first.`;
  }
})();
