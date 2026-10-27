const $ = (id) => document.getElementById(id);
const labels = {human: 'Mensch', random: 'Random', mcts: 'MCTS', policy: 'Policy'};
const suitNames = ['Karo', 'Herz', 'Pik', 'Kreuz'];
const kindNames = {bid: 'reizt', pickup: 'entscheidet über den Skat', press: 'drückt', declare: 'sagt an', play: 'wählt eine Karte', confirm_concession: 'prüft die Aufgabe', confirm_claim: 'prüft die Abkürzung'};
const decisionNames = {bid: 'Reizen', pickup: 'Skataufnahme', press: 'Drücken', declare: 'Spielansage', play: 'Kartenspiel', confirm_concession: 'Aufgabe bestätigen', confirm_claim: 'Abkürzung bestätigen'};
const specialActions = {concede: 'Aufgeben', open: 'Gegenhand öffnen', claim0: 'Spielgewinn beanspruchen', claim1: 'Schneider beanspruchen', claim2: 'Schwarz beanspruchen'};
const manualSeat = seat => seat.kind === 'human' || seat.execution === 'manual';
const terminal = new Set(['finished', 'passed', 'error', 'stopped']);
const bestPolicySettings = {
  kind: 'policy', policy: 'benchmark_runs/iss-run-1/policy.pt',
  bidding: 'neural', biddingModel: 'benchmark_runs/bidding-run-1/bidding.pt', biddingThreshold: 0.55,
  declaration: 'heuristic', declarationModel: '', pickup: 'always', pickupModel: '', pickupMargin: 0, beliefModel: '',
};
const seatPresets = [
  {
    label: 'Beste Policy gegen Random',
    description: 'Höchste Policy-Punktzahl im festen Random-Vergleich: ISS-Policy mit neuronalem Reizen, heuristischer Ansage und Standard-Skataufnahme.',
    settings: bestPolicySettings,
  },
  {
    label: 'Alles neuronal',
    description: 'ISS-Policy mit neuronalem Reizen, neuronaler Ansage und neuronaler Skataufnahme.',
    settings: {...bestPolicySettings,
      declaration: 'neural', declarationModel: 'benchmark_runs/declaration-run-3/declaration.pt',
      pickup: 'neural', pickupModel: 'benchmark_runs/pickup-run-2/pickup.pt'},
  },
];
let state = null, appConfig = null, busy = false, selected = [], selectedOption = null, pendingId = null, generation = 0;
let deck = read('skat.deck') || 'fr';
if (!['fr', 'de', 'by'].includes(deck)) deck = 'fr';

function read(key) { try { return localStorage.getItem(key); } catch { return null; } }
function save(key, value) { try { value === null ? localStorage.removeItem(key) : localStorage.setItem(key, value); } catch { /* Storage may be disabled. */ } }
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function button(text, action, className = '') {
  const node = el('button', className, text);
  node.type = 'button'; node.addEventListener('click', action);
  return node;
}
function field(label, input) {
  const node = el('label', '', label); node.append(input); return node;
}
function input(name, value = '', type = 'text') {
  const node = el('input'); node.name = name; node.type = type; node.value = value; return node;
}
function select(name, options, value) {
  const node = el('select'); node.name = name;
  for (const [key, label] of options) { const option = el('option', '', label); option.value = key; node.append(option); }
  node.value = value; return node;
}
function showError(error) { $('error').textContent = error.message || String(error); $('error').hidden = false; }
function clearError() { $('error').hidden = true; }
async function api(path, method = 'GET', body) {
  const response = await fetch(path, {method, headers: body === undefined ? {} : {'Content-Type': 'application/json'}, body: body === undefined ? undefined : JSON.stringify(body)});
  const value = response.status === 204 ? null : await response.json();
  if (!response.ok) { const error = new Error(value?.error || `HTTP ${response.status}`); error.status = response.status; throw error; }
  return value;
}
function declarationName(d) {
  if (!d) return 'Noch keine Ansage';
  return [d.kind === 'suit' ? suitNames[d.suit] : d.kind === 'grand' ? 'Grand' : 'Null', d.hand && 'Hand', d.ouvert ? 'Ouvert' : d.schwarz_announced ? 'Schwarz angesagt' : d.schneider_announced && 'Schneider angesagt'].filter(Boolean).join(' · ');
}
function cardLabel(card) {
  const rank = deck === 'fr' ? ({J: 'Bube', Q: 'Dame', K: 'König', X: 'Zehn', A: 'Ass'}[card.rank] || card.rank) : ({J: 'Unter', Q: 'Ober', K: 'König', X: 'Zehn', A: 'Ass'}[card.rank] || card.rank);
  const suit = deck === 'fr' ? {'♦': 'Karo', '♥': 'Herz', '♠': 'Pik', '♣': 'Kreuz'} : {'♦': 'Schellen', '♥': 'Herz', '♠': 'Grün', '♣': 'Eichel'};
  return `${suit[card.suit]} ${rank}`;
}
function cardView(card, {compact = false, legal = false, chosen = false, suggested = false, onClick} = {}) {
  const node = onClick ? button('', onClick) : el('div');
  node.className = `card ${compact ? 'compact' : ''} ${card ? 'face' : 'back'} ${legal ? 'legal' : ''} ${chosen ? 'chosen' : ''} ${suggested ? 'suggested' : ''} deck-${deck}`;
  if (!card) { node.setAttribute('aria-label', 'Verdeckte Karte'); return node; }
  node.title = `${cardLabel(card)}${suggested ? ' · Agentenvorschlag' : ''}`; node.setAttribute('aria-label', node.title);
  if (chosen) node.setAttribute('aria-pressed', 'true');
  const rank = ({X: '10', Q: deck === 'fr' ? 'D' : 'O', J: deck === 'fr' ? 'B' : 'U'}[card.rank] || card.rank);
  const fallback = el('span', `fallback ${['♦', '♥'].includes(card.suit) ? 'red' : ''}`, `${rank}\n${card.suit}`);
  if (appConfig?.decks[deck]) {
    node.classList.add('sprite');
    node.style.backgroundPosition = `${(['A', 'K', 'Q', 'J', 'X', '9', '8', '7'].indexOf(card.rank) / 7) * 100}% ${(['♣', '♠', '♥', '♦'].indexOf(card.suit) / 3) * 100}%`;
  } else node.append(fallback);
  return node;
}
function actionLabel(kind, value) {
  const cardName = id => cardLabel({suit: '♦♥♠♣'[Math.floor(id / 8)], rank: '789JQKXA'[id % 8]});
  if (kind === 'press') return value.map(cardName).join(' + ');
  if (kind === 'play') return typeof value === 'number' ? cardName(value) : specialActions[value];
  if (kind === 'bid') return value === 0 ? 'Passen' : `Reizwert ${value}`;
  if (kind === 'pickup') return value ? 'Skat aufnehmen' : 'Hand spielen';
  if (kind === 'declare') return declarationName(value);
  return value ? 'Annehmen' : 'Weiterspielen';
}
function isSuggested(kind, value) {
  const same = (left, right) => typeof left === typeof right && (left && typeof left === 'object'
    ? right !== null && Object.keys(left).length === Object.keys(right).length && Object.keys(left).every(key => Object.hasOwn(right, key) && same(left[key], right[key]))
    : left === right);
  return state.pending?.kind === kind && Object.hasOwn(state.pending, 'suggestion')
    && same(state.pending.suggestion, value);
}
function setupSeats() {
  let stored;
  try { stored = JSON.parse(read('skat.config')); } catch { /* Use defaults. */ }
  $('seats').replaceChildren();
  for (let i = 0; i < 3; i++) {
    const defaults = stored?.seats?.[i] || {};
    const article = el('fieldset', 'seat-config');
    article.append(el('legend', '', `Platz ${i + 1}`));
    article.append(field('Name', input('label', defaults.label || (i === 0 ? 'Du' : `Bot ${i}`))));
    const shortcuts = el('div', 'seat-presets');
    shortcuts.append(el('small', 'muted', 'Schnellauswahl'));
    article.append(shortcuts);
    const kind = select('kind', Object.entries(labels), defaults.kind || (i === 0 ? 'human' : 'random'));
    article.append(field('Spieler', kind));
    const options = el('div', 'bot-options');
    const execution = select('execution', [['auto', 'Automatisch'], ['manual', 'Manuell · Agent schlägt vor']], defaults.execution || 'auto');
    options.append(field('Ausführung', execution));
    const policy = input('policy', defaults.policy); policy.setAttribute('list', 'checkpoints'); policy.placeholder = 'Pfad zum Policy-Checkpoint';
    const policyField = field('Policy-Modell', policy); options.append(policyField);
    const details = el('details', 'advanced'); details.append(el('summary', '', 'Strategien und Modelle'));
    for (const [name, title, choices, defaultValue] of [
      ['bidding', 'Reizen', [['conservative', 'Konservativ'], ['neural', 'Neuronal']], 'conservative'],
      ['declaration', 'Ansage', [['heuristic', 'Heuristik'], ['neural', 'Neuronal']], 'heuristic'],
      ['pickup', 'Skataufnahme', [['always', 'Standard'], ['neural', 'Neuronal']], 'always'],
    ]) {
      const strategy = select(name, choices, defaults[name] || defaultValue);
      const model = input(name + 'Model', defaults[name + 'Model']); model.setAttribute('list', 'checkpoints');
      const modelField = field('Modellpfad', model);
      strategy.addEventListener('change', () => { modelField.hidden = strategy.value !== 'neural'; });
      modelField.hidden = strategy.value !== 'neural';
      details.append(field(title, strategy), modelField);
    }
    const threshold = input('biddingThreshold', defaults.biddingThreshold ?? 2 / 3, 'number'); threshold.min = 0; threshold.max = 1; threshold.step = 'any';
    const margin = input('pickupMargin', defaults.pickupMargin ?? 0, 'number'); margin.min = 0; margin.max = 1; margin.step = 'any';
    const belief = input('beliefModel', defaults.beliefModel); belief.setAttribute('list', 'checkpoints');
    const beliefField = field('Belief-Modell (optional)', belief);
    details.append(field('Reizschwelle', threshold), field('Mindestvorsprung für Hand', margin), beliefField);
    options.append(details); article.append(options);
    const update = () => {
      options.hidden = kind.value === 'human';
      policyField.hidden = kind.value !== 'policy'; beliefField.hidden = kind.value !== 'mcts';
      // Hidden bot settings must not participate in native form validation.
      for (const control of options.querySelectorAll('input, select')) control.disabled = kind.value === 'human';
      policy.required = kind.value === 'policy';
    };
    kind.addEventListener('change', update);
    for (const preset of seatPresets) {
      const apply = button(preset.label, () => {
        const controls = [...article.querySelectorAll('[name]')];
        for (const control of controls) {
          if (Object.hasOwn(preset.settings, control.name)) control.value = String(preset.settings[control.name]);
        }
        for (const control of controls.filter(node => node.tagName === 'SELECT')) control.dispatchEvent(new Event('change'));
      });
      const models = Object.entries(preset.settings).filter(([key, path]) => path && (key === 'policy' || key.endsWith('Model')));
      const missing = appConfig?.checkpoints && models.some(([, path]) => !appConfig.checkpoints.includes(path));
      apply.disabled = Boolean(missing);
      apply.title = missing ? 'Benötigte Modelle sind lokal nicht vorhanden.' : preset.description;
      shortcuts.append(apply);
    }
    update(); $('seats').append(article);
  }
  if (stored) for (const key of ['seed', 'dealer', 'simulations', 'determinizations', 'workers', 'device']) if (stored[key] != null) $(key).value = stored[key];
}
function getConfig() {
  const config = {seats: []};
  for (const seat of $('seats').children) {
    const value = {};
    for (const node of seat.querySelectorAll('[name]')) value[node.name] = ['biddingThreshold', 'pickupMargin'].includes(node.name) ? Number(node.value) : node.value;
    if (value.kind === 'human') value.execution = 'manual';
    config.seats.push(value);
  }
  for (const key of ['seed', 'dealer', 'simulations', 'determinizations']) config[key] = Number($(key).value);
  config.workers = $('workers').value ? Number($('workers').value) : null;
  config.device = $('device').value;
  if ($('deal-code').value.trim()) config.dealCode = $('deal-code').value.trim();
  return config;
}
function updateDealInput() {
  const code = $('deal-code').value.trim().toUpperCase().replaceAll('-', '');
  $('dealer').disabled = Boolean(code);
  const body = code.slice(3).replaceAll('O', '0').replace(/[IL]/g, '1');
  const alphabet = '0123456789ABCDEFGHJKMNPQRSTVWXYZ';
  if (code.startsWith('SK1') && body.length === 13 && [...body].every(c => alphabet.includes(c))) {
    let dealer = 0;
    for (const char of body.slice(0, 11)) dealer = (dealer * 32 + alphabet.indexOf(char)) % 3;
    $('dealer').value = String(dealer);
  }
}
$('deal-code').addEventListener('input', updateDealInput);
async function copyDealText(text) {
  try { await navigator.clipboard.writeText(text); }
  catch {
    const fallback = el('textarea'); fallback.value = text; fallback.readOnly = true;
    $('deal-tools').append(fallback); fallback.focus(); fallback.select();
  }
}
async function prepareDeal(code) {
  const config = state.config;
  try { await api(`/api/sessions/${state.id}`, 'DELETE'); }
  catch (error) { if (error.status !== 404) { showError(error); return; } }
  generation++; state = null; busy = false; selected = []; pendingId = null;
  save('skat.session', null); save('skat.config', JSON.stringify(config));
  setupSeats(); $('deal-code').value = code; updateDealInput();
  $('game').hidden = true; $('setup').hidden = false; clearError();
}
function dealTools(code) {
  const node = el('div', 'controls');
  if (!code) return node;
  node.append(el('code', '', code),
    button('Code kopieren', () => copyDealText(code)),
    button('Link kopieren', () => copyDealText(`${location.origin}/#deal=${encodeURIComponent(code)}`)),
    button('Erneut spielen', () => prepareDeal(code)));
  return node;
}
function sortedHand(cards, declaration) {
  const suits = ['♣', '♠', '♦', '♥'];
  const isNull = declaration?.kind === 'null';
  const ranks = isNull ? 'AKQJX987' : 'AXKQJ987';
  const trump = declaration?.kind === 'suit' ? ['♦', '♥', '♠', '♣'][declaration.suit] : null;
  const group = card => isNull ? 0 : card.rank === 'J' ? 0 : card.suit === trump ? 1 : 2;
  return [...cards].sort((a, b) => group(a) - group(b)
    || suits.indexOf(a.suit) - suits.indexOf(b.suit)
    || ranks.indexOf(a.rank) - ranks.indexOf(b.rank));
}
function playerView(player) {
  const isSoloist = state.soloist === player.seat;
  const node = el('section', `player ${isSoloist ? 'soloist-player' : ''} ${state.nextPlayer === player.seat || state.pending?.seat === player.seat ? 'active-player' : ''}`);
  const meta = el('div', 'player-meta'); meta.append(el('strong', '', player.label));
  if (isSoloist) meta.append(el('strong', 'soloist-badge', 'Alleinspieler'));
  meta.append(el('span', '', `Platz ${player.seat + 1} · ${labels[player.kind]}${player.kind !== 'human' && manualSeat(state.config.seats[player.seat]) ? ' · manuell' : ''}${state.dealer === player.seat ? ' · Geber' : ''}`));
  meta.append(el('span', '', state.declaration?.kind === 'null' ? `${player.tricks} Stiche` : `${player.points} Augen · ${player.tricks} Stiche`)); node.append(meta);
  const hand = el('div', 'hand');
  for (const card of sortedHand(player.hand, state.declaration)) {
    const pending = state.pending;
    const legal = !busy && pending?.seat === player.seat && ['play', 'press'].includes(pending.kind) && pending.options.includes(card.id);
    const suggested = pending?.seat === player.seat && (isSuggested('play', card.id)
      || (pending.kind === 'press' && pending.suggestion?.includes(card.id)));
    hand.append(cardView(card, {legal, suggested, chosen: selected.includes(card.id), onClick: legal ? () => {
      if (pending.kind === 'press') {
        selected = selected.includes(card.id) ? selected.filter(id => id !== card.id) : selected.length < 2 ? [...selected, card.id] : selected;
        render();
      } else submit(card.id);
    } : undefined}));
  }
  if (!player.hand.length) for (let i = 0; i < player.handCount; i++) hand.append(cardView(null, {compact: true}));
  node.append(hand); return node;
}
function renderTrick(trick, compact = false) {
  const node = el('div', compact ? 'history-cards' : 'trick-cards');
  for (const turn of trick?.cards || []) {
    const winner = trick.cards.length === 3 && turn.seat === trick.winner;
    const item = el('div', winner ? 'turn trick-winner' : 'turn');
    const card = cardView(turn.card, {compact});
    if (winner) {
      card.title = `${cardLabel(turn.card)} · ${state.config.seats[turn.seat].label} · Stich gewonnen`;
      card.setAttribute('aria-label', card.title);
    }
    const name = el('small', '', state.config.seats[turn.seat].label);
    if (compact) {
      name.title = state.config.seats[turn.seat].label;
      item.append(name, card);
    } else item.append(card, name);
    node.append(item);
  }
  return node;
}
function renderActions() {
  const node = $('actions'); node.replaceChildren();
  const pending = state.pending;
  if (!pending) {
    const suggesting = state.active && manualSeat(state.config.seats[state.active.seat]);
    node.append(el('span', 'muted', terminal.has(state.status) ? 'Runde beendet.' : suggesting ? 'Agentenvorschlag wird berechnet …' : 'Die automatischen Spieler sind am Zug.'));
    return;
  }
  node.append(el('strong', '', `${state.config.seats[pending.seat].label}: `));
  if (Object.hasOwn(pending, 'suggestion')) {
    const suggestion = el('div', 'agent-suggestion');
    suggestion.append(el('strong', '', 'Agentenvorschlag: '), el('span', '', actionLabel(pending.kind, pending.suggestion)));
    const accept = button('Vorschlag übernehmen', () => submit(pending.suggestion), 'primary');
    accept.disabled = busy; suggestion.append(accept); node.append(suggestion);
  }
  const add = (label, value, className = '') => {
    const suggested = isSuggested(pending.kind, value);
    const b = button(label, () => submit(value), `${className}${suggested ? ' suggested-action' : ''}`);
    if (suggested) { b.title = 'Agentenvorschlag'; b.setAttribute('aria-label', `${label} · Agentenvorschlag`); }
    b.disabled = busy; node.append(b); return b;
  };
  if (pending.kind === 'play') {
    node.append(el('span', '', busy ? 'Zug wird übernommen …' : 'Eine markierte Karte wählen.'));
    const extra = el('details', 'more-actions'); extra.append(el('summary', '', 'Weitere Aktionen'));
    for (const option of pending.options.filter(option => typeof option === 'string')) {
      const b = button(specialActions[option], () => submit(option), isSuggested('play', option) ? 'suggested-action' : ''); b.disabled = busy; extra.append(b);
    }
    node.append(extra);
  } else if (pending.kind === 'press') {
    node.append(el('span', '', `Zwei Karten drücken (${selected.length}/2).`));
    add('Drücken', selected, 'primary').disabled = busy || selected.length !== 2;
  } else if (pending.kind === 'bid') {
    if (pending.offer !== null) { node.append(el('span', '', `${pending.offer} halten?`)); add('Halten', pending.offer, 'primary'); }
    else {
      const bids = pending.options.filter(value => value !== 0);
      const choices = select('bid', bids.map(value => [String(value), `${value}${isSuggested('bid', value) ? ' · Agentenvorschlag' : ''}`]), selectedOption ?? String(bids[0]));
      choices.addEventListener('change', event => { selectedOption = event.target.value; });
      choices.disabled = busy; node.append(choices);
      const bid = button('Reizen', () => submit(Number(choices.value)), 'primary'); bid.disabled = busy || !bids.length; node.append(bid);
    }
    add('Passen', 0);
  } else if (pending.kind === 'pickup') { add('Skat aufnehmen', true, 'primary'); add('Hand spielen', false); }
  else if (pending.kind === 'declare') {
    const choices = select('declaration', pending.options.map((d, i) => [String(i), `${declarationName(d)}${isSuggested('declare', d) ? ' · Agentenvorschlag' : ''}`]), selectedOption ?? '0');
    choices.addEventListener('change', event => { selectedOption = event.target.value; });
    choices.disabled = busy; node.append(choices);
    const declare = button('Ansagen', () => submit(pending.options[Number(choices.value)]), 'primary'); declare.disabled = busy; node.append(declare);
  } else {
    node.append(el('span', '', pending.kind === 'confirm_claim' ? `Abkürzung des Alleinspielers annehmen (Stufe ${pending.level})?` : 'Aufgabe des Alleinspielers annehmen?'));
    add('Annehmen', true); add('Weiterspielen', false);
  }
}
function renderComparisons() {
  const panel = $('comparison-panel');
  const stats = state.comparisonStats || [];
  panel.hidden = !stats.length;
  const summary = $('comparison-stats'); summary.replaceChildren();
  const rate = counter => `${counter.matches}/${counter.total} · ${counter.total ? (100 * counter.matches / counter.total).toLocaleString('de-DE', {maximumFractionDigits: 1}) + ' %' : '–'}`;
  for (const seat of stats) {
    const block = el('div', 'comparison-seat');
    block.append(el('h2', '', state.config.seats[seat.seat].label));
    const table = el('table'), body = el('tbody');
    for (const [name, counter] of [['Gesamt', seat], ...Object.entries(decisionNames).map(([kind, name]) => [name, seat.byKind[kind] || {matches: 0, total: 0}])]) {
      const row = el('tr'); row.append(el('th', '', name), el('td', '', rate(counter))); body.append(row);
    }
    table.append(body); block.append(table); summary.append(block);
  }
  const log = $('comparison-history'); log.replaceChildren();
  const table = el('table'), head = el('tr'), thead = el('thead'), body = el('tbody');
  for (const name of ['Runde', 'Spieler', 'Entscheidung', 'Agentenvorschlag', 'Deine Wahl', 'Vergleich']) head.append(el('th', '', name));
  thead.append(head); table.append(thead);
  for (const entry of state.comparisons || []) {
    const row = el('tr');
    for (const text of [entry.roundIndex + 1, state.config.seats[entry.seat].label, decisionNames[entry.kind], actionLabel(entry.kind, entry.suggestion), actionLabel(entry.kind, entry.actual), entry.matches ? 'Übereinstimmung' : 'Abweichung']) row.append(el('td', '', text));
    body.append(row);
  }
  table.append(body); log.append(table);
  if (!state.comparisons?.length) log.append(el('p', 'muted', 'Noch keine Entscheidungen verglichen.'));
}
function renderScores() {
  const table = $('scores'); table.replaceChildren();
  const head = el('tr'); head.append(el('th', '', 'Runde'));
  for (const seat of state.config.seats) head.append(el('th', '', seat.label)); head.append(el('th', '', 'Ergebnis'));
  const thead = el('thead'); thead.append(head); table.append(thead);
  const body = el('tbody');
  for (const record of state.records) {
    const row = el('tr'); row.append(el('td', '', record.round));
    for (let i = 0; i < 3; i++) row.append(el('td', '', record.result.soloist === i ? record.result.score : '–'));
    row.append(el('td', '', record.result.soloist_won === null ? 'Eingepasst' : `${declarationName(record.result.declaration)} · ${record.result.soloist_won ? 'gewonnen' : 'verloren'}`)); row.lastElementChild.append(dealTools(record.dealCode)); body.append(row);
  }
  table.append(body); const foot = el('tfoot'), total = el('tr'); total.append(el('th', '', 'Summe'));
  for (const score of state.scores) total.append(el('th', '', score)); total.append(el('th')); foot.append(total); table.append(foot);
}
function render() {
  if (!state) return;
  $('setup').hidden = true; $('game').hidden = false;
  if (pendingId !== state.pending?.id) { pendingId = state.pending?.id; selected = []; selectedOption = null; busy = false; }
  $('deal-tools').replaceChildren(dealTools(state.dealCode));
  $('round-title').textContent = `Runde ${state.roundIndex + 1}`;
  $('contract').replaceChildren(
    el(state.declaration ? 'strong' : 'span', state.declaration ? 'contract-name' : 'muted', declarationName(state.declaration)),
    el('span', 'muted', ` · Reizwert ${state.highestBid || 0}`));
  $('pause').textContent = state.paused ? 'Weiter' : 'Pause';
  $('pause').disabled = $('step').disabled = terminal.has(state.status);
  $('speed').value = state.speedMs; $('speed-label').value = `${state.speedMs} ms`; $('reveal').checked = state.reveal;
  const bottomSeat = state.config.seats.findIndex(manualSeat);
  const bottom = bottomSeat < 0 ? 0 : bottomSeat;
  $('players-top').replaceChildren(...(state.players || []).filter(p => p.seat !== bottom).map(playerView));
  $('player-bottom').replaceChildren(...(state.players || []).filter(p => p.seat === bottom).map(playerView));
  const latest = state.history?.at(-1);
  const current = state.currentTrick?.cards.length ? state.currentTrick : latest;
  $('trick').replaceChildren(el('small', 'muted', state.currentTrick?.cards.length ? 'Aktueller Stich' : latest ? 'Letzter Stich' : ''), renderTrick(current));
  $('skat').replaceChildren();
  if (state.skat?.length) { $('skat').append(el('small', 'muted', 'Skat')); for (const card of state.skat) $('skat').append(cardView(card, {compact: true})); }
  $('status').textContent = state.error || (state.status === 'loading' ? 'Runde wird vorbereitet …' : state.status === 'stopped' ? 'Partie beendet' : state.pending ? `${state.config.seats[state.pending.seat].label} ist am Zug` : terminal.has(state.status) ? 'Runde beendet' : state.active ? `${state.config.seats[state.active.seat].label} ${kindNames[state.active.kind]} …${state.paused && !manualSeat(state.config.seats[state.active.seat]) ? ' (pausiert)' : ''}` : 'Runde läuft …');
  $('history').replaceChildren();
  if (!state.history?.length) $('history').append(el('p', 'muted', 'Noch keine Stiche.'));
  for (const [index, trick] of (state.history || []).entries()) {
    const row = el('div', 'history-row');
    const summary = el('small', '', `${index + 1}. · `);
    summary.append(el('strong', '', state.config.seats[trick.winner].label));
    if (state.declaration?.kind !== 'null') summary.append(` · ${trick.value} Augen`);
    row.append(summary, renderTrick(trick, true)); $('history').append(row);
  }
  renderActions(); renderScores(); renderComparisons();
  const result = $('result'); result.hidden = !terminal.has(state.status); result.replaceChildren();
  if (!result.hidden) {
    const r = state.result;
    result.append(el('h2', '', state.error ? 'Fehler' : r?.soloist_won == null ? 'Eingepasst' : r.soloist_won ? 'Alleinspieler gewinnt' : 'Gegenpartei gewinnt'));
    if (r?.soloist != null) {
      const tally = r.declaration.kind === 'null' ? `${r.tricks[0]}:${r.tricks[1]} Stiche` : `${r.points[0]}:${r.points[1]} Augen`;
      result.append(el('p', '', `${state.config.seats[r.soloist].label}: ${r.score} Punkte · Spielwert ${r.game_value} · ${tally}${r.reason === 'overbid' ? ' · Überreizt' : ''}`));
    }
    if (state.error) result.append(el('p', 'error-text', state.error));
    if (['finished', 'passed'].includes(state.status)) {
      const next = button('Nächste Runde', async () => {
        next.disabled = true;
        try { await control('next'); } catch { next.disabled = false; }
      }, 'primary'); result.append(next);
    }
  }
}
async function submit(value) {
  if (busy || !state.pending) return;
  const id = state.id, decisionId = state.pending.id;
  busy = true; clearError(); render();
  try { await api(`/api/sessions/${id}/actions`, 'POST', {decisionId, value}); }
  catch (error) {
    if (state?.id === id && state.pending?.id === decisionId) { busy = false; render(); }
    showError(error);
  }
}
async function control(command, value) {
  try { await api(`/api/sessions/${state.id}/control`, 'POST', {command, value, roundIndex: state.roundIndex}); }
  catch (error) { showError(error); throw error; }
}
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function poll(id, token) {
  while (generation === token) {
    try {
      const next = await api(`/api/sessions/${id}?after=${state?.revision ?? -1}`);
      if (generation !== token) return;
      state = next; $('connection').textContent = 'Verbunden · lokal'; render();
      if (state.status === 'error' || state.status === 'stopped') return;
    } catch (error) {
      if (generation !== token) return;
      if (error.status === 404) { save('skat.session', null); state = null; $('game').hidden = true; $('setup').hidden = false; showError(new Error('Die Partie ist nicht mehr verfügbar. Eine neue Partie starten.')); return; }
      $('connection').textContent = 'Verbindung unterbrochen · erneuter Versuch …'; await delay(1500);
    }
  }
}
$('setup-form').addEventListener('submit', async event => {
  event.preventDefault(); clearError();
  const start = event.submitter; start.disabled = true;
  try {
    const config = getConfig(); state = await api('/api/sessions', 'POST', config);
    save('skat.config', JSON.stringify(config)); save('skat.session', state.id);
    render(); poll(state.id, ++generation);
  } catch (error) { showError(error); } finally { start.disabled = false; }
});
$('exit').addEventListener('click', async () => {
  const id = state?.id;
  if (id) { try { await api(`/api/sessions/${id}`, 'DELETE'); } catch (error) { if (error.status !== 404) { showError(error); return; } } }
  generation++; state = null; busy = false; selected = []; selectedOption = null; pendingId = null;
  save('skat.session', null); $('deal-code').value = ''; updateDealInput(); $('game').hidden = true; $('setup').hidden = false; clearError(); $('connection').textContent = 'Lokal';
});
for (const [id, command] of [['pause', () => state.paused ? 'resume' : 'pause'], ['step', () => 'step']]) $(id).addEventListener('click', () => control(command()).catch(() => {}));
$('speed').addEventListener('change', event => control('speed', Number(event.target.value)).catch(() => {}));
$('speed').addEventListener('input', event => { $('speed-label').value = `${event.target.value} ms`; });
$('reveal').addEventListener('change', event => control('reveal', event.target.checked).catch(() => {}));
function deckNote() { $('deck-note').textContent = appConfig?.decks[deck] ? '' : 'Kartenbilder fehlen lokal; Ersatzdarstellung aktiv.'; }
$('deck').value = deck;
$('deck').addEventListener('change', event => { deck = event.target.value; save('skat.deck', deck); deckNote(); render(); });
async function init() {
  try {
    appConfig = await api('/api/config');
    for (const path of appConfig.checkpoints) { const option = el('option'); option.value = path; $('checkpoints').append(option); }
    setupSeats(); deckNote();
    const shared = new URLSearchParams(location.hash.slice(1)).get('deal');
    if (shared !== null) {
      $('deal-code').value = shared; updateDealInput();
      history.replaceState(null, '', location.pathname + location.search);
      return;
    }
    const id = read('skat.session');
    if (id) {
      try { state = await api(`/api/sessions/${id}`); render(); poll(id, ++generation); }
      catch (error) { if (error.status === 404) save('skat.session', null); else throw error; }
    }
  } catch (error) { showError(error); }
}
init();
