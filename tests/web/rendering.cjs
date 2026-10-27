// Exercise the real renderer with a minimal DOM, without a browser dependency.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
  constructor(tagName = 'div') {
    this.tagName = tagName.toUpperCase();
    this.value = '';
    this.children = [];
    this.listeners = {};
    this.attributes = {};
    this.classList = {add() {}};
  }
  set textContent(value) { this.children = [String(value)]; }
  get textContent() {
    return this.children.map(child => typeof child === 'string' ? child : child.textContent).join('');
  }
  get lastElementChild() { return this.children.at(-1); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  addEventListener(type, listener) { (this.listeners[type] ??= []).push(listener); }
  dispatchEvent(event) { for (const listener of this.listeners[event.type] || []) listener(event); }
  querySelectorAll(selector) {
    const matches = node => selector === '[name]' ? Boolean(node.name)
      : selector.split(',').some(tag => node.tagName === tag.trim().toUpperCase());
    return this.children.filter(child => typeof child !== 'string')
      .flatMap(child => [...(matches(child) ? [child] : []), ...child.querySelectorAll(selector)]);
  }
  setAttribute(name, value) { this.attributes[name] = value; }
}

const elements = new Map();
const storage = new Map();
const requests = [];
const context = vm.createContext({
  Event: class { constructor(type) { this.type = type; } },
  localStorage: {
    getItem(key) { return storage.get(key) ?? null; },
    setItem(key, value) { storage.set(key, value); },
    removeItem(key) { storage.delete(key); },
  },
  document: {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, new Element());
      return elements.get(id);
    },
    createElement(tagName) { return new Element(tagName); },
  },
  // Leave initialization pending; rendering itself requires no HTTP calls.
  fetch: (...args) => { requests.push(args); return new Promise(() => {}); },
});
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), context);
const state = JSON.parse(fs.readFileSync(0, 'utf8'));
vm.runInContext(`state = ${JSON.stringify(state)}; render();`, context);
const player = elements.get('player-bottom').textContent;
const result = elements.get('result').textContent;
const history = elements.get('history').textContent;
if (state.declaration.kind === 'null') {
  assert(player.includes('0 Stiche'));
  assert(result.includes('0:10 Stiche'));
  assert(![player, result, history, elements.get('players-top').textContent].some(text => text.includes('Augen')));
} else {
  assert(player.includes('14 Augen · 0 Stiche'));
  assert(result.includes('14:106 Augen'));
  assert(history.includes('10 Augen'));
}
assert(result.includes(`${state.result.score} Punkte · Spielwert ${state.result.game_value}`));

// Check the displayed hand order for every contract, including before bidding.
// Start with the full pack in a different order from every expected display.
const cards = [...'789JQKXA'].flatMap((rank, rankIndex) =>
  [...'♥♦♠♣'].map(suit => ({id: '♦♥♠♣'.indexOf(suit) * 8 + rankIndex, suit, rank})));
const suits = [...'♣♠♦♥'];
const colorCards = (suit, ranks) => [...ranks].map(rank => `${suit}${rank}`);
const jacks = ['♣J', '♠J', '♦J', '♥J'];
const cases = [
  [null, [...jacks, ...suits.flatMap(suit => colorCards(suit, 'AXKQ987'))]],
  [{kind: 'grand'}, [...jacks, ...suits.flatMap(suit => colorCards(suit, 'AXKQ987'))]],
  [{kind: 'null'}, suits.flatMap(suit => colorCards(suit, 'AKQJX987'))],
  [{kind: 'null', ouvert: true}, suits.flatMap(suit => colorCards(suit, 'AKQJX987'))],
  ...[...'♦♥♠♣'].map((trump, suit) => [
    {kind: 'suit', suit},
    [...jacks, ...colorCards(trump, 'AXKQ987'),
      ...suits.filter(color => color !== trump).flatMap(color => colorCards(color, 'AXKQ987'))],
  ]),
];
// The reported deal after pickup, then after pressing its two singleton suits.
const pickedUp = new Set('♦7 ♦8 ♦Q ♦K ♦X ♥7 ♥8 ♥9 ♥J ♣X ♦9 ♠K'.split(' '));
const twelve = cards.filter(card => pickedUp.has(card.suit + card.rank));
const ten = twelve.filter(card => !['♣X', '♠K'].includes(card.suit + card.rank));
for (const [declaration, expected] of cases) {
  for (const hand of [cards, twelve, ten]) {
    const input = JSON.stringify(hand);
    const held = new Set(hand.map(card => card.suit + card.rank));
    vm.runInContext(`state.declaration = ${JSON.stringify(declaration)}; state.players[0].hand = ${input};`, context);
    const playerView = vm.runInContext('playerView(state.players[0])', context);
    const displayed = playerView.lastElementChild.children.map(card => card.title);
    const expectedLabels = expected.filter(name => held.has(name)).map(name => vm.runInContext(
      `cardLabel(${JSON.stringify({suit: name[0], rank: name[1]})})`, context));
    assert.deepEqual(displayed, expectedLabels, JSON.stringify(declaration));
    assert.equal(vm.runInContext('JSON.stringify(state.players[0].hand)', context), input);
  }
}

// Presets fill a single seat, update field visibility, and survive a reload.
vm.runInContext(`appConfig = {checkpoints: [
  'benchmark_runs/iss-run-1/policy.pt', 'benchmark_runs/bidding-run-1/bidding.pt',
  'benchmark_runs/declaration-run-3/declaration.pt', 'benchmark_runs/pickup-run-2/pickup.pt'
]}; setupSeats();`, context);
const seats = elements.get('seats').children;
const controls = seat => Object.fromEntries(seat.querySelectorAll('[name]').map(node => [node.name, node]));
const controlsBefore = seats.map(seat => JSON.stringify(controls(seat), (key, value) => key === 'listeners' ? undefined : value));
const firstBot = seats[1];
const buttons = firstBot.querySelectorAll('button');
assert.deepEqual(buttons.map(button => button.textContent), ['Beste Policy gegen Random', 'Alles neuronal']);
assert(buttons.every(button => !button.disabled));
for (const button of [buttons[1], buttons[0]]) {
  const allNeural = button === buttons[1];
  button.dispatchEvent({type: 'click'});
  const fields = controls(firstBot);
  assert.equal(fields.label.value, 'Bot 1');
  assert.equal(fields.kind.value, 'policy');
  assert.equal(fields.policy.value, 'benchmark_runs/iss-run-1/policy.pt');
  assert.equal(fields.policy.required, true);
  assert.equal(fields.bidding.value, 'neural');
  assert.equal(fields.biddingModel.value, 'benchmark_runs/bidding-run-1/bidding.pt');
  assert.equal(fields.biddingThreshold.value, '0.55');
  assert.equal(fields.declaration.value, allNeural ? 'neural' : 'heuristic');
  assert.equal(fields.declarationModel.value, allNeural ? 'benchmark_runs/declaration-run-3/declaration.pt' : '');
  assert.equal(fields.pickup.value, allNeural ? 'neural' : 'always');
  assert.equal(fields.pickupModel.value, allNeural ? 'benchmark_runs/pickup-run-2/pickup.pt' : '');
  assert.equal(fields.pickupMargin.value, '0');
  assert.equal(fields.beliefModel.value, '');
  assert(Object.values(fields).every(field => !field.disabled));
  const modelLabels = firstBot.querySelectorAll('label').filter(label => label.children.includes(fields.declarationModel) || label.children.includes(fields.pickupModel));
  assert(modelLabels.every(label => label.hidden === !allNeural));
  const config = vm.runInContext('getConfig()', context);
  assert.equal(config.seats[1].biddingThreshold, 0.55);
  assert.equal(config.seats[1].pickupMargin, 0);
  assert.equal(config.seats[1].declaration, allNeural ? 'neural' : 'heuristic');
}
for (const i of [0, 2]) {
  assert.equal(JSON.stringify(controls(seats[i]), (key, value) => key === 'listeners' ? undefined : value), controlsBefore[i]);
}
seats[0].querySelectorAll('button')[1].dispatchEvent({type: 'click'});
assert.equal(controls(seats[0]).kind.value, 'policy');
assert.equal(controls(seats[0]).label.value, 'Du');
assert(Object.values(controls(seats[0])).every(field => !field.disabled));
vm.runInContext("save('skat.config', JSON.stringify(getConfig())); setupSeats();", context);
assert.equal(controls(elements.get('seats').children[1]).policy.value, 'benchmark_runs/iss-run-1/policy.pt');
assert.equal(controls(elements.get('seats').children[0]).declaration.value, 'neural');
vm.runInContext('appConfig.checkpoints = []; setupSeats();', context);
assert(elements.get('seats').children.every(seat => seat.querySelectorAll('button').every(button => button.disabled)));

// Manual execution is independent of agent presets and survives saved settings.
vm.runInContext("appConfig.checkpoints = ['benchmark_runs/iss-run-1/policy.pt', 'benchmark_runs/bidding-run-1/bidding.pt', 'benchmark_runs/declaration-run-3/declaration.pt', 'benchmark_runs/pickup-run-2/pickup.pt']; setupSeats();", context);
const manualSeat = elements.get('seats').children[1];
controls(manualSeat).execution.value = 'manual';
manualSeat.querySelectorAll('button')[0].dispatchEvent({type: 'click'});
assert.equal(controls(manualSeat).execution.value, 'manual');
controls(manualSeat).kind.value = 'human';
controls(manualSeat).kind.dispatchEvent({type: 'change'});
assert(controls(manualSeat).execution.disabled);
assert.equal(vm.runInContext('getConfig().seats[1].execution', context), 'manual');
controls(manualSeat).kind.value = 'mcts';
controls(manualSeat).kind.dispatchEvent({type: 'change'});
assert(!controls(manualSeat).execution.disabled);
vm.runInContext("save('skat.config', JSON.stringify(getConfig())); setupSeats();", context);
assert.equal(controls(elements.get('seats').children[1]).execution.value, 'manual');

// Suggestions label every action, remain separate from selection, and submit
// the cached recommendation with the current decision ID.
const declaration = {kind: 'grand', suit: null, hand: false, ouvert: false, schneider_announced: false, schwarz_announced: false};
const pendingCases = [
  {kind: 'bid', options: [0, 18, 20], current: 0, offer: null, suggestion: 20},
  {kind: 'bid', options: [0, 18], current: 0, offer: 18, suggestion: 0},
  {kind: 'pickup', options: [true, false], suggestion: false},
  {kind: 'press', options: [0, 1, 2], suggestion: [1, 2]},
  {kind: 'declare', options: [{...declaration, kind: 'null'}, declaration], suggestion: Object.fromEntries(Object.entries(declaration).reverse())},
  {kind: 'play', options: [0, 1, 2, 'concede'], suggestion: 1},
  {kind: 'confirm_concession', options: [true, false], suggestion: false},
  {kind: 'confirm_claim', options: [true, false], level: 1, suggestion: true},
];
vm.runInContext(`appConfig.decks = {}; state.id = 'manual-session'; state.status = 'running'; state.result = null;
  state.config.seats = [{kind: 'random', execution: 'auto', label: 'Bot'}, {kind: 'policy', execution: 'manual', label: 'Du'}, {kind: 'random', execution: 'auto', label: 'Bot 2'}];
  state.players.forEach((player, i) => { player.label = state.config.seats[i].label; player.kind = state.config.seats[i].kind; });
  state.players[1].hand = ${JSON.stringify(cards.filter(card => card.id < 3))};`, context);
for (const [i, pending] of pendingCases.entries()) {
  vm.runInContext(`state.pending = ${JSON.stringify({...pending, id: `decision-${i}`, seat: 1})}; render();`, context);
  const actions = elements.get('actions');
  assert(actions.textContent.includes('Agentenvorschlag: '));
  assert(elements.get('player-bottom').textContent.includes('Du'));
  assert.equal(vm.runInContext('selected.length', context), 0);
  const accept = actions.querySelectorAll('button').find(button => button.textContent === 'Vorschlag übernehmen');
  assert(accept && !accept.disabled);
  if (pending.kind === 'press') {
    const handCards = elements.get('player-bottom').querySelectorAll('button');
    assert.equal(handCards.filter(card => card.className.includes('suggested')).length, 2);
    assert(handCards.filter(card => card.className.includes('suggested')).every(card => card.attributes['aria-label'].includes('Agentenvorschlag')));
    assert(!handCards.some(card => card.className.includes('chosen')));
    assert(actions.querySelectorAll('button').find(button => button.textContent === 'Drücken').disabled);
  }
  if (pending.kind === 'bid' && pending.offer === null) {
    const choices = actions.querySelectorAll('select')[0];
    assert.equal(choices.value, '18');
    assert(choices.children.some(option => option.value === '20' && option.textContent.includes('Agentenvorschlag')));
  }
  if (pending.kind === 'declare') {
    const choices = actions.querySelectorAll('select')[0];
    assert.equal(choices.value, '0');
    assert(choices.children[1].textContent.includes('Agentenvorschlag'));
  }
  const before = requests.length;
  accept.dispatchEvent({type: 'click'});
  assert.equal(requests.length, before + 1);
  const [url, request] = requests.at(-1);
  assert.equal(url, '/api/sessions/manual-session/actions');
  assert.deepEqual(JSON.parse(request.body), {decisionId: `decision-${i}`, value: pending.suggestion});
  assert(elements.get('actions').querySelectorAll('button').every(button => button.disabled));
}

// Polling/control updates must preserve a chosen declaration or bid until the
// decision changes. In particular Grand must not silently become default Karo.
for (const pending of [
  {id: 'keep-grand', kind: 'declare', options: [{...declaration, kind: 'suit', suit: 0}, declaration], suggestion: {...declaration, kind: 'suit', suit: 0}},
  {id: 'keep-bid', kind: 'bid', options: [0, 18, 20], current: 0, offer: null, suggestion: 18},
]) {
  vm.runInContext(`state.pending = ${JSON.stringify({...pending, seat: 1})}; render();`, context);
  const choice = elements.get('actions').querySelectorAll('select')[0];
  const selectedValue = pending.kind === 'declare' ? '1' : '20';
  choice.value = selectedValue;
  choice.dispatchEvent({type: 'change', target: choice});
  vm.runInContext('state = JSON.parse(JSON.stringify(state)); render();', context);
  assert.equal(elements.get('actions').querySelectorAll('select')[0].value, selectedValue);
  const confirm = elements.get('actions').querySelectorAll('button').find(button => button.textContent === (pending.kind === 'declare' ? 'Ansagen' : 'Reizen'));
  confirm.dispatchEvent({type: 'click'});
  assert.deepEqual(JSON.parse(requests.at(-1)[1].body), {decisionId: pending.id, value: pending.kind === 'declare' ? declaration : 20});
  vm.runInContext(`state.pending.id += '-next'; render();`, context);
  assert.equal(elements.get('actions').querySelectorAll('select')[0].value, pending.kind === 'declare' ? '0' : '18');
}

// A person's own card click submits that card rather than the recommendation.
vm.runInContext(`state.pending = {id: 'own-choice', seat: 1, kind: 'play', options: [0, 1, 2], suggestion: 1}; render();`, context);
const ownCard = elements.get('player-bottom').querySelectorAll('button').find(card => card.title === 'Karo 7');
assert(ownCard);
ownCard.dispatchEvent({type: 'click'});
assert.deepEqual(JSON.parse(requests.at(-1)[1].body), {decisionId: 'own-choice', value: 0});

// Comparisons retain seat labels, action names, counts, and empty denominators.
vm.runInContext(`state.comparisonStats = [{seat: 1, matches: 1, total: 2, byKind: {play: {matches: 1, total: 2}}}];
  state.comparisons = [
    {roundIndex: 0, seat: 1, kind: 'play', suggestion: 1, actual: 0, matches: false},
    {roundIndex: 1, seat: 1, kind: 'play', suggestion: 2, actual: 2, matches: true}
  ]; renderComparisons();`, context);
assert.equal(elements.get('comparison-panel').hidden, false);
assert(elements.get('comparison-stats').textContent.includes('Du'));
assert(elements.get('comparison-stats').textContent.includes('1/2 · 50 %'));
assert(elements.get('comparison-stats').textContent.includes('0/0 · –'));
const comparisonHistory = elements.get('comparison-history').textContent;
assert(comparisonHistory.includes('Abweichung') && comparisonHistory.includes('Übereinstimmung'));
assert(comparisonHistory.includes('Karo 8') && comparisonHistory.includes('Karo 7'));
vm.runInContext('state.comparisonStats = []; renderComparisons();', context);
assert.equal(elements.get('comparison-panel').hidden, true);
