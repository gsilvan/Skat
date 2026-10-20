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
  setAttribute() {}
}

const elements = new Map();
const storage = new Map();
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
  fetch: () => new Promise(() => {}),
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
