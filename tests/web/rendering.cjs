// Exercise the real renderer with a minimal DOM, without a browser dependency.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
  constructor() {
    this.children = [];
    this.classList = {add() {}};
  }
  set textContent(value) { this.children = [String(value)]; }
  get textContent() {
    return this.children.map(child => typeof child === 'string' ? child : child.textContent).join('');
  }
  get lastElementChild() { return this.children.at(-1); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  addEventListener() {}
  setAttribute() {}
}

const elements = new Map();
const context = vm.createContext({
  document: {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, new Element());
      return elements.get(id);
    },
    createElement() { return new Element(); },
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
