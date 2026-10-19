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
