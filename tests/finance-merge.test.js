// Run: node tests/finance-merge.test.js
const assert = require('assert');
const { mergeState } = require('../public/finance/merge.js');

const cat = (id, u) => ({ id, emoji: '•', name: id, color: '#fff', deleted: false, u });
const base = (over) => Object.assign({
  updatedAt: 1, settings: { budget: 0, cycleDay: 1 }, catsU: 0,
  categories: [cat('food'), cat('bills')], entries: []
}, over);

// No remote doc yet: local passes through.
const solo = base({ entries: [{ id: 'a', amount: 100, ts: 5 }] });
assert.strictEqual(mergeState(solo, null), solo);

// Entries from both devices survive.
let m = mergeState(
  base({ entries: [{ id: 'a', amount: 100, ts: 5 }] }),
  base({ entries: [{ id: 'b', amount: 200, ts: 6 }] })
);
assert.deepStrictEqual(m.entries.map(e => e.id).sort(), ['a', 'b']);

// Newer edit wins regardless of side.
m = mergeState(
  base({ entries: [{ id: 'a', amount: 100, ts: 5 }] }),
  base({ entries: [{ id: 'a', amount: 999, ts: 5, u: 50 }] })
);
assert.strictEqual(m.entries[0].amount, 999);

// Delete tombstone beats an older live copy, so it does not come back.
m = mergeState(
  base({ entries: [{ id: 'a', amount: 100, ts: 5 }] }),
  base({ entries: [{ id: 'a', deleted: true, u: 60 }] })
);
assert.strictEqual(m.entries[0].deleted, true);

// Fresh device (updatedAt 0, default settings) must not clobber cloud settings.
m = mergeState(
  base({ updatedAt: 0 }),
  base({ updatedAt: 10, settings: { budget: 2500000, cycleDay: 5 } })
);
assert.strictEqual(m.settings.budget, 2500000);

// Newer settings change wins even from the older doc.
m = mergeState(
  base({ updatedAt: 1, settings: { budget: 100, cycleDay: 1, u: 99 } }),
  base({ updatedAt: 10, settings: { budget: 200, cycleDay: 1, u: 20 } })
);
assert.strictEqual(m.settings.budget, 100);

// Category order from the side that reordered last; categories added on the other side are kept.
m = mergeState(
  base({ catsU: 30, categories: [cat('bills'), cat('food')] }),
  base({ catsU: 10, categories: [cat('food'), cat('bills'), cat('pets', 10)] })
);
assert.deepStrictEqual(m.categories.map(c => c.id), ['bills', 'food', 'pets']);

console.log('finance merge: all checks passed');
