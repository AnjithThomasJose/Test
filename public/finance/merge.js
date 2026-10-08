// Merges two finance states so devices never overwrite each other's work.
// Entries and categories merge by id (newest `u` wins; deletes are tombstones).
// Settings and category order come from whichever side changed them last.
(function (root) {
  function stamp(x) { return x.u || x.ts || 0; }

  // Tie-break: the doc with the later overall updatedAt wins (a fresh device has 0).
  function pickSide(localU, remoteU, local, remote) {
    if (remoteU !== localU) return remoteU > localU ? remote : local;
    return (remote.updatedAt || 0) > (local.updatedAt || 0) ? remote : local;
  }

  function mergeById(localList, remoteList) {
    const byId = {};
    (localList || []).forEach(function (x) { byId[x.id] = x; });
    (remoteList || []).forEach(function (x) {
      const mine = byId[x.id];
      if (!mine || stamp(x) > stamp(mine)) byId[x.id] = x;
    });
    return byId;
  }

  function mergeState(local, remote) {
    if (!remote) return local;

    const entries = mergeById(local.entries, remote.entries);

    const cats = mergeById(local.categories, remote.categories);
    const orderSrc = pickSide(local.catsU || 0, remote.catsU || 0, local, remote);
    const order = orderSrc.categories.map(function (c) { return c.id; });
    Object.keys(cats).forEach(function (id) { if (order.indexOf(id) < 0) order.push(id); });

    const settingsSrc = pickSide((local.settings || {}).u || 0, (remote.settings || {}).u || 0, local, remote);

    return {
      updatedAt: Math.max(local.updatedAt || 0, remote.updatedAt || 0),
      settings: settingsSrc.settings,
      catsU: Math.max(local.catsU || 0, remote.catsU || 0),
      categories: order.map(function (id) { return cats[id]; }),
      entries: Object.keys(entries).map(function (id) { return entries[id]; })
    };
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = { mergeState: mergeState };
  else root.mergeState = mergeState;
})(this);
