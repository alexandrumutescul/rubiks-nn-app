/* search.js: the beam search of rubiks/search.py (beam_search), in plain JavaScript.
 *
 * Pure: no DOM and no ONNX Runtime. The policy is a function the caller passes in, so the page (ONNX Runtime Web)
 * and the parity harness (search-test.html) run exactly this code. Classic script, no build step; exposes
 * window.RubiksSearch (and module.exports under Node).
 *
 *   RubiksSearch.beamSearch(policyFn, state, options) -> Promise<{moves, nodes, depth, ms}>
 *
 *   policyFn  async (states: Uint8Array[]) -> Float32Array (or array) of states.length * 18 values, row-major:
 *             the logits of the 18 moves (U U' U2 R R' R2 F F' F2 D D' D2 L L' L2 B B' B2) for each state.
 *             Called ONCE per depth with the whole beam.
 *   state     54 colours 0..5, Kociemba facelet order.
 *   options   tables      CUBE_TABLES (move_perms, solved, centres, move_face); required
 *             width       beam width, integer >= 1 (default 4096, as in search.py; the page passes its setting)
 *             maxDepth    integer >= 0 (default 40)
 *             noBacktrack default true (search.py no_backtrack)
 *             normalize   default true: log-softmax the policy's logits; false: they already are log-probabilities
 *             signal      AbortSignal: checked before and after each policy call (rejects with an AbortError)
 *             onDepth     called once per depth with {depth, maxDepth, beam, children, nodes, ms}
 *             yieldFn     awaited after each depth (the page passes a setTimeout(0) so it can paint and take clicks)
 *   result    moves: number[] (verified to solve the state) or null when maxDepth is reached;
 *             nodes: states the policy was evaluated on (search.py nodes_expanded); depth: layers expanded
 *             (depth_reached); ms: wall-clock milliseconds of the call.
 *
 * Semantics (search.py beam_search, step by step):
 *   - a solved input returns [] with 0 nodes;
 *   - per depth: every beam state is expanded by every move the canonical filter allows, in parent-major order
 *     (parent 0's moves 0..17, then parent 1's, ...): never the face of the state's last move and, after two moves
 *     on opposite faces, no move on that axis (noBacktrack); the child's score is the float32 sum of the parent's
 *     score and the move's log-probability;
 *   - every child is checked for the solved state BEFORE any truncation; among solved children the best score
 *     wins, ties to the earliest child (torch.argmax); its path is rebuilt from per-layer parent pointers;
 *   - otherwise duplicate states are removed (the best-scored copy survives, ties to the earliest child), and the
 *     `width` best states form the next beam, ordered by score (descending), ties by the state key (ascending):
 *     exactly the order of search.py's stable sorts;
 *   - beam width 1 keeps one state per depth; as at every width, all the allowed children of that state are checked
 *     for the solved state before the cut (search.py beam_search, not its greedy_solve).
 * The log-softmax is (x - max) - log(sum of exp(x - max)) in float64, rounded to float32 (search.py receives the
 * float32 output of torch.log_softmax), so a large common offset in a row of logits cancels exactly.
 */
(function (root) {
  'use strict';

  const N_FACELETS = 54;
  const N_MOVES = 18;
  const KEY_PART_DIGITS = 12;  // search.py packs the 48 non-centre stickers into two base-6 words of 24 digits;
                               // here each word is split into two exact halves of 12 digits (6^12 < 2^53)

  function now() {
    return (root.performance && typeof root.performance.now === 'function') ? root.performance.now() : Date.now();
  }

  function abortError() {
    const e = new Error('search cancelled');
    e.name = 'AbortError';
    e.code = 'cancelled';
    return e;
  }

  function checkAbort(signal) {
    if (signal && signal.aborted) throw abortError();
  }

  // ------------------------------------------------------------------------------------------------------------
  // Tables (from CUBE_TABLES), checked once per tables object
  // ------------------------------------------------------------------------------------------------------------
  const prepared = typeof WeakMap === 'function' ? new WeakMap() : null;

  function prepareTables(T) {
    if (!T || typeof T !== 'object') throw new Error('search: options.tables (CUBE_TABLES) is required');
    if (prepared && prepared.has(T)) return prepared.get(T);
    if (!Array.isArray(T.move_perms) || T.move_perms.length !== N_MOVES) {
      throw new Error('search: tables need 18 move_perms');
    }
    if (!T.solved || T.solved.length !== N_FACELETS) throw new Error('search: tables need a 54-entry solved state');
    if (!T.centres || T.centres.length !== 6) throw new Error('search: tables need the 6 centres');
    const perms = new Uint8Array(N_MOVES * N_FACELETS);
    for (let m = 0; m < N_MOVES; m++) {
      const p = T.move_perms[m];
      const seen = new Uint8Array(N_FACELETS);
      if (!p || p.length !== N_FACELETS) throw new Error('search: move_perms[' + m + '] must have 54 entries');
      for (let i = 0; i < N_FACELETS; i++) {
        const v = p[i];
        if (!Number.isInteger(v) || v < 0 || v >= N_FACELETS || seen[v]) {
          throw new Error('search: move_perms[' + m + '] is not a permutation');
        }
        seen[v] = 1;
        perms[m * N_FACELETS + i] = v;
      }
      // search.py: MOVE_FACE = arange(18) // 3; the filter below relies on it
      if (T.move_face && T.move_face[m] !== Math.floor(m / 3)) {
        throw new Error('search: move_face[' + m + '] must be ' + Math.floor(m / 3));
      }
    }
    const solved = Uint8Array.from(T.solved);
    const centre = new Uint8Array(N_FACELETS);
    T.centres.forEach(function (c) { centre[c] = 1; });
    const nonCentre = [];  // search.py NON_CENTRE: the 48 non-centre facelets, ascending
    for (let i = 0; i < N_FACELETS; i++) if (!centre[i]) nonCentre.push(i);
    if (nonCentre.length !== 2 * 2 * KEY_PART_DIGITS) throw new Error('search: centres must be 6 distinct facelets');
    const pow6 = new Float64Array(KEY_PART_DIGITS);
    for (let d = 0; d < KEY_PART_DIGITS; d++) pow6[d] = Math.pow(6, d);
    const out = { perms: perms, solved: solved, nonCentre: Int32Array.from(nonCentre), pow6: pow6 };
    if (prepared) prepared.set(T, out);
    return out;
  }

  function checkState(state) {
    if (!state || state.length !== N_FACELETS) {
      throw new Error('search: state must have 54 entries, got ' + (state ? state.length : state));
    }
    const s = new Uint8Array(N_FACELETS);
    for (let i = 0; i < N_FACELETS; i++) {
      const c = state[i];
      if (!Number.isInteger(c) || c < 0 || c > 5) throw new Error('search: colours must be in 0..5');
      s[i] = c;
    }
    return s;
  }

  function isSolvedAt(buf, base, solved) {
    for (let i = 0; i < N_FACELETS; i++) if (buf[base + i] !== solved[i]) return false;
    return true;
  }

  // ------------------------------------------------------------------------------------------------------------
  // Policy helpers
  // ------------------------------------------------------------------------------------------------------------
  /** Row-wise log-softmax of rows x 18 logits, in float64: (x - max) - log(sum of exp(x - max)), the exponentials
   *  summed in move order; the result is rounded to float32. The maximum is subtracted before anything else, so a
   *  large common offset (1e6, 1e30) cancels exactly instead of swallowing the log term. A NaN or +Infinity logit, or
   *  a row of -Infinity only, gives a NaN row, which beamSearch reports as an error; a -Infinity logit among finite
   *  ones gives -Infinity (as torch.log_softmax does). */
  function logSoftmax(logits, rows) {
    const out = new Float32Array(rows * N_MOVES);
    for (let r = 0; r < rows; r++) {
      const base = r * N_MOVES;
      let mx = -Infinity;
      for (let j = 0; j < N_MOVES; j++) {
        const v = logits[base + j];
        if (v > mx || v !== v) mx = v;  // a NaN logit makes the whole row NaN (as numpy's max does)
      }
      let s = 0;
      for (let j = 0; j < N_MOVES; j++) s += Math.exp(logits[base + j] - mx);
      const logSum = Math.log(s);
      for (let j = 0; j < N_MOVES; j++) out[base + j] = (logits[base + j] - mx) - logSum;  // Float32Array rounds
    }
    return out;
  }

  /** The (k0 high, k0 low, k1 high, k1 low) parts of search.py's exact state key (state_keys): k0 and k1 are
   *  base-6 numbers over the non-centre stickers 0..23 and 24..47, digit d weighted 6^d. Comparing the four parts
   *  lexicographically orders states exactly as search.py's (k0, k1) int64 keys do. */
  function keyParts(buf, base, tb, out, o) {
    const nc = tb.nonCentre;
    const pw = tb.pow6;
    let k0lo = 0, k0hi = 0, k1lo = 0, k1hi = 0;
    for (let d = 0; d < KEY_PART_DIGITS; d++) {
      k0lo += buf[base + nc[d]] * pw[d];
      k0hi += buf[base + nc[d + 12]] * pw[d];
      k1lo += buf[base + nc[d + 24]] * pw[d];
      k1hi += buf[base + nc[d + 36]] * pw[d];
    }
    out[o] = k0hi;
    out[o + 1] = k0lo;
    out[o + 2] = k1hi;
    out[o + 3] = k1lo;
  }

  // ------------------------------------------------------------------------------------------------------------
  // The search
  // ------------------------------------------------------------------------------------------------------------
  async function beamSearch(policyFn, state, options) {
    const opts = options || {};
    const t0 = now();
    const tb = prepareTables(opts.tables);
    const s0 = checkState(state);
    const width = opts.width === undefined ? 4096 : opts.width;
    const maxDepth = opts.maxDepth === undefined ? 40 : opts.maxDepth;
    if (!Number.isInteger(width) || width < 1 || !Number.isInteger(maxDepth) || maxDepth < 0) {
      throw new RangeError('search: need an integer width >= 1 and maxDepth >= 0');
    }
    if (typeof policyFn !== 'function') throw new TypeError('search: policyFn must be a function');
    const noBacktrack = opts.noBacktrack !== false;
    const normalize = opts.normalize !== false;
    const signal = opts.signal || null;
    const onDepth = typeof opts.onDepth === 'function' ? opts.onDepth : null;
    const yieldFn = typeof opts.yieldFn === 'function' ? opts.yieldFn : null;
    const perms = tb.perms;
    const solved = tb.solved;

    if (isSolvedAt(s0, 0, solved)) return { moves: [], nodes: 0, depth: 0, ms: now() - t0 };

    let beam = Uint8Array.from(s0);              // B x 54
    let score = new Float32Array(1);             // cumulative log-probability (float32, as in search.py)
    let lastFace = Int8Array.of(-1);
    let lockAxis = Int8Array.of(-1);
    const histParent = [];                       // per kept layer: index into the previous beam
    const histMove = [];
    let nodes = 0;

    for (let depth = 1; depth <= maxDepth; depth++) {
      checkAbort(signal);
      const b = beam.length / N_FACELETS;
      nodes += b;
      const batch = new Array(b);
      for (let i = 0; i < b; i++) batch[i] = beam.subarray(i * N_FACELETS, (i + 1) * N_FACELETS);
      const raw = await policyFn(batch);
      checkAbort(signal);
      if (!raw || raw.length !== b * N_MOVES) {
        throw new Error('search: policyFn returned ' + (raw ? raw.length : raw) + ' values, expected ' + b + ' x ' +
                        N_MOVES);
      }
      const logp = normalize ? logSoftmax(raw, b) : Float32Array.from(raw);

      // candidates in parent-major order (search.py: allowed.reshape(-1).nonzero())
      let n = 0;
      for (let p = 0; p < b; p++) {
        for (let m = 0; m < N_MOVES; m++) {
          const face = (m / 3) | 0;
          if (face === lastFace[p]) continue;
          if (noBacktrack && lockAxis[p] >= 0 && face % 3 === lockAxis[p]) continue;
          n++;
        }
      }
      const cParent = new Int32Array(n);
      const cMove = new Uint8Array(n);
      const cScore = new Float32Array(n);
      const children = new Uint8Array(n * N_FACELETS);
      let c = 0;
      for (let p = 0; p < b; p++) {
        const pb = p * N_FACELETS;
        for (let m = 0; m < N_MOVES; m++) {
          const face = (m / 3) | 0;
          if (face === lastFace[p]) continue;
          if (noBacktrack && lockAxis[p] >= 0 && face % 3 === lockAxis[p]) continue;
          cParent[c] = p;
          cMove[c] = m;
          cScore[c] = score[p] + logp[p * N_MOVES + m];  // float32 add (exact via float64, then rounded)
          const cb = c * N_FACELETS;
          const pm = m * N_FACELETS;
          for (let i = 0; i < N_FACELETS; i++) children[cb + i] = beam[pb + perms[pm + i]];  // new[i] = old[perm[i]]
          c++;
        }
      }
      for (let k = 0; k < n; k++) {
        if (cScore[k] !== cScore[k]) throw new Error('search: the policy gave NaN log-probabilities');
      }
      if (onDepth) onDepth({ depth: depth, maxDepth: maxDepth, beam: b, children: n, nodes: nodes, ms: now() - t0 });

      // solved children are checked BEFORE any truncation; best score, ties to the earliest (torch.argmax)
      let j = -1;
      for (let k = 0; k < n; k++) {
        if (isSolvedAt(children, k * N_FACELETS, solved) && (j < 0 || cScore[k] > cScore[j])) j = k;
      }
      if (j >= 0) {
        const path = [cMove[j]];
        let bi = cParent[j];
        for (let k = histParent.length - 1; k >= 0; k--) {
          path.push(histMove[k][bi]);
          bi = histParent[k][bi];
        }
        path.reverse();
        verify(s0, path, tb);
        return { moves: path, nodes: nodes, depth: depth, ms: now() - t0 };
      }
      if (depth === maxDepth) break;

      // dedupe (search.py _dedupe_best): order by (key ascending, score descending, index ascending), keep the
      // first child of each key; the survivors stay in key order
      const keys = new Float64Array(n * 4);
      for (let k = 0; k < n; k++) keyParts(children, k * N_FACELETS, tb, keys, k * 4);
      const order = new Array(n);
      for (let k = 0; k < n; k++) order[k] = k;
      order.sort(function (a, z) {
        const ka = a * 4, kz = z * 4;
        for (let t = 0; t < 4; t++) {
          const d = keys[ka + t] - keys[kz + t];
          if (d !== 0) return d < 0 ? -1 : 1;
        }
        const sa = cScore[a], sz = cScore[z];
        if (sa > sz) return -1;
        if (sa < sz) return 1;
        return a - z;
      });
      const uniq = [];
      for (let k = 0; k < n; k++) {
        const cur = order[k] * 4;
        if (k > 0) {
          const prev = order[k - 1] * 4;
          if (keys[cur] === keys[prev] && keys[cur + 1] === keys[prev + 1] &&
              keys[cur + 2] === keys[prev + 2] && keys[cur + 3] === keys[prev + 3]) continue;
        }
        uniq.push(order[k]);
      }

      // keep the `width` best (search.py: stable descending sort of the survivors' scores): ties keep key order
      const pos = new Array(uniq.length);
      for (let u = 0; u < uniq.length; u++) pos[u] = u;
      pos.sort(function (a, z) {
        const sa = cScore[uniq[a]], sz = cScore[uniq[z]];
        if (sa > sz) return -1;
        if (sa < sz) return 1;
        return a - z;
      });
      const nb = Math.min(width, uniq.length);
      const nextBeam = new Uint8Array(nb * N_FACELETS);
      const nextScore = new Float32Array(nb);
      const nextFace = new Int8Array(nb);
      const nextLock = new Int8Array(nb);
      const selParent = new Int32Array(nb);
      const selMove = new Uint8Array(nb);
      for (let i = 0; i < nb; i++) {
        const k = uniq[pos[i]];
        nextBeam.set(children.subarray(k * N_FACELETS, (k + 1) * N_FACELETS), i * N_FACELETS);
        nextScore[i] = cScore[k];
        selParent[i] = cParent[k];
        selMove[i] = cMove[k];
        const newFace = (cMove[k] / 3) | 0;
        const parentFace = lastFace[cParent[k]];
        // search.py _next_lock: the axis is locked when this move follows a move on the opposite face
        nextLock[i] = (parentFace >= 0 && parentFace % 3 === newFace % 3) ? newFace % 3 : -1;
        nextFace[i] = newFace;
      }
      beam = nextBeam;
      score = nextScore;
      lastFace = nextFace;
      lockAxis = nextLock;
      histParent.push(selParent);
      histMove.push(selMove);
      if (yieldFn) await yieldFn();
    }
    return { moves: null, nodes: nodes, depth: maxDepth, ms: now() - t0 };
  }

  /** Never trust the search (search.py _verify): the path must solve the state under the tables' moves. */
  function verify(s0, path, tb) {
    let s = Uint8Array.from(s0);
    for (let k = 0; k < path.length; k++) {
      const next = new Uint8Array(N_FACELETS);
      const pm = path[k] * N_FACELETS;
      for (let i = 0; i < N_FACELETS; i++) next[i] = s[tb.perms[pm + i]];
      s = next;
    }
    if (!isSolvedAt(s, 0, tb.solved)) throw new Error('search produced a non-solving path (' + path.length + ' moves)');
  }

  const api = {
    beamSearch: beamSearch,
    logSoftmax: logSoftmax,
    prepareTables: prepareTables,
    N_MOVES: N_MOVES,
  };
  root.RubiksSearch = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
