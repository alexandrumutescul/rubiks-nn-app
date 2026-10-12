/* solver.js: cube logic driven ONLY by window.CUBE_TABLES, and the pluggable solver interface.
 *
 * Classic script (no modules, no build step, no DOM access at load), so it also runs under file:// and in Node
 * (the headless checks load cube-tables.js and this file into a vm context). Exposes window.RubiksSolver.
 *
 * The solver contract (every backend implements it):
 *
 *   class Solver {
 *     async solve(state, context) -> { moves: number[], source: string, ms: number }
 *   }
 *
 *   state    Uint8Array(54): colours 0..5 in Kociemba facelet order (the page passes a copy).
 *   context  { history: number[] | null,  moves applied since the cube was last solved (null when unknown, e.g.
 *                                          after "Random state"),
 *              logic,                      the cube logic below (apply, isSolved, ...),
 *              signal: AbortSignal | null  aborted when the user presses Stop,
 *              onProgress: function(text)  optional: a backend may report progress (the in-browser network
 *                                          reports its loading steps and one line per search depth) }
 *   moves    move indices into CUBE_TABLES.move_names ("U", "U'", "U2", "R", ...).
 *   A backend that cannot solve REJECTS with an Error (preferably SolverError); the page shows the message.
 *   The page verifies every returned sequence with the tables before animating it; a backend is never trusted.
 */
(function (root) {
  'use strict';

  function now() {
    return (root.performance && typeof root.performance.now === 'function') ? root.performance.now() : Date.now();
  }

  // ------------------------------------------------------------------------------------------------------------
  // Seeded PRNG (mulberry32) and seed parsing
  // ------------------------------------------------------------------------------------------------------------
  /** mulberry32: a small, fast 32-bit PRNG. Returns a function giving floats in [0, 1). */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /** Seed text -> uint32. Decimal digits are taken as a number (mod 2^32); any other text is hashed (FNV-1a). */
  function seedFromText(text) {
    const s = String(text).trim();
    if (/^\d+$/.test(s)) {
      // mod 2^32 without losing precision for long digit strings
      let v = 0;
      for (let i = 0; i < s.length; i++) v = (v * 10 + (s.charCodeAt(i) - 48)) % 4294967296;
      return v >>> 0;
    }
    let h = 0x811c9dc5;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return h >>> 0;
  }

  // ------------------------------------------------------------------------------------------------------------
  // Cube logic from the exported tables
  // ------------------------------------------------------------------------------------------------------------
  function gcd(a, b) { while (b) { const t = a % b; a = b; b = t; } return a; }

  function createCubeLogic(T) {
    if (!T || !Array.isArray(T.move_perms) || !Array.isArray(T.solved)) {
      throw new Error('CUBE_TABLES missing or malformed: regenerate web/cube-tables.js with python -m rubiks.export_web');
    }
    const N = T.solved.length;            // 54 facelets
    const NM = T.move_names.length;       // 18 moves
    const NF = T.face_names.length;       // 6 faces
    const perms = T.move_perms.map(function (p) { return Int16Array.from(p); });
    const solved = Uint8Array.from(T.solved);
    const inverse = T.inverse_move.slice();
    const moveIndex = new Map(T.move_names.map(function (n, i) { return [n, i]; }));
    const movesOfFace = [];
    for (let f = 0; f < NF; f++) movesOfFace.push([]);
    for (let m = 0; m < NM; m++) movesOfFace[T.move_face[m]].push(m);
    const opposite = [];
    for (let f = 0; f < NF; f++) opposite.push(-1);
    T.scramble_filter.opposite.forEach(function (pair) { opposite[pair[0]] = pair[1]; opposite[pair[1]] = pair[0]; });

    function checkMove(m) {
      if (!Number.isInteger(m) || m < 0 || m >= NM) throw new Error('move index out of range: ' + m);
      return m;
    }

    /** new[i] = old[perm[i]]; returns a new Uint8Array. */
    function applyMove(state, m) {
      const p = perms[checkMove(m)];
      const out = new Uint8Array(N);
      for (let i = 0; i < N; i++) out[i] = state[p[i]];
      return out;
    }

    function applySequence(state, seq) {
      let s = Uint8Array.from(state);
      for (let k = 0; k < seq.length; k++) s = applyMove(s, seq[k]);
      return s;
    }

    function statesEqual(a, b) {
      if (a.length !== b.length) return false;
      for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
      return true;
    }

    function isSolved(state) { return statesEqual(state, solved); }

    function inverseSequence(seq) {
      const out = [];
      for (let k = seq.length - 1; k >= 0; k--) out.push(inverse[checkMove(seq[k])]);
      return out;
    }

    /** The single permutation equivalent to applying seq in order (state_after = state[p]). */
    function sequencePerm(seq) {
      let p = new Int16Array(N);
      for (let i = 0; i < N; i++) p[i] = i;
      for (let k = 0; k < seq.length; k++) {
        const q = perms[checkMove(seq[k])];
        const next = new Int16Array(N);
        for (let i = 0; i < N; i++) next[i] = p[q[i]];
        p = next;
      }
      return p;
    }

    /** Order of a permutation: lcm of its cycle lengths. */
    function permOrder(p) {
      const seen = new Uint8Array(p.length);
      let order = 1;
      for (let s = 0; s < p.length; s++) {
        if (seen[s]) continue;
        let len = 0;
        for (let j = s; !seen[j]; j = p[j]) { seen[j] = 1; len++; }
        order = order / gcd(order, len) * len;
      }
      return order;
    }

    /** "R U' F2" -> [3, 1, 8]. Accepts typographic primes and R2' / R'2 (= R2). Throws on an unknown token. */
    function parseMoves(text) {
      const norm = String(text).replace(/[’′´`]/g, "'");
      const tokens = norm.split(/[\s,;]+/).filter(function (t) { return t.length > 0; });
      return tokens.map(function (tok) {
        let name = tok;
        if (/^[A-Za-z](2'|'2)$/.test(name)) name = name[0] + '2';
        if (!moveIndex.has(name)) {
          throw new Error('unknown move "' + tok + '" (use ' + T.move_names.join(' ') + ')');
        }
        return moveIndex.get(name);
      });
    }

    function formatMoves(seq) {
      return seq.map(function (m) { return T.move_names[checkMove(m)]; }).join(' ');
    }

    function moveByName(name) { return moveIndex.has(name) ? moveIndex.get(name) : -1; }

    function isMoveList(moves) {
      return Array.isArray(moves) && moves.every(function (m) { return Number.isInteger(m) && m >= 0 && m < NM; });
    }

    function toFaceletString(state) {
      let s = '';
      for (let i = 0; i < N; i++) s += T.face_names[state[i]];
      return s;
    }

    /** 54-char URFDLB string -> Uint8Array. Checks length, alphabet, centres and 9 stickers per colour only;
     *  full reachability (twists, flips, parity) is checked by the remote solver (rubiks.cubie on the server). */
    function fromFaceletString(text) {
      const s = String(text).replace(/\s+/g, '').toUpperCase();
      if (s.length !== N) throw new Error('facelet string must have ' + N + ' characters, got ' + s.length);
      const out = new Uint8Array(N);
      for (let i = 0; i < N; i++) {
        const c = T.face_names.indexOf(s[i]);
        if (c < 0) throw new Error('invalid facelet character "' + s[i] + '" at index ' + i);
        out[i] = c;
      }
      for (let f = 0; f < NF; f++) {
        if (out[T.centres[f]] !== solved[T.centres[f]]) {
          throw new Error('centre of face ' + T.face_names[f] + ' (index ' + T.centres[f] + ') must be ' +
                          T.face_names[solved[T.centres[f]]]);
        }
      }
      const counts = colourCounts(out);
      for (let c = 0; c < NF; c++) {
        if (counts[c] !== N / NF) throw new Error('colour ' + T.face_names[c] + ' appears ' + counts[c] + ' times, not 9');
      }
      return out;
    }

    function colourCounts(state) {
      const counts = new Array(NF).fill(0);
      for (let i = 0; i < state.length; i++) counts[state[i]]++;
      return counts;
    }

    /** A canonical scramble: the dataset generator's filter (CUBE_TABLES.scramble_filter), from rng in [0, 1).
     *  face(0) uniform over 6; face(t) uniform over the faces != face(t-1), and when face(t-2), face(t-1) are
     *  opposite also != face(t-2); the move uniform over the 3 moves of the chosen face. */
    function scramble(rng, depth) {
      const faces = [];
      const moves = [];
      for (let t = 0; t < depth; t++) {
        const allowed = [];
        for (let f = 0; f < NF; f++) {
          if (t >= 1 && f === faces[t - 1]) continue;
          if (t >= 2 && opposite[faces[t - 2]] === faces[t - 1] && f === faces[t - 2]) continue;
          allowed.push(f);
        }
        const face = allowed[Math.floor(rng() * allowed.length)];
        const choices = movesOfFace[face];
        faces.push(face);
        moves.push(choices[Math.floor(rng() * choices.length)]);
      }
      return moves;
    }

    /** True if the sequence obeys the scramble filter (used by the checks). */
    function obeysScrambleFilter(moves) {
      for (let t = 1; t < moves.length; t++) {
        const f = T.move_face[moves[t]];
        const p1 = T.move_face[moves[t - 1]];
        if (f === p1) return false;
        if (t >= 2) {
          const p2 = T.move_face[moves[t - 2]];
          if (opposite[p2] === p1 && f === p2) return false;
        }
      }
      return true;
    }

    /** Integer rotation matrix (rows) of the animation of move m: -90 deg * move_quarter_turns[m] about the
     *  face normal (right-hand rule), Rodrigues: R v = c v + s (n x v) + (1 - c)(n . v) n. */
    function turnRotation(m) {
      const n = T.face_normal[T.move_face[m]];
      const theta = -T.move_quarter_turns[m] * Math.PI / 2;
      const c = Math.round(Math.cos(theta));
      const s = Math.round(Math.sin(theta));
      const K = [[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]];
      const R = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) R[i][j] = (i === j ? c : 0) + s * K[i][j] + (1 - c) * n[i] * n[j];
      }
      return R;
    }

    /** Animation angle (radians, right-hand rule about face_normal[move_face[m]]) used by the 3D view. */
    function turnAngle(m) { return -T.move_quarter_turns[m] * Math.PI / 2; }

    function inTurnedLayer(m, pos) {
      const n = T.face_normal[T.move_face[m]];
      return pos[0] * n[0] + pos[1] * n[1] + pos[2] * n[2] === 1;
    }

    /** Check the tables against themselves at load. Returns a report; report.ok is false on any mismatch. */
    function verifyTables() {
      const errors = [];
      const rep = { ok: false, vectorsOk: 0, vectorsTotal: 0, identityOk: false, ordersOk: false, ordersDetail: {},
                    turnsOk: 0, turnsTotal: NM, permsOk: false, uniformOk: false, uniformTotal: 0, errors: errors };
      try {
        // the 18 move tables are permutations, and inverse_move inverts them
        let permsOk = perms.length === NM;
        for (let m = 0; m < NM && permsOk; m++) {
          const seen = new Uint8Array(N);
          for (let i = 0; i < N; i++) { const v = perms[m][i]; if (v < 0 || v >= N || seen[v]) permsOk = false; else seen[v] = 1; }
          const pq = sequencePerm([m, inverse[m]]);
          for (let i = 0; i < N; i++) if (pq[i] !== i) permsOk = false;
        }
        rep.permsOk = permsOk;
        if (!permsOk) errors.push('move_perms / inverse_move are not consistent permutations');

        // seeded test vectors computed by rubiks.cube
        const vectors = T.test_vectors.sequences;
        rep.vectorsTotal = vectors.length;
        vectors.forEach(function (v, k) {
          const s = applySequence(solved, v.moves);
          if (statesEqual(s, v.state)) rep.vectorsOk++;
          else if (errors.length < 8) errors.push('test vector ' + k + ' does not reproduce');
        });

        rep.identityOk = isSolved(applySequence(solved, parseMoves(T.test_vectors.identity)));
        if (!rep.identityOk) errors.push('identity sequence "' + T.test_vectors.identity + '" does not return to solved');

        rep.ordersOk = true;
        Object.keys(T.test_vectors.orders).forEach(function (text) {
          const got = permOrder(sequencePerm(parseMoves(text)));
          rep.ordersDetail[text] = got;
          if (got !== T.test_vectors.orders[text]) {
            rep.ordersOk = false;
            errors.push('order of "' + text + '" is ' + got + ', expected ' + T.test_vectors.orders[text]);
          }
        });

        // the 3D animation rule lands every sticker exactly where the permutation puts it
        const key = function (p, n) { return p.join(',') + '|' + n.join(','); };
        const index = new Map();
        for (let i = 0; i < N; i++) index.set(key(T.sticker_pos[i], T.sticker_normal[i]), i);
        const mul = function (R, v) {
          return [R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
                  R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
                  R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2]];
        };
        for (let m = 0; m < NM; m++) {
          const R = turnRotation(m);
          let ok = true;
          for (let j = 0; j < N; j++) {
            const i = inTurnedLayer(m, T.sticker_pos[j])
              ? index.get(key(mul(R, T.sticker_pos[j]), mul(R, T.sticker_normal[j])))
              : j;
            if (i === undefined || perms[m][i] !== j) { ok = false; break; }
          }
          if (ok) rep.turnsOk++;
          else errors.push('3D turn of ' + T.move_names[m] + ' does not match its permutation');
        }

        // uniform states: shape and colour counts (full validity is checked in Python by the export tests)
        const uni = T.uniform_states;
        rep.uniformTotal = uni.length;
        rep.uniformOk = uni.length > 0 && uni.every(function (s) {
          if (s.length !== N) return false;
          const c = colourCounts(s);
          return c.every(function (x) { return x === N / NF; });
        });
        if (!rep.uniformOk) errors.push('uniform_states malformed');
      } catch (e) {
        errors.push('table check crashed: ' + (e && e.message ? e.message : e));
      }
      rep.ok = errors.length === 0 && rep.vectorsOk === rep.vectorsTotal && rep.vectorsTotal > 0 &&
               rep.identityOk && rep.ordersOk && rep.turnsOk === NM && rep.permsOk && rep.uniformOk;
      rep.summary = 'tables verified ' + rep.vectorsOk + '/' + rep.vectorsTotal + ' vectors, identity ' +
                    (rep.identityOk ? 'OK' : 'FAILED') + ', orders ' + (rep.ordersOk ? 'OK' : 'FAILED');
      return rep;
    }

    return {
      tables: T, N: N, NM: NM, NF: NF, solved: solved,
      applyMove: applyMove, applySequence: applySequence, statesEqual: statesEqual, isSolved: isSolved,
      inverseMove: function (m) { return inverse[checkMove(m)]; }, inverseSequence: inverseSequence,
      sequencePerm: sequencePerm, permOrder: permOrder,
      parseMoves: parseMoves, formatMoves: formatMoves, moveByName: moveByName, isMoveList: isMoveList,
      toFaceletString: toFaceletString, fromFaceletString: fromFaceletString, colourCounts: colourCounts,
      scramble: scramble, obeysScrambleFilter: obeysScrambleFilter,
      turnRotation: turnRotation, turnAngle: turnAngle, inTurnedLayer: inTurnedLayer,
      verifyTables: verifyTables,
    };
  }

  /** Verify a solver's answer against the tables: { valid, solves, finalState }. Never trusts the backend. */
  function verifySolution(logic, state, moves) {
    if (!logic.isMoveList(moves)) return { valid: false, solves: false, finalState: null };
    const finalState = logic.applySequence(state, moves);
    return { valid: true, solves: logic.isSolved(finalState), finalState: finalState };
  }

  // ------------------------------------------------------------------------------------------------------------
  // Solvers
  // ------------------------------------------------------------------------------------------------------------
  /** An error a backend reports to the user. code: 'unavailable' | 'cannot' | 'network' | 'timeout' | 'http' |
   *  'protocol' | 'cancelled'. */
  class SolverError extends Error {
    constructor(message, code) {
      super(message);
      this.name = 'SolverError';
      this.code = code || 'error';
    }
  }

  /** The interface. Subclasses set id / label in the constructor and implement solve(). */
  class Solver {
    constructor(id, label) {
      this.id = id;
      this.label = label;
    }

    /** false for a backend that is known not to work yet (the UI says so instead of calling solve). */
    get available() { return true; }

    /** Why the backend is unavailable (shown in the UI), or '' when available. */
    get unavailableReason() { return ''; }

    /**
     * @param {Uint8Array} state   54 colours, Kociemba facelet order
     * @param {{history: number[]|null, logic: object, signal: AbortSignal|null}} context
     * @returns {Promise<{moves: number[], source: string, ms: number}>}
     */
    async solve(state, context) { // eslint-disable-line no-unused-vars
      throw new SolverError('solve() is not implemented by ' + this.constructor.name, 'unavailable');
    }
  }

  /** Inverts the page's own move history since the last solved state. Cannot solve states the page did not
   *  reach by its own moves ("Random state", a loaded facelet string). Useful today; also a correctness probe. */
  class ReverseHistorySolver extends Solver {
    constructor(logic) {
      super('history', 'Reverse history (moves made on this page)');
      this.logic = logic;
    }

    async solve(state, context) {
      const t0 = now();
      const history = context ? context.history : null;
      if (!Array.isArray(history)) {
        throw new SolverError('reverse history cannot solve this state: it was not reached from a solved cube by ' +
                              'moves made on this page (Random state or a loaded state). Use the remote solver.',
                              'cannot');
      }
      const moves = this.logic.inverseSequence(history);
      return { moves: moves, source: 'reverse-history', ms: now() - t0 };
    }
  }

  /** POST {state: [54 ints]} to url; expects {moves: [indices]} or {error}. Times out after timeoutMs. */
  class RemoteSolver extends Solver {
    constructor(logic, url, options) {
      super('remote', 'Remote solver (POST /solve)');
      const opts = options || {};
      this.logic = logic;
      this.url = url || RemoteSolver.DEFAULT_URL;
      this.timeoutMs = opts.timeoutMs || 20000;
      this.fetchImpl = opts.fetchImpl || null;  // injectable for tests
    }

    async solve(state, context) {
      const t0 = now();
      const url = this.url;
      const fetchFn = this.fetchImpl || (typeof root.fetch === 'function' ? root.fetch.bind(root) : null);
      if (!fetchFn) throw new SolverError('this browser has no fetch(); the remote solver cannot be used', 'unavailable');
      const signal = context && context.signal ? context.signal : null;
      if (signal && signal.aborted) throw new SolverError('cancelled', 'cancelled');

      const controller = typeof root.AbortController === 'function' ? new root.AbortController() : null;
      let timer = null;
      let onAbort = null;
      const timeout = new Promise(function (resolve, reject) {
        timer = setTimeout(function () {
          if (controller) controller.abort();
          reject(new SolverError('the remote solver at ' + url + ' did not answer within ' +
                                 Math.round(this.timeoutMs / 1000) + ' s', 'timeout'));
        }.bind(this), this.timeoutMs);
        if (signal) {
          onAbort = function () {
            if (controller) controller.abort();
            reject(new SolverError('cancelled', 'cancelled'));
          };
          signal.addEventListener('abort', onAbort);
        }
      }.bind(this));
      const work = (async function () {
        const init = {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ state: Array.from(state) }),
        };
        if (controller) init.signal = controller.signal;
        const resp = await fetchFn(url, init);
        const text = await resp.text();
        return { resp: resp, text: text };
      })();
      work.catch(function () {});  // a late rejection after the timeout is expected, not an error

      let result;
      try {
        result = await Promise.race([work, timeout]);
      } catch (e) {
        if (e instanceof SolverError) throw e;
        throw new SolverError('could not reach the remote solver at ' + url + ' (' + (e && e.message ? e.message : e) +
                              '). Start it with: python -m rubiks.serve --port 8765 --web web', 'network');
      } finally {
        clearTimeout(timer);
        if (signal && onAbort) signal.removeEventListener('abort', onAbort);
      }

      let data = null;
      try { data = JSON.parse(result.text); } catch (e) { data = null; }
      const status = result.resp.status;
      if (!result.resp.ok) {
        const detail = data && typeof data.error === 'string' ? data.error : (result.text || '').slice(0, 200);
        throw new SolverError('remote solver: HTTP ' + status + (detail ? ': ' + detail : ''), 'http');
      }
      if (!data || typeof data !== 'object') throw new SolverError('remote solver: the response is not JSON', 'protocol');
      if (typeof data.error === 'string') throw new SolverError('remote solver: ' + data.error, 'http');
      if (!this.logic.isMoveList(data.moves)) {
        throw new SolverError('remote solver: "moves" must be a list of move indices 0..' + (this.logic.NM - 1),
                              'protocol');
      }
      const name = typeof data.solver === 'string' ? data.solver : url;
      const out = { moves: data.moves.slice(), source: 'remote: ' + name, ms: now() - t0 };
      if (typeof data.ms === 'number') out.serverMs = data.ms;
      return out;
    }
  }
  RemoteSolver.DEFAULT_URL = 'http://localhost:8765/solve';

  // ------------------------------------------------------------------------------------------------------------
  // The trained policy network in the browser: ONNX Runtime Web + the beam search of search.js
  // ------------------------------------------------------------------------------------------------------------
  const ORT_VERSION = '1.20.1';  // pinned; ort.min.js and its .wasm / .mjs files come from this cdnjs folder
  const ORT_CDN = 'https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/';
  const MODEL_FILE = 'policy-v2.onnx';
  // Relative to the page: the repository root served as a whole (page at /web/), then next to the page (the GitHub
  // Pages site publishes the file as model/policy-v2.onnx; a server that serves web/ only needs a copy at
  // web/model/policy-v2.onnx). See web/model/README.md.
  const MODEL_URLS = ['../models/' + MODEL_FILE, 'models/' + MODEL_FILE, 'model/' + MODEL_FILE];
  const ONNX_INPUT = 'onehot';    // float32 (batch, 324): feature = position * 6 + colour (models/policy-v2.onnx.json)
  const ONNX_OUTPUT = 'logits';   // float32 (batch, 18), the order of CUBE_TABLES.move_names
  const N_COLOURS = 6;
  const FILE_REASON = 'needs a server or the Pages URL: this page is opened as file://';

  function isFilePage() {
    return !!(root.location && root.location.protocol === 'file:');
  }

  function errText(e) { return e && e.message ? e.message : String(e); }

  /** Let the browser paint and handle clicks (Stop) between search depths. */
  function yieldToBrowser() { return new Promise(function (resolve) { setTimeout(resolve, 0); }); }

  /** Add a classic script tag once; resolves when it has run. */
  function loadScript(url) {
    return new Promise(function (resolve, reject) {
      const doc = root.document;
      if (!doc || !doc.head) { reject(new Error('no document to load ' + url + ' into')); return; }
      const el = doc.createElement('script');
      el.src = url;
      el.async = true;
      el.referrerPolicy = 'no-referrer';
      el.onload = function () { resolve(); };
      el.onerror = function () { el.remove(); reject(new Error('could not load ' + url)); };
      doc.head.appendChild(el);
    });
  }

  /**
   * OnnxSolver: models/policy-v2.onnx with ONNX Runtime Web (wasm backend, loaded from cdnjs on first use) and the
   * beam search of search.js, which reproduces rubiks/search.py beam_search: width this.width (the page's setting,
   * 64..4096, default 1024), at most 40 moves, one session.run per depth over the whole beam. Under file:// the
   * browser cannot fetch the model, so the backend reports itself unavailable there.
   */
  class OnnxSolver extends Solver {
    constructor(logic, options) {
      super('onnx', 'Neural network (ONNX, in the browser)');
      const opts = options || {};
      this.logic = logic;
      this.width = opts.width || OnnxSolver.WIDTH_DEFAULT;
      this.maxDepth = opts.maxDepth || OnnxSolver.MAX_DEPTH;
      this.modelUrl = opts.modelUrl || null;      // one explicit URL (the page's setting); null: try MODEL_URLS
      this.ortVersion = opts.ortVersion || ORT_VERSION;
      this.ortBase = ORT_CDN + this.ortVersion + '/';
      this.ortUrl = this.ortBase + 'ort.min.js';
      this.loaded = null;                         // {session, url, name, threads, ortVersion}
      this._loading = null;                       // {key, promise} of a load in progress
      this._lock = Promise.resolve();             // one search at a time
      this._active = 0;                           // searches holding or waiting for the lock
    }

    get available() { return !isFilePage(); }

    get unavailableReason() { return isFilePage() ? FILE_REASON : ''; }

    /** The model URLs this solver tries, in order. */
    modelCandidates() { return this.modelUrl ? [this.modelUrl] : MODEL_URLS.slice(); }

    /** One-hot encoding of the model contract: float32 (n, 324), feature = position * 6 + colour. */
    static encode(states) {
      const n = states.length;
      const N = 54;
      const x = new Float32Array(n * N * N_COLOURS);
      for (let r = 0; r < n; r++) {
        const s = states[r];
        const base = r * N * N_COLOURS;
        for (let p = 0; p < N; p++) x[base + p * N_COLOURS + s[p]] = 1;
      }
      return x;
    }

    async loadOrt(onProgress) {
      let ort = root.ort;
      if (!ort || !ort.InferenceSession) {
        if (onProgress) onProgress('loading ONNX Runtime Web ' + this.ortVersion + ' from cdnjs…');
        try {
          await loadScript(this.ortUrl);
        } catch (e) {
          throw new SolverError('could not load ONNX Runtime Web from cdnjs (' + this.ortUrl + '); the in-browser ' +
                                'network needs internet access for it', 'network');
        }
        ort = root.ort;
        if (!ort || !ort.InferenceSession) throw new SolverError(this.ortUrl + ' did not define ort', 'unavailable');
      }
      // The .wasm (and .mjs) files from the same cdnjs folder; its wasm build has SIMD built in. Threads need
      // cross-origin isolation (COOP/COEP headers, which neither GitHub Pages nor the dev server sends); without it
      // ONNX Runtime runs single-threaded.
      ort.env.wasm.wasmPaths = this.ortBase;
      const cores = (root.navigator && root.navigator.hardwareConcurrency) || 2;
      ort.env.wasm.numThreads = root.crossOriginIsolated ? Math.max(1, Math.min(4, Math.floor(cores / 2))) : 1;
      return ort;
    }

    /** Load ONNX Runtime and the model once; returns {session, url, name, threads, ortVersion, loadMs}. */
    async ensureSession(onProgress) {
      const key = this.modelCandidates().join('\n') + '\n' + this.ortVersion;
      if (this.loaded && this.loaded.key === key) return Object.assign({}, this.loaded, { loadMs: 0 });
      if (!this._loading || this._loading.key !== key) {
        const self = this;
        const promise = (async function () {
          const t0 = now();
          const ort = await self.loadOrt(onProgress);
          const tried = [];
          for (const url of self.modelCandidates()) {
            if (onProgress) onProgress('loading the model: ' + url + '…');
            let resp;
            try {
              resp = await root.fetch(url);
            } catch (e) {
              tried.push(url + ' (' + errText(e) + ')');
              continue;
            }
            if (!resp.ok) { tried.push(url + ' (HTTP ' + resp.status + ')'); continue; }
            const type = (resp.headers && resp.headers.get('content-type')) || '';
            if (/text\/html/i.test(type)) { tried.push(url + ' (an HTML page, not a model)'); continue; }
            let session;
            try {
              const bytes = new Uint8Array(await resp.arrayBuffer());
              session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'],
                                                                   graphOptimizationLevel: 'all' });
            } catch (e) {
              tried.push(url + ' (ONNX Runtime could not load it: ' + errText(e) + ')');
              continue;
            }
            if (session.inputNames.indexOf(ONNX_INPUT) < 0 || session.outputNames.indexOf(ONNX_OUTPUT) < 0) {
              tried.push(url + ' (inputs ' + session.inputNames.join(',') + ', outputs ' +
                         session.outputNames.join(',') + '; expected ' + ONNX_INPUT + ' -> ' + ONNX_OUTPUT + ')');
              continue;
            }
            const file = url.split(/[?#]/)[0].split('/').pop() || url;
            return { key: key, session: session, url: url, name: file.replace(/\.onnx$/i, ''),
                     threads: ort.env.wasm.numThreads, ortVersion: self.ortVersion, loadMs: now() - t0 };
          }
          throw new SolverError('could not load the model; tried ' + tried.join('; ') + '. Serve the page with the ' +
                                'model next to it (see web/model/README.md) or open the GitHub Pages site.',
                                'unavailable');
        })();
        this._loading = { key: key, promise: promise };
      }
      const loading = this._loading;
      try {
        const rec = await loading.promise;
        this.loaded = rec;
        return rec;
      } finally {
        if (this._loading === loading) this._loading = null;  // a failed load is retried on the next solve
      }
    }

    /** The policy function search.js calls: one session.run over the whole batch, logits (n * 18). */
    policyFnFor(session) {
      const ort = root.ort;
      const NM = this.logic.NM;
      return async function (states) {
        const n = states.length;
        const feeds = {};
        feeds[ONNX_INPUT] = new ort.Tensor('float32', OnnxSolver.encode(states), [n, 54 * N_COLOURS]);
        const out = await session.run(feeds);
        const y = out[ONNX_OUTPUT];
        if (!y || !y.data || y.data.length !== n * NM) {
          throw new SolverError('the model returned ' + (y && y.data ? y.data.length : 'no') + ' logits, expected ' +
                                n + ' x ' + NM, 'protocol');
        }
        return y.data;
      };
    }

    /** Load everything and return the policy function (for search-test.html). */
    async createPolicyFn(onProgress) {
      const rec = await this.ensureSession(onProgress);
      return this.policyFnFor(rec.session);
    }

    _acquire() {
      const self = this;
      const prev = this._lock;
      let release;
      const mine = new Promise(function (resolve) { release = function () { self._active--; resolve(); }; });
      this._active++;
      this._lock = prev.then(function () { return mine; });
      return prev.then(function () { return release; });
    }

    async solve(state, context) {
      const ctx = context || {};
      const signal = ctx.signal || null;
      const onProgress = typeof ctx.onProgress === 'function' ? ctx.onProgress : null;
      if (!this.available) throw new SolverError(FILE_REASON, 'unavailable');
      const search = root.RubiksSearch;
      if (!search) throw new SolverError('search.js did not load', 'unavailable');
      const width = this.width;
      const maxDepth = this.maxDepth;
      if (!Number.isInteger(width) || width < 1 || width > OnnxSolver.WIDTH_MAX) {
        throw new SolverError('beam width must be a whole number from 1 to ' + OnnxSolver.WIDTH_MAX, 'cannot');
      }
      const cancelled = function () { return new SolverError('cancelled', 'cancelled'); };
      if (signal && signal.aborted) throw cancelled();
      if (onProgress && this._active > 0) onProgress('waiting for the previous search to stop…');
      const release = await this._acquire();  // a cancelled search stops at its next depth
      try {
        if (signal && signal.aborted) throw cancelled();
        const rec = await this.ensureSession(onProgress);
        if (signal && signal.aborted) throw cancelled();
        if (onProgress) onProgress('searching: beam ' + width + ', model ' + rec.url);
        await yieldToBrowser();  // paint the line before the first session.run
        let res;
        try {
          res = await search.beamSearch(this.policyFnFor(rec.session), state, {
            tables: this.logic.tables, width: width, maxDepth: maxDepth, signal: signal, yieldFn: yieldToBrowser,
            onDepth: function (p) {
              if (onProgress) {
                onProgress('depth ' + p.depth + '/' + p.maxDepth + ': beam ' + p.beam + ', ' + p.nodes + ' nodes, ' +
                           Math.round(p.ms) + ' ms');
              }
            },
          });
        } catch (e) {
          if (e && e.name === 'AbortError') throw cancelled();
          if (e instanceof SolverError) throw e;
          throw new SolverError('in-browser search failed: ' + errText(e), 'error');
        }
        if (res.moves === null) {
          throw new SolverError('no solution found within ' + maxDepth + ' moves at beam width ' + width + ' (' +
                                res.nodes + ' nodes); a larger beam width may find one', 'cannot');
        }
        const out = { moves: res.moves, source: 'onnx: ' + rec.name + ' (beam ' + width + ')', ms: res.ms,
                      nodes: res.nodes };
        if (rec.loadMs > 0) out.loadMs = rec.loadMs;
        return out;
      } finally {
        release();
      }
    }
  }
  OnnxSolver.ORT_VERSION = ORT_VERSION;
  OnnxSolver.MODEL_URLS = MODEL_URLS.slice();
  OnnxSolver.WIDTH_MIN = 64;       // the page's setting
  OnnxSolver.WIDTH_MAX = 4096;
  OnnxSolver.WIDTH_DEFAULT = 1024;
  OnnxSolver.MAX_DEPTH = 40;

  /** The backends offered by the page, in menu order. Add a new backend here (or push onto the returned list). */
  function createSolvers(logic, options) {
    const opts = options || {};
    return [
      new ReverseHistorySolver(logic),
      new RemoteSolver(logic, opts.remoteUrl, { timeoutMs: opts.timeoutMs }),
      new OnnxSolver(logic, opts.onnx),
    ];
  }

  const api = {
    mulberry32: mulberry32,
    seedFromText: seedFromText,
    createCubeLogic: createCubeLogic,
    verifySolution: verifySolution,
    SolverError: SolverError,
    Solver: Solver,
    ReverseHistorySolver: ReverseHistorySolver,
    RemoteSolver: RemoteSolver,
    OnnxSolver: OnnxSolver,
    createSolvers: createSolvers,
  };
  root.RubiksSolver = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
