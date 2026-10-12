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
 *              signal: AbortSignal | null  aborted when the user presses Stop }
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

  /**
   * OnnxSolver: NOT IMPLEMENTED YET. A documented stub for the trained network (phase 2).
   *
   * Intended design (TODO, when the network exists):
   *  - The training code exports the network to ONNX (torch.onnx.export) as e.g. web/model/rubiks-nn.onnx, with a
   *    small JSON sidecar that pins the input encoding and the output meaning (move order = CUBE_TABLES.move_names).
   *  - The page loads ONNX Runtime Web from cdnjs on first use, by injecting
   *      <script src="https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/<pinned version>/ort.min.js">
   *    and then `this.session = await ort.InferenceSession.create(modelUrl)`.
   *  - encode(state): the input tensor exactly as in training (the design fixes a 54-sticker one-hot, 324 features;
   *    the layout must be read from the sidecar, not assumed here).
   *  - solve(state, context): a search loop over the network's output (greedy or beam over the 18-move policy, or
   *    weighted A* over a cost-to-go value), applying moves with context.logic, stopping when logic.isSolved()
   *    or after maxMoves; it returns {moves, source: 'onnx: <model name>', ms}. The page verifies the result
   *    exactly like any other backend's.
   *  - ONNX Runtime Web fetches its .wasm files and the model with fetch(), which fails under file://, so this
   *    backend will need the page served over http (python -m rubiks.serve) or the model embedded in a .js file.
   * The server-side alternative needs no page change: replace the backend in rubiks/serve.py (_backend_solve)
   * and keep using RemoteSolver.
   */
  class OnnxSolver extends Solver {
    constructor(logic, options) {
      super('onnx', 'Neural network (ONNX, in the browser)');
      const opts = options || {};
      this.logic = logic;
      this.modelUrl = opts.modelUrl || 'model/rubiks-nn.onnx';
      this.ortUrl = opts.ortUrl || null;  // TODO: pin https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/<ver>/ort.min.js
      this.session = null;
    }

    get available() { return false; }

    get unavailableReason() { return 'model not available yet'; }

    async solve(state, context) { // eslint-disable-line no-unused-vars
      // TODO(phase 2): load ort + the model, encode(state), search with the policy/value, return the moves.
      throw new SolverError('model not available yet', 'unavailable');
    }
  }

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
