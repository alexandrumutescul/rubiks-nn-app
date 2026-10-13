#!/usr/bin/env bash
# Start the Rubik's cube web app in Docker on the first free host port from 8765 to 8800 (macOS and Linux).
# Run ./run.sh --help for the options. Needs Docker; bash 3.2 or newer.
set -euo pipefail

NAME=rubiks-nn-app
LABEL=app=rubiks-nn-app   # on every launch; the app's container is found by its name or by this label
LOCAL_IMAGE=rubiks-nn-app
PUBLISHED_IMAGE=ghcr.io/alexandrumutescul/rubiks-nn-app:latest
PORT_SPAN=35          # ports tried: FIRST .. FIRST + 35 (8765 .. 8800 by default)
HEALTH_TIMEOUT=60     # seconds to wait for GET /solver
SOLVE_TIMEOUT=30      # seconds each test solve may take
STOP_TIMEOUT=10       # seconds "stop" waits, after docker stop and docker rm, for the app's containers to be gone
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The start-up test cubes, as states (54 colours). The first is one turn from solved, so it must come back solved at
# any --max-depth; the second, six moves from solved, is checked as well when --max-depth is 6 or more.
TEST_MOVES="R"
TEST_STATE='[0,0,2,0,0,2,0,0,2,1,1,1,1,1,1,1,1,1,2,2,3,2,2,3,2,2,3,3,3,5,3,3,5,3,3,5,4,4,4,4,4,4,4,4,4,0,5,5,0,5,5,0,5,5]'
TEST2_MOVES="R U F' L2 D B'"
TEST2_STATE='[0,2,2,3,0,0,3,1,1,5,5,3,3,1,0,4,2,0,5,3,3,5,2,2,2,2,2,0,0,0,3,3,4,2,1,5,5,4,4,5,4,4,4,5,1,1,1,1,4,5,1,4,0,3]'

# One POST /solve, run by the container's own Python (always in the image, unlike curl or wget on the host):
#   docker exec -i <container> python - '<state>' <seconds>    with this program on stdin.
# It prints one line and exits 0 when the answer has "verified": true (the server's own check that the moves solve
# the cube), 2 for an HTTP error, 3 for no answer within the time limit, 4 for an answer that is not verified. Any
# other status (1: Python failed; 125 to 127: docker exec failed) means the check itself could not run.
IFS= read -r -d '' SOLVE_CHECK <<'PY' || true
import json, sys, urllib.error, urllib.request

request = urllib.request.Request(
    'http://127.0.0.1:8765/solve', data=json.dumps({'state': json.loads(sys.argv[1])}).encode(),
    headers={'Content-Type': 'application/json'})
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never a proxy for 127.0.0.1
try:
    with opener.open(request, timeout=float(sys.argv[2])) as response:
        text = response.read().decode('utf-8', 'replace')
except urllib.error.HTTPError as exc:
    print('HTTP %d %s' % (exc.code, exc.read().decode('utf-8', 'replace').strip()[:300]))
    sys.exit(2)
except OSError as exc:  # a time-out, or nothing listening
    print('no answer: %s' % exc)
    sys.exit(3)
try:
    answer = json.loads(text)
except ValueError:
    answer = None
if isinstance(answer, dict) and answer.get('verified') is True:
    moves, ms = answer.get('moves') or [], answer.get('ms')
    print('%s (%d move%s, %s ms)' % (answer.get('notation') or 'no moves', len(moves), '' if len(moves) == 1 else 's',
                                     '%.0f' % ms if isinstance(ms, (int, float)) else '?'))
    sys.exit(0)
print('not verified: %s' % text.strip()[:300])
sys.exit(4)
PY

usage() {
  cat <<USAGE
Usage: ./run.sh [start] [options]    start the app (the default command)
       ./run.sh stop                 stop it

Options:
  --image IMAGE     use this image instead of building one here, e.g.
                    --image $PUBLISHED_IMAGE
  --port N          first host port to try (default \$PORT or 8765); N .. N+$PORT_SPAN are tried in turn
  --backend NAME    policy or hybrid (default \$BACKEND or policy)
  --beam N          beam width of the search (default \$BEAM or 1024; 256 is faster, 4096 finds shorter solutions)
  --max-depth N     longest solution the search tries (default \$MAX_DEPTH or 40)
  --open            open the page in the default browser once it is up
  -h, --help        this help

Environment:
  ASTAR_NODES      positive integer A* expanded-node cap for hybrid (default 200000)

Without --image the image is built from this directory (docker build -t $LOCAL_IMAGE .); the first build downloads
the Python base image and the CPU build of PyTorch, later builds are cached.

Once the server answers, the container's own Python asks it to solve a cube one turn from solved (and, with
--max-depth 6 or more, one six moves from solved). The start counts as a success only if that answer comes back
verified; otherwise the script exits with an error and leaves the container running for docker logs $NAME.
A copy of the app that is already running gets the same checks (the first cube only) before its address is printed.
USAGE
}

die() { echo "run.sh: $*" >&2; exit 1; }
is_uint() { case "$1" in '' | *[!0-9]*) return 1 ;; *) return 0 ;; esac; }

COMMAND=start
IMAGE=""
FIRST_PORT="${PORT:-8765}"
BACKEND="${BACKEND:-policy}"
BACKEND_SOURCE='$BACKEND'
ASTAR_NODES="${ASTAR_NODES-200000}"
BEAM="${BEAM:-1024}"
MAX_DEPTH="${MAX_DEPTH:-40}"
OPEN=0
while [ $# -gt 0 ]; do
  case "$1" in
    start | stop) COMMAND="$1" ;;
    --image) [ $# -ge 2 ] || die "--image needs a value"; IMAGE="$2"; shift ;;
    --image=*) IMAGE="${1#*=}" ;;
    --port) [ $# -ge 2 ] || die "--port needs a value"; FIRST_PORT="$2"; shift ;;
    --port=*) FIRST_PORT="${1#*=}" ;;
    --backend) [ $# -ge 2 ] || die "--backend needs a value"; BACKEND="$2"; BACKEND_SOURCE=--backend; shift ;;
    --backend=*) BACKEND="${1#*=}"; BACKEND_SOURCE=--backend ;;
    --beam) [ $# -ge 2 ] || die "--beam needs a value"; BEAM="$2"; shift ;;
    --beam=*) BEAM="${1#*=}" ;;
    --max-depth) [ $# -ge 2 ] || die "--max-depth needs a value"; MAX_DEPTH="$2"; shift ;;
    --max-depth=*) MAX_DEPTH="${1#*=}" ;;
    --open) OPEN=1 ;;
    -h | --help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
  shift
done

is_uint "$FIRST_PORT" && [ "$FIRST_PORT" -ge 1 ] && [ "$FIRST_PORT" -le $((65535 - PORT_SPAN)) ] ||
  die "--port must be a number from 1 to $((65535 - PORT_SPAN)), got '$FIRST_PORT'"
case "$BACKEND" in policy | hybrid) ;; *) die "$BACKEND_SOURCE must be policy or hybrid, got '$BACKEND'" ;; esac
[[ "$ASTAR_NODES" =~ ^0*[1-9][0-9]*$ ]] || die "\$ASTAR_NODES must be a positive integer, got '$ASTAR_NODES'"
is_uint "$BEAM" && [ "$BEAM" -ge 1 ] || die "--beam must be a whole number >= 1, got '$BEAM'"
is_uint "$MAX_DEPTH" && [ "$MAX_DEPTH" -ge 1 ] || die "--max-depth must be a whole number >= 1, got '$MAX_DEPTH'"
LAST_PORT=$((FIRST_PORT + PORT_SPAN))

command -v docker >/dev/null 2>&1 || die "docker not found: install Docker Desktop (macOS) or Docker Engine (Linux)"
docker info >/dev/null 2>&1 ||
  die "cannot talk to Docker: is Docker Desktop (or the docker service) running? On Linux, is this user in the docker group?"

# The app's containers: the one named $NAME and any labelled $LABEL (Docker Compose, or a docker run typed by hand),
# each once, as "<id> <state> <name>" lines.
app_containers() {
  local ids id seen=" " info state cname
  ids="$(docker container inspect -f '{{.Id}}' "$NAME" 2>/dev/null || true)"
  ids="$ids $(docker ps -a -q --no-trunc --filter "label=$LABEL" 2>/dev/null || true)"
  for id in $ids; do
    case "$seen" in *" $id "*) continue ;; esac
    seen="$seen$id "
    info="$(docker container inspect -f '{{.State.Status}} {{.Name}}' "$id" 2>/dev/null || true)"
    if [ -n "$info" ]; then
      state="${info%% *}"
      cname="${info#* }"
      echo "$id $state ${cname#/}"
    fi
  done
}

state_of() { docker container inspect -f '{{.State.Status}}' "$1" 2>/dev/null || true; }

# the host port a container publishes 8765 on ("0.0.0.0:8766" -> 8766)
host_port() { docker port "$1" 8765/tcp 2>/dev/null | sed -n '1s/.*://p' || true; }

# The ids of the app's containers that still exist (named $NAME or labelled $LABEL, in any state), each once;
# status 1 when Docker cannot be asked.
app_ids_left() {
  local by_label by_name
  by_label="$(docker ps -a -q --no-trunc --filter "label=$LABEL" 2>/dev/null)" || return 1
  by_name="$(docker ps -a --no-trunc --format '{{.ID}} {{.Names}}' 2>/dev/null |
             awk -v n="$NAME" '$2 == n { print $1 }')" || return 1
  printf '%s\n%s\n' "$by_name" "$by_label" | awk 'NF && !seen[$0]++'
}

open_url() {
  if [ "$(uname -s)" = Darwin ] && command -v open >/dev/null 2>&1; then
    open "$1"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$1" >/dev/null 2>&1 &
  elif command -v wslview >/dev/null 2>&1; then
    wslview "$1"
  else
    echo "(no browser opener found; open the address above by hand)"
  fi
}

# Checks of a running container: after a start, and for a copy that is already running.
http_get() {  # body of a GET, or a failure; status 2 when neither curl nor wget is installed
  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 3 "$1" 2>/dev/null
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- -T 3 "$1" 2>/dev/null
  else
    return 2
  fi
}

# is_up PORT ID: GET /solver answers; keep its backend for the reuse check.
READY_BACKEND=""
is_up() {
  local body rc=0
  body="$(http_get "http://127.0.0.1:$1/solver")" || rc=$?
  if [ "$rc" = 2 ]; then  # no HTTP client here: use the image's own Python
    rc=0
    body="$(docker exec "$2" python -c 'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:8765/solver", timeout=3).read().decode())' 2>/dev/null)" || rc=$?
  fi
  [ "$rc" = 0 ] || return 1
  READY_BACKEND="$(printf '%s' "$body" | sed -n 's/.*"backend"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
  [ -n "$READY_BACKEND" ]
}

# wait_ready ID PORT: waits up to HEALTH_TIMEOUT s until is_up; 0 = it answers, 1 = the container is no longer
# running, 2 = no answer in time
wait_ready() {
  local deadline=$((SECONDS + HEALTH_TIMEOUT))
  until is_up "$2" "$1"; do
    [ "$(state_of "$1")" = running ] || return 1
    [ "$SECONDS" -lt "$deadline" ] || return 2
    sleep 1
  done
  return 0
}

# check_solve ID STATE: one POST /solve from inside container ID; prints the one-line result, returns its status
# (above)
check_solve() { docker exec -i "$1" python - "$2" "$SOLVE_TIMEOUT" <<<"$SOLVE_CHECK" 2>&1; }

# first_test_solve ID NOTE: the mandatory test solve ($TEST_MOVES); exits with an error that ends in NOTE unless the
# answer comes back verified
first_test_solve() {
  local answer rc=0
  answer="$(check_solve "$1" "$TEST_STATE")" || rc=$?
  case "$rc" in
    0) echo "test solve OK: the network solved $TEST_MOVES with $answer" ;;
    3) die "the test solve ($TEST_MOVES) got no answer within ${SOLVE_TIMEOUT} s: $answer; $2" ;;
    *) die "the test solve ($TEST_MOVES) failed: $answer; $2" ;;
  esac
}

if [ "$COMMAND" = stop ]; then
  found="$(app_containers)"
  failures=""
  if [ -n "$found" ]; then
    while read -r id state cname; do
      # docker stop and docker rm also fail for a container that is already gone or already being removed (--rm),
      # so a failure counts only if the container is still there afterwards (checked below)
      out="$(docker stop "$id" </dev/null 2>&1)" || failures="$failures
  docker stop $cname: $out"
      out="$(docker rm -f "$id" </dev/null 2>&1)" || failures="$failures
  docker rm -f $cname: $out"
    done <<EOF
$found
EOF
  fi
  # No container of the app may be left, by name or label. Removal by --rm can lag docker stop by a moment.
  deadline=$((SECONDS + STOP_TIMEOUT))
  while :; do
    rc=0
    left="$(app_ids_left)" || rc=$?
    [ "$rc" = 0 ] && [ -z "$left" ] && break
    if [ "$SECONDS" -ge "$deadline" ]; then
      [ "$rc" = 0 ] || die "cannot check that the app has stopped (docker ps failed); check with" \
                           "docker ps -a --filter label=$LABEL${failures:+
docker said:$failures}"
      list=""
      for id in $left; do
        info="$(docker container inspect -f '{{.Name}} (state: {{.State.Status}})' "$id" 2>/dev/null || true)"
        info="${info#/}"
        list="$list
  ${info:-$id}"
      done
      die "a container of this app is still there after ${STOP_TIMEOUT} s:$list${failures:+
docker said:$failures}
remove it by hand (docker rm -f <name>) and run ./run.sh stop again"
    fi
    sleep 1
  done
  if [ -z "$found" ]; then
    echo "$NAME is not running"
  else
    printf '%s\n' "$found" | awk 'NF { print "stopped " $3 }'
  fi
  exit 0
fi

# -- already running? --------------------------------------------------------------------------------------------
# A running copy gets the same checks as a new start (GET /solver answers, then the first test cube comes back
# verified) before its address is printed; if they fail, the container is left as it is.
found="$(app_containers)"
running="$(printf '%s\n' "$found" | awk '$2 == "running" && !n { print; n = 1 }')"
if [ -n "$running" ]; then
  read -r id state cname <<<"$running"
  port="$(host_port "$id")"
  [ -n "$port" ] || die "$cname (a container of this app) is running but publishes no port; stop it with ./run.sh stop"
  left_as_is="$cname is left as it is, so you can look: docker logs $cname; stop it with ./run.sh stop"
  echo "$cname is already running on port $port; checking that it answers ..."
  rc=0
  wait_ready "$id" "$port" || rc=$?
  case "$rc" in
    0) ;;
    1)
      st="$(state_of "$id")"
      die "$cname stopped while this script waited for it to answer${st:+ (state: $st)}; docker logs $cname shows" \
          "why while the container exists. Run ./run.sh again to start the app"
      ;;
    *) die "$cname gives no answer on http://localhost:$port/solver after ${HEALTH_TIMEOUT} s; $left_as_is" ;;
  esac
  [ "$READY_BACKEND" = "$BACKEND" ] || die "running with backend $READY_BACKEND; ./run.sh stop first"
  first_test_solve "$id" "$left_as_is"
  echo "$cname is already running: open http://localhost:$port/"
  echo "(stop it with ./run.sh stop)"
  [ "$OPEN" = 1 ] && open_url "http://localhost:$port/"
  exit 0
fi
# None is running. Containers that have exited are cleared away, by id. Any other state (created, restarting, paused)
# can be another start in progress, so that container is left alone and reported.
if [ -n "$found" ]; then
  while read -r id state cname; do
    case "$state" in
      exited | dead)
        echo "removing a stopped container of this app: $cname (state: $state)"
        docker rm -f "$id" </dev/null >/dev/null 2>&1 || true
        ;;
      *)
        die "$cname (a container of this app) is $state; another ./run.sh may be starting it right now. Run" \
            "./run.sh again in a moment, or remove it with ./run.sh stop"
        ;;
    esac
  done <<EOF
$found
EOF
fi

# -- the image -----------------------------------------------------------------------------------------------------
if [ -z "$IMAGE" ]; then
  IMAGE="$LOCAL_IMAGE"
  echo "building the image $IMAGE from $SCRIPT_DIR (cached after the first build) ..."
  docker build -t "$IMAGE" "$SCRIPT_DIR"
else
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "using the local image $IMAGE"
  else
    echo "pulling $IMAGE ..."
    docker pull "$IMAGE" ||
      die "cannot pull $IMAGE (offline, or the image is not public?); leave out --image to build it here instead"
  fi
fi

# -- a free port ---------------------------------------------------------------------------------------------------
port_busy() {
  local p="$1"
  # something answers on the loopback address
  if (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null; then return 0; fi
  # a listener this user can see (lsof), or any listener (ss, Linux)
  if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then return 0; fi
  if command -v ss >/dev/null 2>&1 &&
     ss -ltn 2>/dev/null | awk -v p=":$p" '$4 ~ p"$" { found = 1 } END { exit !found }'; then return 0; fi
  # published by another container, running or not yet listening
  # (no grep -q: an early exit would break the pipe and, under pipefail, read as "free")
  if docker ps --format '{{.Ports}}' 2>/dev/null | grep -E "(^|[ ,:])$p->" >/dev/null; then return 0; fi
  return 1
}

# The container is created, then started, so its id is known before anything can fail. After a failed start only that
# id is removed, never a container found by name: another start may own the name meanwhile.
# 0 = started (STARTED_ID is its id); 1 = the port turned out to be taken (try the next one); 2 = any other failure
# (message printed); 3 = a container named $NAME appeared since the check above (another start).
STARTED_ID=""
try_run() {
  local p="$1" out id
  if ! out="$(docker create --rm --name "$NAME" --label "$LABEL" -p "$p:8765" \
                -e BACKEND="$BACKEND" -e ASTAR_NODES="$ASTAR_NODES" -e BEAM="$BEAM" -e MAX_DEPTH="$MAX_DEPTH" "$IMAGE" 2>&1)"; then
    if printf '%s\n' "$out" | grep -i 'already in use by' >/dev/null; then return 3; fi
    printf '%s\n' "$out" >&2
    return 2
  fi
  id="$(printf '%s\n' "$out" | sed -n '/^[0-9a-f]\{64\}$/p' | tail -n 1)"
  if [ -z "$id" ]; then
    printf '%s\n' "$out" >&2
    echo "docker create printed no container id" >&2
    return 2
  fi
  if out="$(docker start "$id" 2>&1)"; then
    STARTED_ID="$id"
    return 0
  fi
  docker rm -f "$id" >/dev/null 2>&1 || true   # this call's own container (a failed start can leave it behind)
  if printf '%s\n' "$out" |
     grep -Ei 'port is already allocated|address already in use|ports are not available|bind for|forbidden by its access permissions' \
       >/dev/null
  then
    echo "port $p is taken after all; trying the next one" >&2
    return 1
  fi
  printf '%s\n' "$out" >&2
  return 2
}

port=""
p="$FIRST_PORT"
while [ "$p" -le "$LAST_PORT" ]; do
  if port_busy "$p"; then
    echo "port $p is in use, trying $((p + 1))"
  else
    rc=0
    try_run "$p" || rc=$?
    case "$rc" in
      0) port="$p"; break ;;
      1) ;;
      3)
        other_state="$(state_of "$NAME")"
        other_port="$(host_port "$NAME")"
        die "a container named $NAME appeared while this one was being created" \
            "(state: ${other_state:-unknown}${other_port:+, port $other_port}), probably from another start; it is" \
            "left as it is. Run ./run.sh again to check it, or ./run.sh stop to remove it"
        ;;
      *) die "starting the container failed (see above)" ;;
    esac
  fi
  p=$((p + 1))
done
[ -n "$port" ] || die "no free port from $FIRST_PORT to $LAST_PORT; free one, or try another range with --port N"

# -- wait until it answers -----------------------------------------------------------------------------------------
LEFT_RUNNING="the container is left running so you can look: docker logs $NAME; stop it with ./run.sh stop"

echo "started $NAME on port $port (backend $BACKEND, beam $BEAM, max depth $MAX_DEPTH); waiting for the server ..."
rc=0
wait_ready "$STARTED_ID" "$port" || rc=$?
case "$rc" in
  0) ;;
  1)
    die "the container stopped during start-up; to see why, run it in the foreground:
  docker run --rm --name $NAME --label $LABEL -p $port:8765 -e BACKEND=$BACKEND -e ASTAR_NODES=$ASTAR_NODES -e BEAM=$BEAM -e MAX_DEPTH=$MAX_DEPTH $IMAGE"
    ;;
  *) die "no answer on http://localhost:$port/solver after ${HEALTH_TIMEOUT} s; $LEFT_RUNNING" ;;
esac

# -- real solves, so "up" means the network answers, not only the web server ---------------------------------------
first_test_solve "$STARTED_ID" "$LEFT_RUNNING"
if [ "$MAX_DEPTH" -ge 6 ]; then
  rc=0
  answer="$(check_solve "$STARTED_ID" "$TEST2_STATE")" || rc=$?
  case "$rc" in
    0) echo "test solve OK: the network solved $TEST2_MOVES with $answer" ;;
    2) echo "warning: the second test cube ($TEST2_MOVES) was not solved: $answer. At beam $BEAM and max depth" \
            "$MAX_DEPTH some cubes are not solved; a larger --beam solves more" >&2 ;;
    3) echo "warning: the second test solve ($TEST2_MOVES) took more than ${SOLVE_TIMEOUT} s at beam $BEAM" \
            "($answer); the page gives up after 20 s, so restart with a smaller --beam (./run.sh stop;" \
            "./run.sh --beam 1024)" >&2 ;;
    *) die "the second test solve ($TEST2_MOVES) failed: $answer; $LEFT_RUNNING" ;;
  esac
fi

echo
echo "open http://localhost:$port/"
echo "(in the Solve panel pick \"Remote solver (POST /solve)\"; stop the app with ./run.sh stop)"
[ "$OPEN" = 1 ] && open_url "http://localhost:$port/"
exit 0
