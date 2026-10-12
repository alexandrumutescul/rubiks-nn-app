#!/usr/bin/env bash
# Start the Rubik's cube web app in Docker on the first free host port from 8765 to 8800 (macOS and Linux).
# Run ./run.sh --help for the options. Needs Docker; bash 3.2 or newer.
set -euo pipefail

NAME=rubiks-nn-app
LOCAL_IMAGE=rubiks-nn-app
PUBLISHED_IMAGE=ghcr.io/alexandrumutescul/rubiks-nn-app:latest
PORT_SPAN=35          # ports tried: FIRST .. FIRST + 35 (8765 .. 8800 by default)
HEALTH_TIMEOUT=60     # seconds to wait for GET /solver
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# "R U F' L2 D B'" applied to the solved cube: the post-start check asks the server to solve it
TEST_STATE='[0,2,2,3,0,0,3,1,1,5,5,3,3,1,0,4,2,0,5,3,3,5,2,2,2,2,2,0,0,0,3,3,4,2,1,5,5,4,4,5,4,4,4,5,1,1,1,1,4,5,1,4,0,3]'

usage() {
  cat <<USAGE
Usage: ./run.sh [start] [options]    start the app (the default command)
       ./run.sh stop                 stop it

Options:
  --image IMAGE     use this image instead of building one here, e.g.
                    --image $PUBLISHED_IMAGE
  --port N          first host port to try (default \$PORT or 8765); N .. N+$PORT_SPAN are tried in turn
  --beam N          beam width of the search (default \$BEAM or 1024; 256 is faster, 4096 finds shorter solutions)
  --max-depth N     longest solution the search tries (default \$MAX_DEPTH or 40)
  --open            open the page in the default browser once it is up
  -h, --help        this help

Without --image the image is built from this directory (docker build -t $LOCAL_IMAGE .); the first build downloads
the Python base image and the CPU build of PyTorch, later builds are cached.
USAGE
}

die() { echo "run.sh: $*" >&2; exit 1; }
is_uint() { case "$1" in '' | *[!0-9]*) return 1 ;; *) return 0 ;; esac; }

COMMAND=start
IMAGE=""
FIRST_PORT="${PORT:-8765}"
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
is_uint "$BEAM" && [ "$BEAM" -ge 1 ] || die "--beam must be a whole number >= 1, got '$BEAM'"
is_uint "$MAX_DEPTH" && [ "$MAX_DEPTH" -ge 1 ] || die "--max-depth must be a whole number >= 1, got '$MAX_DEPTH'"
LAST_PORT=$((FIRST_PORT + PORT_SPAN))

command -v docker >/dev/null 2>&1 || die "docker not found: install Docker Desktop (macOS) or Docker Engine (Linux)"
docker info >/dev/null 2>&1 ||
  die "cannot talk to Docker: is Docker Desktop (or the docker service) running? On Linux, is this user in the docker group?"

container_state() { docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || true; }

# the host port the running container publishes 8765 on ("0.0.0.0:8766" -> 8766)
host_port() { docker port "$NAME" 8765/tcp 2>/dev/null | sed -n '1s/.*://p' || true; }

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

if [ "$COMMAND" = stop ]; then
  state="$(container_state)"
  if [ -z "$state" ]; then
    echo "$NAME is not running"
  else
    docker stop "$NAME" >/dev/null 2>&1 || true
    docker rm -f "$NAME" >/dev/null 2>&1 || true   # normally already gone (--rm)
    echo "stopped $NAME"
  fi
  exit 0
fi

# -- already running? --------------------------------------------------------------------------------------------
state="$(container_state)"
if [ "$state" = running ]; then
  port="$(host_port)"
  [ -n "$port" ] || die "a container named $NAME is running but publishes no port; stop it with ./run.sh stop"
  echo "$NAME is already running: open http://localhost:$port/"
  echo "(stop it with ./run.sh stop)"
  [ "$OPEN" = 1 ] && open_url "http://localhost:$port/"
  exit 0
elif [ -n "$state" ]; then
  echo "removing a stopped container named $NAME (state: $state)"
  docker rm -f "$NAME" >/dev/null
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

# 0 = started; 1 = the port turned out to be taken (try the next one); 2 = any other failure (message printed)
try_run() {
  local p="$1" out
  if out="$(docker run --rm -d --name "$NAME" -p "$p:8765" -e BEAM="$BEAM" -e MAX_DEPTH="$MAX_DEPTH" "$IMAGE" 2>&1)"
  then
    return 0
  fi
  docker rm -f "$NAME" >/dev/null 2>&1 || true
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
    if [ "$rc" = 0 ]; then port="$p"; break; fi
    [ "$rc" = 1 ] || die "docker run failed (see above)"
  fi
  p=$((p + 1))
done
[ -n "$port" ] || die "no free port from $FIRST_PORT to $LAST_PORT; free one, or try another range with --port N"

# -- wait until it answers -----------------------------------------------------------------------------------------
http_get() {  # body of a GET, or a failure; status 2 when neither curl nor wget is installed
  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 3 "$1" 2>/dev/null
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- -T 3 "$1" 2>/dev/null
  else
    return 2
  fi
}

is_up() {
  local body rc=0
  body="$(http_get "http://127.0.0.1:$port/solver")" || rc=$?
  if [ "$rc" = 2 ]; then  # no HTTP client here: rely on the image's own health check
    [ "$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{end}}' "$NAME" 2>/dev/null)" = healthy ]
    return
  fi
  [ "$rc" = 0 ] && printf '%s' "$body" | grep '"backend": "policy"' >/dev/null
}

echo "started $NAME on port $port (beam $BEAM, max depth $MAX_DEPTH); waiting for the server ..."
deadline=$((SECONDS + HEALTH_TIMEOUT))
until is_up; do
  if [ "$(container_state)" != running ]; then
    die "the container stopped during start-up; to see why, run it in the foreground:
  docker run --rm -p $port:8765 -e BEAM=$BEAM -e MAX_DEPTH=$MAX_DEPTH $IMAGE"
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    die "no answer on http://localhost:$port/solver after ${HEALTH_TIMEOUT} s; see: docker logs $NAME"
  fi
  sleep 1
done

# -- one real solve, so "up" means the network answers, not only the web server --------------------------------------
if command -v curl >/dev/null 2>&1; then
  rc=0
  answer="$(curl -fsS --max-time 30 -H 'Content-Type: application/json' -d "{\"state\": $TEST_STATE}" \
            "http://127.0.0.1:$port/solve" 2>&1)" || rc=$?
  if [ "$rc" = 28 ]; then
    echo "warning: the test solve took more than 30 s at beam $BEAM; the page gives up after 20 s, so restart" \
         "with a smaller --beam (./run.sh stop; ./run.sh --beam 1024)" >&2
  elif [ "$rc" != 0 ]; then
    die "the test solve failed: $answer (see: docker logs $NAME)"
  elif printf '%s' "$answer" | grep '"verified": true' >/dev/null; then
    notation="$(printf '%s' "$answer" | sed -n 's/.*"notation": "\([^"]*\)".*/\1/p')"
    echo "test solve OK: the network solved R U F' L2 D B' with $notation"
  else
    die "the test solve did not verify: $answer"
  fi
fi

echo
echo "open http://localhost:$port/"
echo "(in the Solve panel pick \"Remote solver (POST /solve)\"; stop the app with ./run.sh stop)"
[ "$OPEN" = 1 ] && open_url "http://localhost:$port/"
exit 0
