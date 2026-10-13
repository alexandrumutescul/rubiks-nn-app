<#
.SYNOPSIS
  Start the Rubik's cube web app in Docker on the first free host port from 8765 to 8800 (Windows PowerShell 5.1 or
  PowerShell 7 on any OS).

.DESCRIPTION
  Builds the image from this directory (or uses -Image), finds a free host port, creates and starts the container
  (docker create --rm --name rubiks-nn-app --label app=rubiks-nn-app -p <port>:8765 ..., then docker start) and waits
  until GET /solver answers (up to 60 s). Then the container's own Python asks the server to solve a cube one turn from
  solved (and, with -MaxDepth 6 or more, one six moves from solved). The start counts as a success only if that
  answer comes back verified; otherwise the script exits with an error and leaves the container running for
  docker logs rubiks-nn-app. If the app is already running (a container named rubiks-nn-app or labelled
  app=rubiks-nn-app), that copy gets the same checks (the first test cube only) and its address is printed instead.
  BACKEND supplies the default backend when -Backend is omitted. ASTAR_NODES is the positive integer A* expanded-node
  cap for hybrid (default 200000).

.PARAMETER Backend
  policy or hybrid (default $env:BACKEND or policy).

.PARAMETER Help
  Show usage, including the ASTAR_NODES environment setting, and exit.

.EXAMPLE
  .\run.ps1
.EXAMPLE
  .\run.ps1 -Open
.EXAMPLE
  .\run.ps1 -Image ghcr.io/alexandrumutescul/rubiks-nn-app:latest
.EXAMPLE
  .\run.ps1 -Beam 256
.EXAMPLE
  .\run.ps1 stop

.NOTES
  If scripts are blocked by the execution policy: powershell -ExecutionPolicy Bypass -File .\run.ps1
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('start', 'stop')]
    [string]$Command = 'start',

    # use this image instead of building one here
    [string]$Image = '',

    # first host port to try; Port .. Port+35 are tried in turn
    [ValidateRange(1, 65500)]
    [int]$Port = 8765,

    # solver backend
    [string]$Backend = 'policy',

    # beam width of the search (256 is faster, 4096 finds shorter solutions)
    [ValidateRange(1, 1000000)]
    [int]$Beam = 1024,

    # longest solution the search tries, in moves
    [ValidateRange(1, 1000)]
    [int]$MaxDepth = 40,

    # open the page in the default browser once it is up
    [switch]$Open,

    [switch]$Help
)

if ($Help) {
    Write-Output @'
Usage: .\run.ps1 [start] [-Image IMAGE] [-Port N] [-Backend policy|hybrid] [-Beam N] [-MaxDepth N] [-Open]
       .\run.ps1 stop
       .\run.ps1 -Help
Backend defaults to $env:BACKEND or policy; Beam defaults to 1024 and MaxDepth to 40.
$env:ASTAR_NODES must be a positive integer A* expanded-node cap for hybrid (default 200000).
'@
    exit 0
}

# native commands report failure through $LASTEXITCODE, checked after each call
$ErrorActionPreference = 'Continue'

$Name = 'rubiks-nn-app'
$Label = 'app=rubiks-nn-app'   # on every launch; the app's container is found by its name or by this label
$LocalImage = 'rubiks-nn-app'
$PortSpan = 35
$HealthTimeout = 60
$AstarNodes = if ($null -ne $env:ASTAR_NODES) { $env:ASTAR_NODES } else { '200000' }
$SolveTimeout = 30   # seconds each test solve may take
$StopTimeout = 10    # seconds "stop" waits, after docker stop and docker rm, for the app's containers to be gone
# The start-up test cubes, as states (54 colours). The first is one turn from solved, so it must come back solved at
# any -MaxDepth; the second, six moves from solved, is checked as well when -MaxDepth is 6 or more.
$TestMoves = 'R'
$TestState = '[0,0,2,0,0,2,0,0,2,1,1,1,1,1,1,1,1,1,2,2,3,2,2,3,2,2,3,3,3,5,3,3,5,3,3,5,4,4,4,4,4,4,4,4,4,0,5,5,0,5,5,0,5,5]'
$Test2Moves = "R U F' L2 D B'"
$Test2State = '[0,2,2,3,0,0,3,1,1,5,5,3,3,1,0,4,2,0,5,3,3,5,2,2,2,2,2,0,0,0,3,3,4,2,1,5,5,4,4,5,4,4,4,5,1,1,1,1,4,5,1,4,0,3]'

# One POST /solve, run by the container's own Python (always in the image, unlike an HTTP client on the host):
#   docker exec -i <container> python - '<state>' <seconds>    with this program on stdin.
# It prints one line and exits 0 when the answer has "verified": true (the server's own check that the moves solve
# the cube), 2 for an HTTP error, 3 for no answer within the time limit, 4 for an answer that is not verified. Any
# other status (1: Python failed; 125 to 127: docker exec failed) means the check itself could not run.
$SolveCheck = @'
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
'@

function Fail([string]$Message) {
    Write-Host "run.ps1: $Message" -ForegroundColor Red
    exit 1
}

$BackendSource = '-Backend'
if (-not $PSBoundParameters.ContainsKey('Backend')) {
    $BackendSource = '$env:BACKEND'
    if ($env:BACKEND) { $Backend = $env:BACKEND }
}
if ($Backend -cnotin @('policy', 'hybrid')) {
    Fail "$BackendSource must be policy or hybrid, got '$Backend'"
}
if ($AstarNodes -notmatch '\A0*[1-9][0-9]*\z') {
    Fail "`$env:ASTAR_NODES must be a positive integer, got '$AstarNodes'"
}

function Get-ContainerState([string]$Ref) {
    $state = & docker container inspect -f '{{.State.Status}}' $Ref 2>$null
    if ($LASTEXITCODE -ne 0) { return '' }
    return ([string]$state).Trim()
}

function Get-HostPort([string]$Ref) {
    # "0.0.0.0:8766" -> 8766
    $lines = @(& docker port $Ref 8765/tcp 2>$null)
    if ($LASTEXITCODE -ne 0 -or $lines.Count -eq 0) { return '' }
    return ([string]$lines[0]).Split(':')[-1].Trim()
}

function Get-AppContainers {
    # The app's containers: the one named $Name and any labelled $Label (Docker Compose, or a docker run typed by
    # hand), each once, as objects with Id, State and Name.
    $ids = @()
    $byName = & docker container inspect -f '{{.Id}}' $Name 2>$null
    if ($LASTEXITCODE -eq 0) { $ids += @($byName) }
    $byLabel = & docker ps -a -q --no-trunc --filter "label=$Label" 2>$null
    if ($LASTEXITCODE -eq 0) { $ids += @($byLabel) }
    $seen = @{}
    foreach ($raw in $ids) {
        $id = ([string]$raw).Trim()
        if ($id -eq '' -or $seen.ContainsKey($id)) { continue }
        $seen[$id] = $true
        $info = & docker container inspect -f '{{.State.Status}} {{.Name}}' $id 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $info) { continue }
        $parts = @(([string]$info).Trim() -split ' ', 2)
        $containerName = if ($parts.Count -gt 1) { $parts[1].TrimStart('/') } else { $id }
        [pscustomobject]@{ Id = $id; State = $parts[0]; Name = $containerName }
    }
}

function Get-AppIdsLeft {
    # The ids of the app's containers that still exist (named $Name or labelled $Label, in any state), each once, as
    # @{ Ok; Ids }. Ok is $false when Docker could not be asked.
    $byLabel = @(& docker ps -a -q --no-trunc --filter "label=$Label" 2>$null)
    if ($LASTEXITCODE -ne 0) { return @{ Ok = $false; Ids = @() } }
    $all = @(& docker ps -a --no-trunc --format '{{.ID}} {{.Names}}' 2>$null)
    if ($LASTEXITCODE -ne 0) { return @{ Ok = $false; Ids = @() } }
    $ids = @()
    foreach ($line in $all) {
        $parts = @(([string]$line).Trim() -split ' ', 2)
        if ($parts.Count -eq 2 -and $parts[1] -ceq $Name -and $ids -notcontains $parts[0]) { $ids += $parts[0] }
    }
    foreach ($raw in $byLabel) {
        $id = ([string]$raw).Trim()
        if ($id -ne '' -and $ids -notcontains $id) { $ids += $id }
    }
    return @{ Ok = $true; Ids = $ids }
}

function Open-Url([string]$Url) {
    try { Start-Process $Url } catch { Write-Host "(could not open a browser; open the address above by hand)" }
}

function Test-PortBusy([int]$P) {
    # any listener on this machine
    try {
        $listeners = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
        foreach ($ep in $listeners) { if ($ep.Port -eq $P) { return $true } }
    } catch { }
    # something answers on the loopback address
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $task = $client.ConnectAsync('127.0.0.1', $P)
        if ($task.Wait(500) -and $client.Connected) { return $true }
    } catch { } finally { $client.Close() }
    # published by another container
    $published = @(& docker ps --format '{{.Ports}}' 2>$null)
    foreach ($line in $published) { if ([string]$line -match "(^|[ ,:])${P}->") { return $true } }
    return $false
}

function Test-Up([int]$P) {
    try {
        $info = Invoke-RestMethod -Uri "http://127.0.0.1:$P/solver" -TimeoutSec 3 -UseBasicParsing
        $script:ReadyBackend = [string]$info.backend
        return ($script:ReadyBackend -ne '')
    } catch { return $false }
}

function Invoke-SolveCheck([string]$Id, [string]$State) {
    # one POST /solve from inside the container: Code = the checker's exit status (above), Text = its one-line result
    $out = $SolveCheck | & docker exec -i $Id python - $State $SolveTimeout 2>&1
    $code = $LASTEXITCODE
    return @{ Code = $code; Text = ($out | Out-String).Trim() }
}

# Checks of a running container: after a start, and for a copy that is already running.
function Wait-Ready([string]$Id, [int]$P) {
    # waits up to $HealthTimeout s until GET /solver on host port $P answers: 'up', 'stopped' (the container is no
    # longer running) or 'timeout'
    $deadline = (Get-Date).AddSeconds($HealthTimeout)
    while (-not (Test-Up $P)) {
        if ((Get-ContainerState $Id) -ne 'running') { return 'stopped' }
        if ((Get-Date) -gt $deadline) { return 'timeout' }
        Start-Sleep -Seconds 1
    }
    return 'up'
}

function Invoke-FirstTestSolve([string]$Id, [string]$Note) {
    # the mandatory test solve ($TestMoves): exits with an error that ends in $Note unless the answer comes back
    # verified
    $check = Invoke-SolveCheck $Id $TestState
    switch ($check.Code) {
        0 { Write-Host "test solve OK: the network solved $TestMoves with $($check.Text)" }
        3 { Fail "the test solve ($TestMoves) got no answer within $SolveTimeout s: $($check.Text); $Note" }
        default { Fail "the test solve ($TestMoves) failed: $($check.Text); $Note" }
    }
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Fail 'docker not found: install Docker Desktop'
}
& docker info *> $null
if ($LASTEXITCODE -ne 0) { Fail 'cannot talk to Docker: is Docker Desktop running?' }

if ($Command -eq 'stop') {
    $found = @(Get-AppContainers)
    $failures = @()
    foreach ($c in $found) {
        # docker stop and docker rm also fail for a container that is already gone or already being removed (--rm),
        # so a failure counts only if the container is still there afterwards (checked below)
        $out = & docker stop $c.Id 2>&1
        if ($LASTEXITCODE -ne 0) {
            $failures += "docker stop $($c.Name): " + ((@($out) | ForEach-Object { "$_".Trim() }) -join ' ')
        }
        $out = & docker rm -f $c.Id 2>&1
        if ($LASTEXITCODE -ne 0) {
            $failures += "docker rm -f $($c.Name): " + ((@($out) | ForEach-Object { "$_".Trim() }) -join ' ')
        }
    }
    # No container of the app may be left, by name or label. Removal by --rm can lag docker stop by a moment.
    $deadline = (Get-Date).AddSeconds($StopTimeout)
    while ($true) {
        $left = Get-AppIdsLeft
        if ($left.Ok -and $left.Ids.Count -eq 0) { break }
        if ((Get-Date) -gt $deadline) {
            $said = ''
            if ($failures.Count -gt 0) { $said = "`ndocker said:`n  " + ($failures -join "`n  ") }
            if (-not $left.Ok) {
                Fail ("cannot check that the app has stopped (docker ps failed); check with " +
                      "docker ps -a --filter label=$Label$said")
            }
            $list = @(foreach ($leftId in $left.Ids) {
                $info = & docker container inspect -f '{{.Name}} (state: {{.State.Status}})' $leftId 2>$null
                if ($LASTEXITCODE -eq 0 -and $info) { ([string]$info).Trim().TrimStart('/') } else { $leftId }
            })
            Fail ("a container of this app is still there after $StopTimeout s:`n  " + ($list -join "`n  ") + $said +
                  "`nremove it by hand (docker rm -f <name>) and run .\run.ps1 stop again")
        }
        Start-Sleep -Seconds 1
    }
    if ($found.Count -eq 0) {
        Write-Host "$Name is not running"
    } else {
        foreach ($c in $found) { Write-Host "stopped $($c.Name)" }
    }
    exit 0
}

# -- already running? ------------------------------------------------------------------------------------------------
# A running copy gets the same checks as a new start (GET /solver answers, then the first test cube comes back
# verified) before its address is printed; if they fail, the container is left as it is.
$found = @(Get-AppContainers)
$running = @($found | Where-Object { $_.State -eq 'running' })
if ($running.Count -gt 0) {
    $c = $running[0]
    $runningPort = Get-HostPort $c.Id
    if ($runningPort -eq '') {
        Fail "$($c.Name) (a container of this app) is running but publishes no port; stop it with .\run.ps1 stop"
    }
    $leftAsIs = "$($c.Name) is left as it is, so you can look: docker logs $($c.Name); stop it with .\run.ps1 stop"
    Write-Host "$($c.Name) is already running on port $runningPort; checking that it answers ..."
    $ready = Wait-Ready $c.Id $runningPort
    if ($ready -eq 'stopped') {
        $st = Get-ContainerState $c.Id
        if ($st -ne '') { $st = " (state: $st)" }
        Fail ("$($c.Name) stopped while this script waited for it to answer$st; docker logs $($c.Name) shows why " +
              'while the container exists. Run .\run.ps1 again to start the app')
    }
    if ($ready -eq 'timeout') {
        Fail "$($c.Name) gives no answer on http://localhost:$runningPort/solver after $HealthTimeout s; $leftAsIs"
    }
    if ($script:ReadyBackend -cne $Backend) {
        Fail "running with backend $script:ReadyBackend; .\run.ps1 stop first"
    }
    Invoke-FirstTestSolve $c.Id $leftAsIs
    Write-Host "$($c.Name) is already running: open http://localhost:$runningPort/"
    Write-Host '(stop it with .\run.ps1 stop)'
    if ($Open) { Open-Url "http://localhost:$runningPort/" }
    exit 0
}
# None is running. Containers that have exited are cleared away, by id. Any other state (created, restarting, paused)
# can be another start in progress, so that container is left alone and reported.
foreach ($c in $found) {
    if ($c.State -eq 'exited' -or $c.State -eq 'dead') {
        Write-Host "removing a stopped container of this app: $($c.Name) (state: $($c.State))"
        & docker rm -f $c.Id *> $null
    } else {
        Fail ("$($c.Name) (a container of this app) is $($c.State); another start may be under way. Run " +
              '.\run.ps1 again in a moment, or remove it with .\run.ps1 stop')
    }
}

# -- the image ---------------------------------------------------------------------------------------------------------
if ($Image -eq '') {
    $Image = $LocalImage
    Write-Host "building the image $Image from $PSScriptRoot (cached after the first build) ..."
    & docker build -t $Image $PSScriptRoot
    if ($LASTEXITCODE -ne 0) { Fail 'docker build failed (see above)' }
} else {
    & docker image inspect $Image *> $null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "using the local image $Image"
    } else {
        Write-Host "pulling $Image ..."
        & docker pull $Image
        if ($LASTEXITCODE -ne 0) {
            Fail "cannot pull $Image (offline, or the image is not public?); leave out -Image to build it here instead"
        }
    }
}

# -- a free port -------------------------------------------------------------------------------------------------------
# The container is created, then started, so its id is known before anything can fail. After a failed start only that
# id is removed, never a container found by name: another start may own the name meanwhile.
$chosen = 0
$started = ''
for ($p = $Port; $p -le $Port + $PortSpan; $p++) {
    if (Test-PortBusy $p) {
        Write-Host "port $p is in use, trying $($p + 1)"
        continue
    }
    $out = & docker create --rm --name $Name --label $Label -p "${p}:8765" -e "BACKEND=$Backend" -e "ASTAR_NODES=$AstarNodes" -e "BEAM=$Beam" -e "MAX_DEPTH=$MaxDepth" `
        $Image 2>&1
    $code = $LASTEXITCODE
    $text = ($out | Out-String)
    if ($code -ne 0) {
        if ($text -match 'already in use by') {
            $otherState = Get-ContainerState $Name
            $otherPort = Get-HostPort $Name
            if ($otherState -eq '') { $otherState = 'unknown' }
            if ($otherPort -ne '') { $otherState += ", port $otherPort" }
            Fail ("a container named $Name appeared while this one was being created (state: $otherState), " +
                  'probably from another start; it is left as it is. Run .\run.ps1 again to check it, or ' +
                  '.\run.ps1 stop to remove it')
        }
        Write-Host $text
        Fail 'creating the container failed (see above)'
    }
    $id = ''
    foreach ($line in @($out)) {
        $candidate = ([string]$line).Trim()
        if ($candidate -match '^[0-9a-f]{64}$') { $id = $candidate }
    }
    if ($id -eq '') {
        Write-Host $text
        Fail 'docker create printed no container id (see above)'
    }
    $out = & docker start $id 2>&1
    if ($LASTEXITCODE -eq 0) { $chosen = $p; $started = $id; break }
    $text = ($out | Out-String)
    & docker rm -f $id *> $null   # this run's own container (a failed start can leave it behind)
    if ($text -match 'port is already allocated|address already in use|ports are not available|bind for|forbidden by its access permissions') {
        Write-Host "port $p is taken after all; trying the next one"
        continue
    }
    Write-Host $text
    Fail 'starting the container failed (see above)'
}
if ($chosen -eq 0) {
    Fail "no free port from $Port to $($Port + $PortSpan); free one, or try another range with -Port N"
}

# -- wait until it answers ---------------------------------------------------------------------------------------------
$leftRunning = "the container is left running so you can look: docker logs $Name; stop it with .\run.ps1 stop"
Write-Host "started $Name on port $chosen (backend $Backend, beam $Beam, max depth $MaxDepth); waiting for the server ..."
$ready = Wait-Ready $started $chosen
if ($ready -eq 'stopped') {
    Fail ("the container stopped during start-up; to see why, run it in the foreground:`n" +
          "  docker run --rm --name $Name --label $Label -p ${chosen}:8765 " +
          "-e BACKEND=$Backend -e ASTAR_NODES=$AstarNodes -e BEAM=$Beam -e MAX_DEPTH=$MaxDepth $Image")
}
if ($ready -eq 'timeout') {
    Fail "no answer on http://localhost:$chosen/solver after $HealthTimeout s; $leftRunning"
}

# -- real solves, so "up" means the network answers, not only the web server -------------------------------------------
Invoke-FirstTestSolve $started $leftRunning
if ($MaxDepth -ge 6) {
    $check = Invoke-SolveCheck $started $Test2State
    switch ($check.Code) {
        0 { Write-Host "test solve OK: the network solved $Test2Moves with $($check.Text)" }
        2 {
            Write-Host ("warning: the second test cube ($Test2Moves) was not solved: $($check.Text). At beam $Beam " +
                        "and max depth $MaxDepth some cubes are not solved; a larger -Beam solves more") `
                -ForegroundColor Yellow
        }
        3 {
            Write-Host ("warning: the second test solve ($Test2Moves) took more than $SolveTimeout s at beam $Beam " +
                        "($($check.Text)); the page gives up after 20 s, so restart with a smaller -Beam " +
                        "(.\run.ps1 stop; .\run.ps1 -Beam 1024)") -ForegroundColor Yellow
        }
        default { Fail "the second test solve ($Test2Moves) failed: $($check.Text); $leftRunning" }
    }
}

Write-Host ''
Write-Host "open http://localhost:$chosen/"
Write-Host '(in the Solve panel pick "Remote solver (POST /solve)"; stop the app with .\run.ps1 stop)'
if ($Open) { Open-Url "http://localhost:$chosen/" }
exit 0
