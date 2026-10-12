<#
.SYNOPSIS
  Start the Rubik's cube web app in Docker on the first free host port from 8765 to 8800 (Windows PowerShell 5.1 or
  PowerShell 7 on any OS).

.DESCRIPTION
  Builds the image from this directory (or uses -Image), finds a free host port, runs the container
  (docker run --rm -d --name rubiks-nn-app -p <port>:8765 ...), waits until GET /solver answers (up to 60 s), checks
  one real solve and prints the address. If the app is already running it prints its address instead.

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

    # beam width of the search (256 is faster, 4096 finds shorter solutions)
    [ValidateRange(1, 1000000)]
    [int]$Beam = 1024,

    # longest solution the search tries, in moves
    [ValidateRange(1, 1000)]
    [int]$MaxDepth = 40,

    # open the page in the default browser once it is up
    [switch]$Open
)

# native commands report failure through $LASTEXITCODE, checked after each call
$ErrorActionPreference = 'Continue'

$Name = 'rubiks-nn-app'
$LocalImage = 'rubiks-nn-app'
$PortSpan = 35
$HealthTimeout = 60
# "R U F' L2 D B'" applied to the solved cube: the post-start check asks the server to solve it
$TestState = '[0,2,2,3,0,0,3,1,1,5,5,3,3,1,0,4,2,0,5,3,3,5,2,2,2,2,2,0,0,0,3,3,4,2,1,5,5,4,4,5,4,4,4,5,1,1,1,1,4,5,1,4,0,3]'

function Fail([string]$Message) {
    Write-Host "run.ps1: $Message" -ForegroundColor Red
    exit 1
}

function Get-ContainerState {
    $state = & docker inspect -f '{{.State.Status}}' $Name 2>$null
    if ($LASTEXITCODE -ne 0) { return '' }
    return ([string]$state).Trim()
}

function Get-HostPort {
    # "0.0.0.0:8766" -> 8766
    $lines = @(& docker port $Name 8765/tcp 2>$null)
    if ($LASTEXITCODE -ne 0 -or $lines.Count -eq 0) { return '' }
    return ([string]$lines[0]).Split(':')[-1].Trim()
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
        return ($info.backend -eq 'policy')
    } catch { return $false }
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Fail 'docker not found: install Docker Desktop'
}
& docker info *> $null
if ($LASTEXITCODE -ne 0) { Fail 'cannot talk to Docker: is Docker Desktop running?' }

if ($Command -eq 'stop') {
    if ((Get-ContainerState) -eq '') {
        Write-Host "$Name is not running"
    } else {
        & docker stop $Name *> $null
        & docker rm -f $Name *> $null   # normally already gone (--rm)
        Write-Host "stopped $Name"
    }
    exit 0
}

# -- already running? ------------------------------------------------------------------------------------------------
$state = Get-ContainerState
if ($state -eq 'running') {
    $running = Get-HostPort
    if ($running -eq '') { Fail "a container named $Name is running but publishes no port; stop it with .\run.ps1 stop" }
    Write-Host "$Name is already running: open http://localhost:$running/"
    Write-Host '(stop it with .\run.ps1 stop)'
    if ($Open) { Open-Url "http://localhost:$running/" }
    exit 0
} elseif ($state -ne '') {
    Write-Host "removing a stopped container named $Name (state: $state)"
    & docker rm -f $Name *> $null
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
$chosen = 0
for ($p = $Port; $p -le $Port + $PortSpan; $p++) {
    if (Test-PortBusy $p) {
        Write-Host "port $p is in use, trying $($p + 1)"
        continue
    }
    $out = & docker run --rm -d --name $Name -p "${p}:8765" -e "BEAM=$Beam" -e "MAX_DEPTH=$MaxDepth" $Image 2>&1
    if ($LASTEXITCODE -eq 0) { $chosen = $p; break }
    $text = ($out | Out-String)
    & docker rm -f $Name *> $null
    if ($text -match 'port is already allocated|address already in use|ports are not available|bind for|forbidden by its access permissions') {
        Write-Host "port $p is taken after all; trying the next one"
        continue
    }
    Write-Host $text
    Fail 'docker run failed (see above)'
}
if ($chosen -eq 0) {
    Fail "no free port from $Port to $($Port + $PortSpan); free one, or try another range with -Port N"
}

# -- wait until it answers ---------------------------------------------------------------------------------------------
Write-Host "started $Name on port $chosen (beam $Beam, max depth $MaxDepth); waiting for the server ..."
$deadline = (Get-Date).AddSeconds($HealthTimeout)
while (-not (Test-Up $chosen)) {
    if ((Get-ContainerState) -ne 'running') {
        Fail ("the container stopped during start-up; to see why, run it in the foreground:`n" +
              "  docker run --rm -p ${chosen}:8765 -e BEAM=$Beam -e MAX_DEPTH=$MaxDepth $Image")
    }
    if ((Get-Date) -gt $deadline) {
        Fail "no answer on http://localhost:$chosen/solver after $HealthTimeout s; see: docker logs $Name"
    }
    Start-Sleep -Seconds 1
}

# -- one real solve, so "up" means the network answers, not only the web server ----------------------------------------
try {
    $answer = Invoke-RestMethod -Uri "http://127.0.0.1:$chosen/solve" -Method Post -ContentType 'application/json' `
        -Body "{`"state`": $TestState}" -TimeoutSec 30 -UseBasicParsing
    if ($answer.verified -ne $true) { Fail "the test solve did not verify: $($answer | ConvertTo-Json -Compress)" }
    Write-Host "test solve OK: the network solved R U F' L2 D B' with $($answer.notation)"
} catch {
    Write-Host "warning: the test solve failed or took more than 30 s ($($_.Exception.Message)); see: docker logs $Name" `
        -ForegroundColor Yellow
}

Write-Host ''
Write-Host "open http://localhost:$chosen/"
Write-Host '(in the Solve panel pick "Remote solver (POST /solve)"; stop the app with .\run.ps1 stop)'
if ($Open) { Open-Url "http://localhost:$chosen/" }
exit 0
