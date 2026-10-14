# start.ps1 -- Preflight the GPU, then launch the bge-m3 embedding service.
#
# ASCII-only on purpose: Windows PowerShell 5.1 reads BOM-less files as ANSI,
# so non-ASCII text here would come out as mojibake.

[CmdletBinding()]
param(
    [switch]$Force,                                        # auto-degrade instead of prompting
    [int]$Port = 8000,
    [string]$BindHost = "127.0.0.1",
    [string]$Python = "",                                  # empty -> .env.local -> env -> PATH
    [string]$EnvOut = ".env.runtime"
)

$ErrorActionPreference = "Stop"

# Repo root = parent of this script's directory
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Machine-local values live in the gitignored .env.local (see .env.example), so that
# per-machine absolute paths never end up in the repository.
# -Encoding UTF8 is required: these files are LF + UTF-8, and PowerShell 5.1 would
# otherwise decode them as ANSI, where a GBK double-byte pair can swallow the LF and
# merge every line into one - the key=value parse then silently returns nothing.
function Get-LocalValue([string]$key) {
    $file = Join-Path $root ".env.local"
    if (-not (Test-Path $file)) { return "" }
    foreach ($line in Get-Content -Encoding UTF8 -Path $file) {
        if ($line -match "^\s*$key\s*=\s*(.*)$") { return $matches[1].Trim() }
    }
    return ""
}

if (-not $Python) { $Python = Get-LocalValue "BGE_PYTHON" }
if (-not $Python) { $Python = $env:BGE_PYTHON }

if (-not $Python) {
    Write-Warning "No interpreter configured (.env.local BGE_PYTHON); falling back to 'python' on PATH."
    $Python = "python"
}
elseif (-not (Test-Path $Python)) {
    Write-Warning "Interpreter $Python not found, falling back to 'python' on PATH."
    $Python = "python"
}

# Fail fast if that interpreter is not the project env: the PATH fallback commonly
# lands on a base conda env, and the first symptom would otherwise be an obscure
# "No module named uvicorn" printed only after the whole VRAM preflight has run.
$eap = $ErrorActionPreference
$ErrorActionPreference = "Continue"   # native stderr must not become terminating here
& $Python -c "import fastapi, uvicorn"
$depExit = $LASTEXITCODE
$ErrorActionPreference = $eap
if ($depExit -ne 0) {
    Write-Host "ERROR: interpreter '$Python' cannot import the service dependencies. Set BGE_PYTHON in .env.local (see .env.example)." -ForegroundColor Red
    exit 3
}

# ── Port availability ──
# uvicorn binds only after the FastAPI lifespan finishes, i.e. after the model has
# been loaded. A port conflict would therefore burn ~10s and ~1.5GB of VRAM and then
# exit anyway. Probe first. ExclusiveAddressUse is required on Windows, where two
# sockets may otherwise bind the same port without error.
try {
    $probe = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Parse($BindHost), $Port)
    $probe.ExclusiveAddressUse = $true
    $probe.Start()
    $probe.Stop()
} catch {
    # Write-Host, not Write-Error: with $ErrorActionPreference = "Stop" the latter
    # throws and the script dies with exit 1 regardless of what follows.
    Write-Host "ERROR: port $Port on $BindHost is already in use. Use -Port <other> or stop the listener." -ForegroundColor Red
    exit 1
}

# ── Runtime environment ──
$env:CUDA_VISIBLE_DEVICES      = "0"
# HF_HOME keeps the 2.1GB weight cache on D: instead of the system drive.
$hfHome = Get-LocalValue "BGE_HF_HOME"
if (-not $hfHome) { $hfHome = Join-Path $root "models" }
$env:HF_HOME                   = $hfHome
$env:HF_HUB_DISABLE_TELEMETRY  = "1"
$env:PYTHONUTF8                = "1"
# NOTE: do NOT point HF_ENDPOINT at hf-mirror.com, and do NOT set
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
#  * hf-mirror answers LFS blobs with a 308 back to huggingface.co and drops the
#    X-Repo-Commit header; huggingface_hub will not follow a cross-host redirect,
#    so every large file fails with "Distant resource does not seem to be on
#    huggingface.co" while small JSON files succeed - the failure looks random.
#  * expandable_segments is a no-op on Windows (torch 2.3 warns "not supported on
#    this platform"), so it defends against nothing here. Fragmentation is instead
#    absorbed by the token-budget batching plus the OOM downgrade path.

# ── Gate: check VRAM before the model claims any ──
$preArgs = @("-m", "src.preflight", "--env-out", $EnvOut)
if ($Force) { $preArgs += "--force" }

& $Python @preArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: preflight failed (exit $LASTEXITCODE). Service not started." -ForegroundColor Red
    exit $LASTEXITCODE
}

# Preflight may have degraded the preset; uvicorn must inherit that decision.
# Explicit foreach: reading $matches from a ForEach-Object block while -match ran in a
# Where-Object block only works by accident of variable scoping.
$envFile = Join-Path $root $EnvOut
if (Test-Path $envFile) {
    foreach ($line in Get-Content -Encoding UTF8 -Path $envFile) {
        if ($line -match "^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$") {
            Set-Item -Path ("env:" + $matches[1]) -Value $matches[2].Trim()
            Write-Host "  inherited $($matches[1])=$($matches[2].Trim()) from $EnvOut"
        }
    }
}

# --workers 1 is mandatory: extra workers duplicate the model in VRAM.
Write-Host "Starting bge-m3 embedding service on http://${BindHost}:${Port} (single worker)..."
& $Python -m uvicorn src.api:app --host $BindHost --port $Port --workers 1
# Propagate the exit code: powershell -File otherwise reports 0 even when uvicorn
# died on startup, so a wrapper would treat a dead service as a successful start.
exit $LASTEXITCODE
