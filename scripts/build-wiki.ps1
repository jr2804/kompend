# build-wiki.ps1 — Build the demo wiki for A/B model comparison
#
# Prerequisites:
#   - Ollama running with models pulled (see config files for model list)
#   - uv installed (https://docs.astral.sh/uv/)
#
# Usage:
#   .\scripts\build-wiki.ps1 scripts\config-cloud.toml     # enterprise LLM (~1M ctx)
#   .\scripts\build-wiki.ps1 scripts\config-local.toml      # local LLM (128K ctx)
#   .\scripts\build-wiki.ps1 scripts\config-cloud.toml -Force
#   .\scripts\build-wiki.ps1 scripts\config-cloud.toml -WhatIf
#
# A/B testing:
#   Compare wiki structure, article count, and processing time between
#   the two configs. Quality differences due to model selection are out
#   of scope — we are testing the pipeline under different conditions.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$ConfigFile,

    [switch]$Force,
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"

# ── Validate config ─────────────────────────────────────────────────────
$ConfigPath = Resolve-Path $ConfigFile -ErrorAction Stop
Write-Host "Using config: $ConfigPath" -ForegroundColor Cyan

# ── Read build settings from config ─────────────────────────────────────
# We parse the TOML to extract [build] section values. Since PowerShell
# doesn't have native TOML support, we use a simple regex approach for
# the specific keys we need.

function Get-ConfigValue($Key) {
    $lines = Get-Content $ConfigPath
    $inSection = $false
    $section = ""
    foreach ($line in $lines) {
        if ($line -match '^\[(.+)\]') {
            $section = $matches[1]
            $inSection = $false
        }
        elseif ($line -match "^\s*$Key\s*=\s*""(.+)""") {
            return $matches[1]
        }
        elseif ($line -match "^\s*$Key\s*=\s*(.+)") {
            return $matches[1].Trim()
        }
    }
    return $null
}

$LLMModel    = Get-ConfigValue "default_model"
$VlmModel    = Get-ConfigValue "vlm_model"
$EmbedModel  = Get-ConfigValue "embed_model"
$OutputName  = Get-ConfigValue "output_name"
$ProjectRoot = $PSScriptRoot | Split-Path -Parent
$OutputPath  = Join-Path $ProjectRoot $OutputName

if (-not $LLMModel) {
    Write-Error "Config missing [compile.llm] default_model"
    exit 1
}
if (-not $OutputName) {
    Write-Error "Config missing [build] output_name"
    exit 1
}

Write-Host "  Model:  $LLMModel" -ForegroundColor DarkGray
if ($VlmModel)  { Write-Host "  VLM:    $VlmModel" -ForegroundColor DarkGray }
Write-Host "  Embed:  $EmbedModel" -ForegroundColor DarkGray
Write-Host "  Output: $OutputPath" -ForegroundColor DarkGray

$InputFiles = Join-Path $ProjectRoot "tests/data/sources"
$UrlFile    = Join-Path $ProjectRoot "tests/data/urls.txt"

# ── Clean (if -Force) ──────────────────────────────────────────────────
if ($Force) {
    if (Test-Path $OutputPath) {
        Write-Host "Removing $OutputPath" -ForegroundColor Yellow
        if (-not $WhatIf) { Remove-Item -Recurse -Force $OutputPath }
    }
}

# ── Stage 0: Init (scaffold project structure) ─────────────────────────
Write-Host ""
Write-Host "=== Stage 0: Initialize project ===" -ForegroundColor Cyan
$InitArgs = @(
    "uv", "run", "knox", "init",
    $OutputPath,
    "--bare",
    "--force"
)

Write-Host ""
Write-Host ($InitArgs -join " ") -ForegroundColor DarkGray
Write-Host ""

if (-not $WhatIf) {
    & $InitArgs[0] $InitArgs[1..($InitArgs.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        Write-Error "knox init failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

# ── Stage 1: Ingest (T0+T1: download + parse) ────────────────────────
Write-Host ""
Write-Host "=== Stage 1: Ingest (download + parse) ===" -ForegroundColor Cyan
$IngestArgs = @(
    "uv", "run", "knox",
    "--wiki-dir",    $OutputPath,
    "ingest",
    $InputFiles,
    "--url-file",    $UrlFile,
    "--workers",     "1",
    "--config",      $ConfigPath
)

if ($Force) {
    $IngestArgs += "--force"
    $IngestArgs += "--no-cache"
}

Write-Host ""
Write-Host ($IngestArgs -join " ") -ForegroundColor DarkGray
Write-Host ""

if (-not $WhatIf) {
    & $IngestArgs[0] $IngestArgs[1..($IngestArgs.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        Write-Error "knox ingest failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

# ── Stage 2: Compile (T2+T3+T4+T5: chunk + resolve + compile + output) ─
Write-Host ""
Write-Host "=== Stage 2: Compile (chunk + resolve + compile + output) ===" -ForegroundColor Cyan
$CompileArgs = @(
    "uv", "run", "knox",
    "--wiki-dir",         $OutputPath,
    "compile",
    $InputFiles,
    "--start-at-tier",    "chunk",
    "--llm-default-model", $LLMModel,
    "--embed-model",      $EmbedModel,
    "--resolve-relations",
    "--flavor",           "obsidian",
    "--compiler",         "wicer",
    "--workers",          "1",
    "--verify",
    "--index", "--synthesize", "--progress",
    "--config",           $ConfigPath
)

if ($VlmModel) {
    $CompileArgs += @("--vlm-model", $VlmModel)
}

Write-Host ""
Write-Host ($CompileArgs -join " ") -ForegroundColor DarkGray
Write-Host ""

if (-not $WhatIf) {
    & $CompileArgs[0] $CompileArgs[1..($CompileArgs.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        Write-Error "knox compile failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

# ── Lint (Step 6) ──────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Running Linter (Structural + LLM rules) ===" -ForegroundColor Cyan
$LintArgs = @(
    "uv", "run", "knox",
    "--wiki-dir", $OutputPath,
    "lint",
    "--llm",
    "--llm-default-model", $LLMModel,
    "--fix",
    "--threshold", "80",
    "--config", $ConfigPath
)

Write-Host ($LintArgs -join " ") -ForegroundColor DarkGray
Write-Host ""

if (-not $WhatIf) {
    & $LintArgs[0] $LintArgs[1..($LintArgs.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Linting reported issues or health score fell below 80% (exit code $LASTEXITCODE)"
    }
}

# ── Done ───────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Build complete ===" -ForegroundColor Green
if (-not $WhatIf -and (Test-Path $OutputPath)) {
    $MdFiles = Get-ChildItem -Path $OutputPath -Filter "*.md" -Recurse
    Write-Host "  Output: $OutputPath ($($MdFiles.Count) markdown file(s))"
    $Index = Join-Path $OutputPath "index.md"
    if (Test-Path $Index) {
        Write-Host "  Entry:  $Index" -ForegroundColor Cyan
    }
}
