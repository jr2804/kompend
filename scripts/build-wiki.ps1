# build-wiki.ps1 — Build the demo wiki for A/B model comparison
#
# Prerequisites:
#   - Ollama running with models pulled (see config files for model list)
#   - uv installed (https://docs.astral.sh/uv/)
#
# Usage:
#   .\scripts\build-wiki.ps1 scripts\config-cloud.toml     # enterprise LLM (~1M ctx)
#   .\scripts\build-wiki.ps1 scripts\config-local.toml      # local LLM (128K ctx)
#   .\scripts\build-wiki.ps1 scripts\config-cloud.toml -WhatIf
#
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$ConfigFile,

    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"

# ── Sync environment (ensure latest source is installed) ────────────────
Write-Host "=== Syncing environment (uv sync) ===" -ForegroundColor Cyan
$ProjectRoot = $PSScriptRoot | Split-Path -Parent

if (-not $WhatIf) {
    & uv sync -U --directory $ProjectRoot
    if ($LASTEXITCODE -ne 0) {
        Write-Error "uv sync failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

# ── Validate config ─────────────────────────────────────────────────────
$ConfigPath = Resolve-Path $ConfigFile -ErrorAction Stop
Write-Host "Using config: $ConfigPath" -ForegroundColor Cyan

# ── Read build settings from config ─────────────────────────────────────
# We parse the TOML to extract model settings. Since PowerShell
# doesn't have native TOML support, we use a simple regex approach for
# the specific keys we need.

function Get-ConfigValue($Key) {
    $lines = Get-Content $ConfigPath
    foreach ($line in $lines) {
        if ($line -match "^\s*$Key\s*=\s*""(.+)""") {
            return $matches[1]
        }
        elseif ($line -match "^\s*$Key\s*=\s*(.+)") {
            return $matches[1].Trim()
        }
    }
    return $null
}

$LLMModel    = Get-ConfigValue "default_model"
$EmbedModel  = Get-ConfigValue "embed_model"
# Derive output directory name from config filename:
#   scripts\config-cloud.toml  →  demo-wiki-cloud
#   scripts\config-local.toml   →  demo-wiki-local
$BaseName    = [System.IO.Path]::GetFileNameWithoutExtension($ConfigFile)
$OutputName  = $BaseName -replace "^config-", "demo-wiki-"
$OutputPath  = Join-Path $ProjectRoot $OutputName

if (-not $LLMModel) {
    Write-Error "Config missing [compile.llm] default_model"
    exit 1
}

Write-Host "  Model:  $LLMModel" -ForegroundColor DarkGray
Write-Host "  Embed:  $($EmbedModel)" -ForegroundColor DarkGray
Write-Host "  Output: $OutputPath" -ForegroundColor DarkGray

$InputFiles = Join-Path $ProjectRoot "tests/data/sources"
$UrlFile    = Join-Path $ProjectRoot "tests/data/sources/urls.txt"

# ── Clean ──────────────────────────────────────────────────────────────
# --force --keep-cache on knox init handles directory reset + cache
# preservation. The cache-skip logic in ingest already skips unchanged
# files (hash-based). No manual cache clearing needed.

# ── Stage 0: Init (scaffold project structure) ─────────────────────────
Write-Host ""
Write-Host "=== Stage 0: Initialize project ===" -ForegroundColor Cyan
$InitArgs = @(
    "uv", "run", "knox", "init",
    $OutputPath,
    "--bare",
    "--force",
    "--keep-cache"
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

# $Force no longer forces re-parse. The cache-skip logic in ingest
# already skips unchanged files (hash-based). To force a full re-parse,
# run: knox ingest --no-cache --wiki-dir <dir> manually before this script.

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

# ── Stage 1b: Validate LLM provider ──────────────────────────────────
Write-Host ""
Write-Host "=== Stage 1b: Validate LLM provider ===" -ForegroundColor Cyan
$DoctorArgs = @(
    "uv", "run", "knox",
    "--wiki-dir", $OutputPath,
    "doctor",
    "--verbose",
    "--config", $ConfigPath
)

Write-Host ($DoctorArgs -join " ") -ForegroundColor DarkGray
Write-Host ""

if (-not $WhatIf) {
    & $DoctorArgs[0] $DoctorArgs[1..($DoctorArgs.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "knox doctor reported issues — continuing build"
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
    "--vlm-model",        "",
    "--flavor",           "obsidian",
    "--compiler",         "wicer",
    "--workers",          "1",
    "--verify",
    "--index", "--synthesize", "--progress",
    "--config",           $ConfigPath
)

if ($EmbedModel) {
    $CompileArgs += @("--embed-model", $EmbedModel)
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

# ── Lint (Structural + LLM rules, saves to lint_history.jsonl) ─────────
Write-Host ""
Write-Host "=== Running Linter (Structural + LLM rules) ===" -ForegroundColor Cyan
$LintArgs = @(
    "uv", "run", "knox",
    "--wiki-dir", $OutputPath,
    "lint", "run",
    "--llm",
    "--llm-default-model", $LLMModel,
    "--fix",
    "--config", $ConfigPath
)

Write-Host ($LintArgs -join " ") -ForegroundColor DarkGray
Write-Host ""

if (-not $WhatIf) {
    & $LintArgs[0] $LintArgs[1..($LintArgs.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Linting reported issues (exit code $LASTEXITCODE)"
    }
}

# ── Gap Analysis (source recommendations from lint results) ───────────
Write-Host ""
Write-Host "=== Gap Analysis (source recommendations) ===" -ForegroundColor Cyan
$GapsArgs = @(
    "uv", "run", "knox",
    "--wiki-dir", $OutputPath,
    "lint", "gaps",
    "--config", $ConfigPath
)

Write-Host ($GapsArgs -join " ") -ForegroundColor DarkGray
Write-Host ""

if (-not $WhatIf) {
    & $GapsArgs[0] $GapsArgs[1..($GapsArgs.Length - 1)]
    # gaps exits 0 regardless — it's informational
}

# ── Remediate (auto-fix + LLM-generated concept pages) ────────────────
Write-Host ""
Write-Host "=== Remediate (auto-fix + concept generation) ===" -ForegroundColor Cyan
$RemediateArgs = @(
    "uv", "run", "knox",
    "--wiki-dir", $OutputPath,
    "lint", "remediate",
    "--llm-default-model", $LLMModel,
    "--config", $ConfigPath
)

Write-Host ($RemediateArgs -join " ") -ForegroundColor DarkGray
Write-Host ""

if (-not $WhatIf) {
    & $RemediateArgs[0] $RemediateArgs[1..($RemediateArgs.Length - 1)]
}

# ── Lint History (trend across builds) ────────────────────────────────
Write-Host ""
Write-Host "=== Lint History (trend) ===" -ForegroundColor Cyan
$HistoryArgs = @(
    "uv", "run", "knox",
    "--wiki-dir", $OutputPath,
    "lint", "history",
    "--last", "5",
    "--config", $ConfigPath
)

Write-Host ($HistoryArgs -join " ") -ForegroundColor DarkGray
Write-Host ""

if (-not $WhatIf) {
    & $HistoryArgs[0] $HistoryArgs[1..($HistoryArgs.Length - 1)]
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
