# build-wiki.ps1 — Build the demo wiki
#
# Prerequisites:
#   - Ollama running with models pulled:
#       ollama pull deepseek-v4-flash:cloud
#       ollama pull glm-ocr
#       ollama pull granite-embedding:278m
#   - uv installed (https://docs.astral.sh/uv/)
#
# Usage:
#   .\build-wiki.ps1                            # full build (defaults to wicer compiler, obsidian flavor)
#   .\build-wiki.ps1 -OutputPath "custom-wiki"  # build to a specific directory
#   .\build-wiki.ps1 -Compiler "graph"          # use the heading-based graph compiler instead of wicer
#   .\build-wiki.ps1 -Force                     # clear caches and recompile from scratch
#   .\build-wiki.ps1 -WhatIf                    # show commands without executing
#
# Two-stage flow:
#   Stage 1 (ingest):  knox ingest  — T0+T1: parse + chunk (free, no LLM)
#   Stage 2 (compile): knox compile --start-at-tier resolve — T2+T3+T4: resolve + compile + output

[CmdletBinding()]
param(
    [string]$OutputPath = "demo-wiki",
    [string]$Compiler = "wicer",
    [switch]$Force,
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"

# ── Configuration ──────────────────────────────────────────────────────
$ProjectRoot = $PSScriptRoot
$BaseDir     = Join-Path $ProjectRoot $OutputPath

$LLMModel   = "ollama/deepseek-v4-flash:cloud"
$EmbedModel = "litellm:ollama/granite-embedding:278m"
$EmbedDim   = 768

$SourcesDir = Join-Path $BaseDir "sources"
$OutputDir  = Join-Path $BaseDir "wiki"
$InputFiles = Join-Path $ProjectRoot "tests/data/sources"
$UrlFile    = Join-Path $ProjectRoot "tests/data/urls.txt"

# ── Clean (if -Force) ──────────────────────────────────────────────────
if ($Force) {
    if (Test-Path $BaseDir) {
        Write-Host "Removing $BaseDir" -ForegroundColor Yellow
        if (-not $WhatIf) { Remove-Item -Recurse -Force $BaseDir }
    }
}

# ── Stage 1: Ingest (T0+T1: parse + chunk) ─────────────────────────────
Write-Host ""
Write-Host "=== Stage 1: Ingest (parse + chunk) ===" -ForegroundColor Cyan
$IngestArgs = @(
    "uv", "run", "knox", "ingest",
    $InputFiles,
    "--sources-dir", $SourcesDir,
    "--output-dir",  $OutputDir,
    "--url-file",    $UrlFile,
    "--workers",     "1",
    "--progress"
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

# ── Stage 2: Compile (T2+T3+T4: resolve + compile + output) ────────────
Write-Host ""
Write-Host "=== Stage 2: Compile (resolve + compile + output) ===" -ForegroundColor Cyan
$CompileArgs = @(
    "uv", "run", "knox", "compile",
    $InputFiles,
    "--sources-dir",      $SourcesDir,
    "--output-dir",       $OutputDir,
    "--start-at-tier",    "resolve",
    "--llm-default-model", $LLMModel,
    "--embed-model",      $EmbedModel,
    "--embed-dim",        $EmbedDim,
    "--resolve-relations",
    "--flavor",           "obsidian",
    "--compiler",         $Compiler,
    "--workers",          "1",
    "--verify",
    "--index", "--synthesize", "--progress"
)

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
    "uv", "run", "knox", "lint",
    "--output-dir", (Join-Path $BaseDir "wiki"),
    "--sources-dir", (Join-Path $BaseDir "sources"),
    "--llm",
    "--llm-default-model", $LLMModel,
    "--fix",
    "--threshold", "80"
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
if (-not $WhatIf -and (Test-Path $BaseDir)) {
    $MdFiles = Get-ChildItem -Path $BaseDir -Filter "*.md" -Recurse
    Write-Host "  Output: $BaseDir ($($MdFiles.Count) markdown file(s))"
    $Index = Join-Path $BaseDir "index.md"
    if (Test-Path $Index) {
        Write-Host "  Entry:  $Index" -ForegroundColor Cyan
    }
}
