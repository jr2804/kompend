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
#   .\build-wiki.ps1              # full build
#   .\build-wiki.ps1 -Force       # clear caches and recompile from scratch
#   .\build-wiki.ps1 -WhatIf      # show command without executing

[CmdletBinding()]
param(
    [switch]$Force,
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"

# ── Configuration ──────────────────────────────────────────────────────
$ProjectRoot = $PSScriptRoot
$BaseDir     = Join-Path $ProjectRoot "demo-wiki"

$LLMModel   = "ollama/deepseek-v4-flash:cloud"
$EmbedModel = "ollama/granite-embedding:278m"
$EmbedDim   = 768

# ── Clean (if -Force) ──────────────────────────────────────────────────
if ($Force) {
    if (Test-Path $BaseDir) {
        Write-Host "Removing $BaseDir" -ForegroundColor Yellow
        if (-not $WhatIf) { Remove-Item -Recurse -Force $BaseDir }
    }
}

# ── Build ──────────────────────────────────────────────────────────────
$Args = @(
    "uv", "run", "knox", "compile",
    (Join-Path $ProjectRoot "tests/data/sources"),
    "--sources-dir",    (Join-Path $BaseDir "sources"),
    "--output-dir",     (Join-Path $BaseDir "wiki"),
    "--url-file",       (Join-Path $ProjectRoot "tests/data/urls.txt"),
    "--llm-default-model", $LLMModel,
    "--embed-model",    $EmbedModel,
    "--embed-dim",      $EmbedDim,
    "--index", "--index-store", "--synthesize", "--progress"
)

if ($Force) {
    $Args += "--force"
    $Args += "--no-cache"
}

Write-Host ""
Write-Host ($Args -join " ") -ForegroundColor DarkGray
Write-Host ""

if (-not $WhatIf) {
    & $Args[0] $Args[1..($Args.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        Write-Error "knox compile failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
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
