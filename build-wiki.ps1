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

$LLMModel   = "ollama/deepseek-v4-flash:cloud"
$EmbedModel = "ollama/granite-embedding:278m"
$EmbedDim   = 768

# ── Clean (if -Force) ──────────────────────────────────────────────────
if ($Force) {
    foreach ($Dir in @("sources", "wiki")) {
        $Path = Join-Path $ProjectRoot $Dir
        if (Test-Path $Path) {
            Write-Host "Removing $Path" -ForegroundColor Yellow
            if (-not $WhatIf) { Remove-Item -Recurse -Force $Path }
        }
    }
}

# ── Build ──────────────────────────────────────────────────────────────
$Args = @(
    "uv", "run", "knox", "compile",
    (Join-Path $ProjectRoot "tests/data/sources"),
    "--sources-dir",    (Join-Path $ProjectRoot "sources"),
    "--output-dir",     (Join-Path $ProjectRoot "wiki"),
    "--url-file",       (Join-Path $ProjectRoot "tests/data/urls.txt"),
    "--llm-default-model", $LLMModel,
    "--embed-model",    $EmbedModel,
    "--embed-dim",      $EmbedDim,
    "--index", "--synthesize", "--progress"
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
$OutputDir = Join-Path $ProjectRoot "wiki"
Write-Host ""
Write-Host "=== Build complete ===" -ForegroundColor Green
if (-not $WhatIf -and (Test-Path $OutputDir)) {
    $Files = Get-ChildItem -Path $OutputDir -Filter "*.md" -Recurse
    Write-Host "  Output: $OutputDir ($($Files.Count) markdown file(s))"
}
