# build-wiki.ps1 — Download sources and compile the demo wiki
#
# Prerequisites:
#   - Ollama running with models pulled:
#       ollama pull deepseek-v4-flash:cloud
#       ollama pull glm-ocr
#       ollama pull granite-embedding:278m
#   - uv installed (https://docs.astral.sh/uv/)
#
# Usage:
#   .\build-wiki.ps1              # full build (download + compile)
#   .\build-wiki.ps1 -SkipDownload # skip URL downloads, compile only
#   .\build-wiki.ps1 -Force        # clear caches and recompile from scratch
#   .\build-wiki.ps1 -WhatIf       # show commands without executing

[CmdletBinding()]
param(
    [switch]$SkipDownload,
    [switch]$Force,
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"

# ── Configuration ──────────────────────────────────────────────────────
$ProjectRoot   = $PSScriptRoot
$SourcesDir    = Join-Path $ProjectRoot "sources"
$OutputDir     = Join-Path $ProjectRoot "wiki"
$UrlsFile      = Join-Path $ProjectRoot "tests/data/urls.txt"
$LocalPdfs     = Join-Path $ProjectRoot "tests/data/sources"

$LLMModel      = "ollama/deepseek-v4-flash:cloud"
$EmbedModel    = "ollama/granite-embedding:278m"
$EmbedDim      = 768
$Timeout       = 7200  # seconds (2 hours — large corpora are slow on CPU)

# ── Helper ─────────────────────────────────────────────────────────────
function Invoke-Logged {
    param([string]$Label, [string]$Command)
    Write-Host ""
    Write-Host "=== $Label ===" -ForegroundColor Cyan
    Write-Host "  $Command" -ForegroundColor DarkGray
    Write-Host ""
    if (-not $WhatIf) {
        Invoke-Expression $Command
        if ($LASTEXITCODE -ne 0) {
            Write-Error "'$Label' failed with exit code $LASTEXITCODE"
            exit $LASTEXITCODE
        }
    }
}

# ── Step 0: Clean (if -Force) ──────────────────────────────────────────
if ($Force) {
    Write-Host "Cleaning previous output and caches..." -ForegroundColor Yellow
    $DirsToClean = @($SourcesDir, $OutputDir)
    foreach ($Dir in $DirsToClean) {
        if (Test-Path $Dir) {
            Write-Host "  Removing $Dir"
            if (-not $WhatIf) { Remove-Item -Recurse -Force $Dir }
        }
    }
}

# ── Step 1: Download URLs ──────────────────────────────────────────────
if (-not $SkipDownload) {
    if (-not (Test-Path $SourcesDir)) {
        if (-not $WhatIf) { New-Item -ItemType Directory -Path $SourcesDir | Out-Null }
        Write-Host "Created $SourcesDir"
    }

    # Read non-comment, non-empty lines from urls.txt
    $Urls = Get-Content $UrlsFile |
        Where-Object { $_ -match '^\s*[^#]' } |
        ForEach-Object { $_.Trim() }

    Write-Host ""
    Write-Host "=== Downloading $($Urls.Count) URL(s) ===" -ForegroundColor Cyan
    $i = 0
    foreach ($Url in $Urls) {
        $i++
        Write-Host "  [$i/$($Urls.Count)] $Url"
        if (-not $WhatIf) {
            # Download via knox compile --url (handles User-Agent, redirect, etc.)
            uv run knox compile $Url `
                --sources-dir $SourcesDir `
                --output-dir $OutputDir `
                --mode index-only `
                2>&1 | ForEach-Object { Write-Host "    $_" }
        }
    }
    Write-Host "  Done." -ForegroundColor Green
}

# ── Step 2: Copy local PDFs ────────────────────────────────────────────
if (Test-Path $LocalPdfs) {
    $Pdfs = Get-ChildItem -Path $LocalPdfs -Filter "*.pdf"
    if ($Pdfs.Count -gt 0) {
        if (-not (Test-Path $SourcesDir)) {
            if (-not $WhatIf) { New-Item -ItemType Directory -Path $SourcesDir | Out-Null }
        }
        Write-Host ""
        Write-Host "=== Copying $($Pdfs.Count) local PDF(s) ===" -ForegroundColor Cyan
        foreach ($Pdf in $Pdfs) {
            $Dest = Join-Path $SourcesDir $Pdf.Name
            if (-not (Test-Path $Dest)) {
                Write-Host "  $($Pdf.Name)"
                if (-not $WhatIf) { Copy-Item $Pdf.FullName $Dest }
            } else {
                Write-Host "  $($Pdf.Name) (already exists, skipped)" -ForegroundColor DarkGray
            }
        }
    }
}

# ── Step 3: Compile ────────────────────────────────────────────────────
$CompileArgs = @(
    "uv", "run", "knox", "compile",
    $SourcesDir,
    "--output-dir", $OutputDir,
    "--llm-default-model", $LLMModel,
    "--embed-model", $EmbedModel,
    "--embed-dim", $EmbedDim,
    "--index",
    "--synthesize",
    "--progress"
)

if ($Force) {
    $CompileArgs += "--force"
    $CompileArgs += "--no-cache"
}

$CompileCmd = $CompileArgs -join " "
Invoke-Logged "Compiling wiki (timeout=$Timeout`s)" $CompileCmd

# ── Done ───────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Build complete ===" -ForegroundColor Green
if (-not $WhatIf -and (Test-Path $OutputDir)) {
    $Files = Get-ChildItem -Path $OutputDir -Filter "*.md" -Recurse
    Write-Host "  Output: $OutputDir ($($Files.Count) markdown file(s))"
}
