# build-wiki.ps1 --- Build the demo wiki for A/B model comparison
#
# Prerequisites:
#   - Ollama running with models pulled (see config files for model list)
#   - uv installed (https://docs.astral.sh/uv/)
#
# Usage:
#   .\scripts\build-wiki.ps1 scripts\config-cloud.toml     # enterprise LLM (~1M ctx)
#   .\scripts\build-wiki.ps1 scripts\config-local.toml      # local LLM (128K ctx)
#   .\scripts\build-wiki.ps1 scripts\config-local.toml -LLMModel "ollama/llama3.2"
#   .\scripts\build-wiki.ps1 scripts\config-cloud.toml -VLMModel "ollama-cloud/gemma4:cloud"
#   .\scripts\build-wiki.ps1 scripts\config-local.toml -EmbedModel "litellm:ollama/granite-embedding:278m"
#   .\scripts\build-wiki.ps1 scripts\config-cloud.toml -Preview
#
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$ConfigFile,

    [Parameter(Mandatory = $false)]
    [string]$LLMModel,

    [Parameter(Mandatory = $false)]
    [string]$VLMModel,

    [Parameter(Mandatory = $false)]
    [string]$EmbedModel,

    [switch]$Preview
)

$ErrorActionPreference = "Stop"

# ------ Sync environment (ensure latest source is installed) ------------------------------------------------
Write-Host "=== Syncing environment (uv sync) ===" -ForegroundColor Cyan
$ProjectRoot = $PSScriptRoot | Split-Path -Parent

if (-not $Preview) {
    & uv sync -U --directory $ProjectRoot
    if ($LASTEXITCODE -ne 0) {
        Write-Error "uv sync failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

# ------ Validate config ---------------------------------------------------------------------------------------------------------------------------------------------------------------
$ConfigPath = Resolve-Path $ConfigFile -ErrorAction Stop
Write-Host "Using config: $ConfigPath" -ForegroundColor Cyan

# ------ Read build settings from config ---------------------------------------------------------------------------------------------------------------
# We parse the TOML to extract model settings. Since PowerShell
# doesn't have native TOML support, we use a simple regex approach for
# the specific keys we need.

function Get-ConfigSectionValue($Section, $Key) {
    $lines = Get-Content $ConfigPath
    $currentSection = ""
    $escapedKey = [regex]::Escape($Key)
    $quotedPattern = '^\s*' + $escapedKey + '\s*=\s*"(.+)"\s*(#.*)?$'
    $unquotedPattern = '^\s*' + $escapedKey + '\s*=\s*([^#]+)'

    foreach ($line in $lines) {
        if ($line -match '^\s*\[(.+)\]\s*$') {
            $currentSection = $matches[1].Trim()
            continue
        }

        if ($currentSection -ne $Section) {
            continue
        }

        if ($line -match $quotedPattern) {
            return $matches[1].Trim()
        }

        if ($line -match $unquotedPattern) {
            return $matches[1].Trim()
        }
    }

    return $null
}

$ConfigLLMModel = Get-ConfigSectionValue "compile.llm" "default_model"
$ConfigVLMModel = Get-ConfigSectionValue "compile.llm" "vlm_model"
$ConfigEmbedModel = Get-ConfigSectionValue "compile.embedding" "model"

$EffectiveLLMModel = if ([string]::IsNullOrWhiteSpace($LLMModel)) { $ConfigLLMModel } else { $LLMModel }
$EffectiveVLMModel = if ([string]::IsNullOrWhiteSpace($VLMModel)) { $ConfigVLMModel } else { $VLMModel }
$EffectiveEmbedModel = if ([string]::IsNullOrWhiteSpace($EmbedModel)) { $ConfigEmbedModel } else { $EmbedModel }

# Derive output directory name from config filename:
#   scripts\config-cloud.toml  →  demo-wiki-cloud
#   scripts\config-local.toml   →  demo-wiki-local
$BaseName = [System.IO.Path]::GetFileNameWithoutExtension($ConfigFile)
$OutputName = $BaseName -replace "^config-", "demo-wiki-"
$OutputPath = Join-Path $ProjectRoot $OutputName

if (-not $EffectiveLLMModel) {
    Write-Error "Config missing [compile.llm] default_model"
    exit 1
}

Write-Host "  Model:  $EffectiveLLMModel" -ForegroundColor DarkGray
Write-Host "  VLM:    $EffectiveVLMModel" -ForegroundColor DarkGray
Write-Host "  Embed:  $EffectiveEmbedModel" -ForegroundColor DarkGray
Write-Host "  Output: $OutputPath" -ForegroundColor DarkGray

$InputFiles = Join-Path $ProjectRoot "tests/data/sources/*.pdf"
$UrlFile = Join-Path $ProjectRoot "tests/data/sources/urls.txt"

$Batch2Dir = Join-Path $ProjectRoot "tests/data/batch2"
$Batch2Pdf = Join-Path $ProjectRoot "tests/data/batch2/*.pdf"
$Batch2Urls = Join-Path $Batch2Dir "urls.txt"

# ------ Clean ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --force --keep-cache on knox init handles directory reset + cache
# preservation. The cache-skip logic in ingest already skips unchanged
# files (hash-based). No manual cache clearing needed.

# ------ Stage 0: Init (scaffold project structure) ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Stage 0: Initialize project ===" -ForegroundColor Cyan
$InitArgs = @(
    "uv", "run", "knox", "init",
    $OutputPath,
    "--bare",
    "--force",
    "--keep-cache",
    "--config", $ConfigPath
)

Write-Host ""
Write-Host ($InitArgs -join " ") -ForegroundColor DarkGray
Write-Host ""

if (-not $Preview) {
    & $InitArgs[0] $InitArgs[1..($InitArgs.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        Write-Error "knox init failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

# ------ Stage 0b: Persist model settings in wiki config ---------------------------------------------------------
Write-Host ""
Write-Host "=== Stage 0b: Configure models (knox config set) ===" -ForegroundColor Cyan

function Invoke-ConfigSet($Key, $Value) {
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return
    }

    $cmd = @("uv", "run", "knox", "--wiki-dir", $OutputPath, "config", "set", $Key, $Value)
    Write-Host ($cmd -join " ") -ForegroundColor DarkGray

    if (-not $Preview) {
        & $cmd[0] $cmd[1..($cmd.Length - 1)]
        if ($LASTEXITCODE -ne 0) {
            Write-Error "knox config set failed with exit code $LASTEXITCODE"
            exit $LASTEXITCODE
        }
    }
}

Invoke-ConfigSet "compile.llm.default_model" $EffectiveLLMModel
Invoke-ConfigSet "compile.llm.vlm_model" $EffectiveVLMModel
Invoke-ConfigSet "compile.embedding.model" $EffectiveEmbedModel

# ------ Stage 1: Ingest (T0+T1: download + parse) ------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Stage 1: Ingest (download + parse) ===" -ForegroundColor Cyan
$IngestArgs = @(
    "uv", "run", "knox",
    "--wiki-dir", $OutputPath,
    "ingest",
    $InputFiles,
    "--url-file", $UrlFile,
    "--workers", "1"
)

# $Force no longer forces re-parse. The cache-skip logic in ingest
# already skips unchanged files (hash-based). To force a full re-parse,
# run: knox ingest --no-cache --wiki-dir <dir> manually before this script.

Write-Host ""
Write-Host ($IngestArgs -join " ") -ForegroundColor DarkGray
Write-Host ""

if (-not $Preview) {
    & $IngestArgs[0] $IngestArgs[1..($IngestArgs.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        Write-Error "knox ingest failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

# ------ Stage 1b: Validate LLM provider ------------------------------------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Stage 1b: Validate LLM provider ===" -ForegroundColor Cyan
$DoctorArgs = @(
    "uv", "run", "knox",
    "--wiki-dir", $OutputPath,
    "doctor",
    "--verbose"
)

Write-Host ($DoctorArgs -join " ") -ForegroundColor DarkGray
Write-Host ""

if (-not $Preview) {
    & $DoctorArgs[0] $DoctorArgs[1..($DoctorArgs.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "knox doctor reported issues --- continuing build"
    }
}

# ------ Stage 2: Compile (T2+T3+T4+T5: chunk + resolve + compile + output) ---
Write-Host ""
Write-Host "=== Stage 2: Compile (index + chunk + resolve + compile + output) ===" -ForegroundColor Cyan
$CompileArgs = @(
    "uv", "run", "knox",
    "--wiki-dir", $OutputPath,
    "compile",
    "--workers", "1"
)

Write-Host ""
Write-Host ($CompileArgs -join " ") -ForegroundColor DarkGray
Write-Host ""

if (-not $Preview) {
    & $CompileArgs[0] $CompileArgs[1..($CompileArgs.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        Write-Error "knox compile failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

# ------ Stage 2b: Ingest batch 2 (incremental sources) ------------------------------------------------------
Write-Host ""
Write-Host "=== Stage 2b: Ingest batch 2 (incremental sources) ===" -ForegroundColor Cyan

$Ingest2Args = @(
    "uv", "run", "knox",
    "--wiki-dir", $OutputPath,
    "ingest",
    $Batch2Pdf,
    "--url-file", $Batch2Urls,
    "--workers", "1"
)

Write-Host ""
Write-Host ($Ingest2Args -join " ") -ForegroundColor DarkGray
Write-Host ""

if (-not $Preview) {
    & $Ingest2Args[0] $Ingest2Args[1..($Ingest2Args.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        Write-Error "knox ingest (batch 2) failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

# ------ Stage 2c: Re-compile with batch 2 content ------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Stage 2c: Re-compile (with batch 2 content) ===" -ForegroundColor Cyan
$Compile2Args = @(
    "uv", "run", "knox",
    "--wiki-dir", $OutputPath,
    "compile",
    "--workers", "1"
)

Write-Host ""
Write-Host ($Compile2Args -join " ") -ForegroundColor DarkGray
Write-Host ""

if (-not $Preview) {
    & $Compile2Args[0] $Compile2Args[1..($Compile2Args.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        Write-Error "knox compile (batch 2) failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

# ------ Lint (Structural + LLM rules, saves to lint_history.jsonl) ---------------------------
Write-Host ""
Write-Host "=== Running Linter (Structural + LLM rules) ===" -ForegroundColor Cyan
$LintArgs = @(
    "uv", "run", "knox",
    "--wiki-dir", $OutputPath,
    "lint", "run",
    "--llm",
    "--fix"
)

Write-Host ($LintArgs -join " ") -ForegroundColor DarkGray
Write-Host ""

if (-not $Preview) {
    & $LintArgs[0] $LintArgs[1..($LintArgs.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Linting reported issues (exit code $LASTEXITCODE)"
    }
}

# ------ Gap Analysis (source recommendations from lint results) ---------------------------------
Write-Host ""
Write-Host "=== Gap Analysis (source recommendations) ===" -ForegroundColor Cyan
$GapsArgs = @(
    "uv", "run", "knox",
    "--wiki-dir", $OutputPath,
    "lint", "gaps"
)

Write-Host ($GapsArgs -join " ") -ForegroundColor DarkGray
Write-Host ""

if (-not $Preview) {
    & $GapsArgs[0] $GapsArgs[1..($GapsArgs.Length - 1)]
    # gaps exits 0 regardless --- it's informational
}

# ------ Remediate (auto-fix + LLM-generated concept pages) ------------------------------------------------
Write-Host ""
Write-Host "=== Remediate (auto-fix + concept generation) ===" -ForegroundColor Cyan
$RemediateArgs = @(
    "uv", "run", "knox",
    "--wiki-dir", $OutputPath,
    "lint", "remediate"
)

Write-Host ($RemediateArgs -join " ") -ForegroundColor DarkGray
Write-Host ""

if (-not $Preview) {
    & $RemediateArgs[0] $RemediateArgs[1..($RemediateArgs.Length - 1)]
}

# ------ Lint History (trend across builds) ------------------------------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Lint History (trend) ===" -ForegroundColor Cyan
$HistoryArgs = @(
    "uv", "run", "knox",
    "--wiki-dir", $OutputPath,
    "lint", "history",
    "--last", "5"
)

Write-Host ($HistoryArgs -join " ") -ForegroundColor DarkGray
Write-Host ""

if (-not $Preview) {
    & $HistoryArgs[0] $HistoryArgs[1..($HistoryArgs.Length - 1)]
}

# ------ Load demo content from TOML (learn/capture/write/query phrases) ------------
$DemoContentFile = Join-Path $PSScriptRoot "demo-content.toml"
$DemoContent = $null
if (Test-Path $DemoContentFile) {
    $json = & uv run python -c "import json, sys, tomllib; print(json.dumps(tomllib.load(open(sys.argv[1], 'rb'))))" $DemoContentFile
    $DemoContent = $json | ConvertFrom-Json
} else {
    Write-Host "  No demo-content.toml found --- skipping capture/query stages" -ForegroundColor DarkGray
}

# ------ Helper: format a command line for display, quoting args with spaces ------
function Format-CmdDisplay {
    param([string[]]$CmdArgs)
    return ($CmdArgs | ForEach-Object {
        if ($_ -match '\s') { "'$_'" } else { $_ }
    }) -join ' '
}

# ------ Stage 3: Capture workflow demo (learn + capture + write) ---------------------------
if ($DemoContent) {
    Write-Host ""
    Write-Host "=== Stage 3: Capture workflow demo ===" -ForegroundColor Cyan

    # learn
    if ($DemoContent.learn) {
        foreach ($item in $DemoContent.learn) {
            $cmd = @("uv", "run", "knox", "--wiki-dir", $OutputPath, "learn", $item.text)
            if ($item.tags) {
                foreach ($tag in $item.tags) {
                    $cmd += @("--tag", $tag)
                }
            }
            Write-Host ""
            Write-Host (Format-CmdDisplay $cmd) -ForegroundColor DarkGray
            Write-Host ""
            if (-not $Preview) {
                & $cmd[0] $cmd[1..($cmd.Length - 1)]
                if ($LASTEXITCODE -ne 0) {
                    Write-Warning "knox learn failed (exit $LASTEXITCODE) --- continuing"
                }
            }
        }
    }

    # capture
    if ($DemoContent.capture) {
        foreach ($item in $DemoContent.capture) {
            $cmd = @("uv", "run", "knox", "--wiki-dir", $OutputPath, "capture", $item.text)
            Write-Host ""
            Write-Host (Format-CmdDisplay $cmd) -ForegroundColor DarkGray
            Write-Host ""
            if (-not $Preview) {
                & $cmd[0] $cmd[1..($cmd.Length - 1)]
                if ($LASTEXITCODE -ne 0) {
                    Write-Warning "knox capture failed (exit $LASTEXITCODE) --- continuing"
                }
            }
        }
    }

    # write (resolves source filenames to paths in vault)
    if ($DemoContent.write) {
        foreach ($item in $DemoContent.write) {
            $writeArgs = @("uv", "run", "knox", "--wiki-dir", $OutputPath, "write", $item.title)
            $resolvedCount = 0
            if ($item.sources) {
                foreach ($src in $item.sources) {
                    $resolved = Get-ChildItem -Path (Join-Path $OutputPath "sources") -Filter $src -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
                    if ($resolved) {
                        $writeArgs += @("--source", $resolved)
                        $resolvedCount++
                    } else {
                        Write-Host "  Warning: source '$src' not found in vault" -ForegroundColor Yellow
                    }
                }
            }
            if ($resolvedCount -gt 0) {
                Write-Host ""
                Write-Host (Format-CmdDisplay $writeArgs) -ForegroundColor DarkGray
                Write-Host ""
                if (-not $Preview) {
                    & $writeArgs[0] $writeArgs[1..($writeArgs.Length - 1)]
                    if ($LASTEXITCODE -ne 0) {
                        Write-Warning "knox write failed (exit $LASTEXITCODE) --- continuing"
                    }
                }
            } else {
                Write-Host "  Skipping 'knox write' --- no source files resolved" -ForegroundColor DarkGray
            }
        }
    }

    # ------ Stage 4: Query demo (RAG queries) ------------------------------------------------------------------------------------------
    if ($DemoContent.query) {
        Write-Host ""
        Write-Host "=== Stage 4: Query demo ===" -ForegroundColor Cyan

        foreach ($item in $DemoContent.query) {
            $cmd = @("uv", "run", "knox", "--wiki-dir", $OutputPath, "query", $item.question)
            Write-Host ""
            Write-Host (Format-CmdDisplay $cmd) -ForegroundColor DarkGray
            Write-Host ""
            if (-not $Preview) {
                & $cmd[0] $cmd[1..($cmd.Length - 1)]
                if ($LASTEXITCODE -ne 0) {
                    Write-Warning "knox query failed (exit $LASTEXITCODE) --- continuing"
                }
            }
        }
    }
}

# ------ Stage 5: Status + Trace (diagnostics) ------------------------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Stage 5: Status + Trace ===" -ForegroundColor Cyan

$StatusArgs = @(
    "uv", "run", "knox",
    "--wiki-dir", $OutputPath,
    "status"
)
Write-Host ""
Write-Host ($StatusArgs -join " ") -ForegroundColor DarkGray
Write-Host ""
if (-not $Preview) {
    & $StatusArgs[0] $StatusArgs[1..($StatusArgs.Length - 1)]
}

$TraceArgs = @(
    "uv", "run", "knox",
    "--wiki-dir", $OutputPath,
    "trace"
)
Write-Host ""
Write-Host ($TraceArgs -join " ") -ForegroundColor DarkGray
Write-Host ""
if (-not $Preview) {
    & $TraceArgs[0] $TraceArgs[1..($TraceArgs.Length - 1)]
}

# ------ Stage 6: Organize (dry-run source categorisation) ------------------------------------------------------
Write-Host ""
Write-Host "=== Stage 6: Organize (dry-run) ===" -ForegroundColor Cyan
$OrganizeArgs = @(
    "uv", "run", "knox",
    "--wiki-dir", $OutputPath,
    "organize",
    "--dry-run"
)
Write-Host ""
Write-Host ($OrganizeArgs -join " ") -ForegroundColor DarkGray
Write-Host ""
if (-not $Preview) {
    & $OrganizeArgs[0] $OrganizeArgs[1..($OrganizeArgs.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "knox organize failed (exit $LASTEXITCODE) --- continuing"
    }
}

# ------ Stage 7: Final Lint Check (score after all remediation + write) ----------
# Re-runs the linter read-only (no --fix) so the post-remediation score is
# visible, then re-prints history to show the improvement trend including
# this final run. New pages from `knox write` / concept generation are
# re-evaluated here too.
Write-Host ""
Write-Host "=== Stage 7: Final Lint Check (post-remediation score) ===" -ForegroundColor Cyan
$FinalLintArgs = @(
    "uv", "run", "knox",
    "--wiki-dir", $OutputPath,
    "lint", "run",
    "--llm"
)

Write-Host ""
Write-Host ($FinalLintArgs -join " ") -ForegroundColor DarkGray
Write-Host ""

if (-not $Preview) {
    & $FinalLintArgs[0] $FinalLintArgs[1..($FinalLintArgs.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Final lint reported issues (exit code $LASTEXITCODE)"
    }
}

Write-Host ""
Write-Host "=== Final Lint History (improvement trend) ===" -ForegroundColor Cyan
Write-Host ($HistoryArgs -join " ") -ForegroundColor DarkGray
Write-Host ""
if (-not $Preview) {
    & $HistoryArgs[0] $HistoryArgs[1..($HistoryArgs.Length - 1)]
}

# ------ Done ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Build complete ===" -ForegroundColor Green
if (-not $Preview -and (Test-Path $OutputPath)) {
    $MdFiles = Get-ChildItem -Path $OutputPath -Filter "*.md" -Recurse
    Write-Host "  Output: $OutputPath ($($MdFiles.Count) markdown file(s))"
    $Index = Join-Path $OutputPath "index.md"
    if (Test-Path $Index) {
        Write-Host "  Entry:  $Index" -ForegroundColor Cyan
    }
}
