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
#   .\scripts\build-wiki.ps1 scripts\config-local.toml -EmbedModel "ollama/granite-embedding:278m"
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

# Anchor cwd to the repo root so every relative path and `uv run` (which uses
# the venv in $ProjectRoot) resolves correctly, regardless of where the script
# is invoked from.
Set-Location $ProjectRoot

if (-not $Preview) {
    & uv sync -U
    if ($LASTEXITCODE -ne 0) {
        Write-Error "uv sync failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

# ------ Validate config ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Validate the config exists (relative to cwd = $ProjectRoot). Keep $ConfigFile
# in its original (relative) form for passing to `init --config` — an absolute
# resolved path would carry backslashes that get mangled by the native-command
# call layer on Windows.
$null = Resolve-Path $ConfigFile -ErrorAction Stop
$ConfigPath = $ConfigFile
Write-Host "Using config: $ConfigPath" -ForegroundColor Cyan

# ------ Default models (per-config env-var overrides) ------------------------------------------------
# Config files use config-specific env vars (${KNOX_LOCAL_LLM}, ${KNOX_CLOUD_LLM}, etc.)
# that are expanded at load time. The script sets the right set based on which
# config file is in use. The env vars ${KNOX_LLM_DEFAULT_MODEL} / ${KNOX_VLM_MODEL}
# / ${KNOX_EMBEDDING_MODEL} at the _ENV_MAP layer (Layer 2) still work as a generic
# override, but the config-specific vars are the recommended approach.

# Derive which config we are using
$BaseName = [System.IO.Path]::GetFileNameWithoutExtension($ConfigFile)
$IsLocalConfig = $BaseName -like "*local*"

# Prefer script parameter > existing env var > script default
if ($IsLocalConfig) {
    $env:KNOX_LOCAL_LLM        = if ([string]::IsNullOrWhiteSpace($LLMModel))   { $env:KNOX_LOCAL_LLM        ?? "ollama/gemma4:e2b" }   else { $LLMModel }
    $env:KNOX_LOCAL_VLM        = if ([string]::IsNullOrWhiteSpace($VLMModel))   { $env:KNOX_LOCAL_VLM        ?? "ollama/lmfm2.5" }     else { $VLMModel }
    $env:KNOX_LOCAL_EMBEDDING  = if ([string]::IsNullOrWhiteSpace($EmbedModel)) { $env:KNOX_LOCAL_EMBEDDING  ?? "ollama/granite-embedding:278m" } else { $EmbedModel }
} else {
    $env:KNOX_CLOUD_LLM        = if ([string]::IsNullOrWhiteSpace($LLMModel))   { $env:KNOX_CLOUD_LLM        ?? "ollama-cloud/deepseek-v4-flash" }   else { $LLMModel }
    $env:KNOX_CLOUD_VLM        = if ([string]::IsNullOrWhiteSpace($VLMModel))   { $env:KNOX_CLOUD_VLM        ?? "ollama-cloud/gemma4:31b" }         else { $VLMModel }
    $env:KNOX_CLOUD_EMBEDDING  = if ([string]::IsNullOrWhiteSpace($EmbedModel)) { $env:KNOX_CLOUD_EMBEDDING  ?? "openai/text-embedding-3-large" }   else { $EmbedModel }
}

# Derive output directory name from config filename:
#   scripts\config-cloud.toml  →  demo-wiki-cloud
#   scripts\config-local.toml   →  demo-wiki-local
$OutputName = $BaseName -replace "^config-", "demo-wiki-"
# Relative to cwd ($ProjectRoot) — avoids backslash-mangling when passed to
# native commands via `--wiki-dir`. knox resolves it against cwd.
$OutputPath = $OutputName

Write-Host "  Output: $OutputPath" -ForegroundColor DarkGray

# Input sources are relative to cwd as well.
$InputFiles = "tests/data/sources/*.pdf"
$UrlFile = "tests/data/sources/urls.txt"

$Batch2Dir = "tests/data/batch2"
$Batch2Pdf = "tests/data/batch2/*.pdf"
$Batch2Urls = "$Batch2Dir/urls.txt"

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

# ------ Stage 1: Ingest (T0+T1: download + parse) ------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Stage 1: Ingest (download + parse) ===" -ForegroundColor Cyan
$IngestArgs = @(
    "uv", "run", "knox",
    "--wiki-dir", $OutputPath,
    "ingest",
    $InputFiles,
    "--url-file", $UrlFile
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
        Write-Error "knox doctor failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

# ------ Stage 2: Compile (T2+T3+T4+T5: chunk + resolve + compile + output) ---
Write-Host ""
Write-Host "=== Stage 2: Compile (index + chunk + resolve + compile + output) ===" -ForegroundColor Cyan
$CompileArgs = @(
    "uv", "run", "knox",
    "--wiki-dir", $OutputPath,
    "compile"
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

# ------ Stage 3: Ingest batch 2 (incremental sources) ------------------------------------------------------
Write-Host ""
Write-Host "=== Stage 3: Ingest batch 2 (incremental sources) ===" -ForegroundColor Cyan

$Ingest2Args = @(
    "uv", "run", "knox",
    "--wiki-dir", $OutputPath,
    "ingest",
    $Batch2Pdf,
    "--url-file", $Batch2Urls
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

# ------ Stage 4: Re-compile with batch 2 content ------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Stage 4: Re-compile (with batch 2 content) ===" -ForegroundColor Cyan
$Compile2Args = @(
    "uv", "run", "knox",
    "--wiki-dir", $OutputPath,
    "compile"
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

# ------ Stage 5: Lint (Structural + LLM rules, saves to lint_history.jsonl) ---------------------------
Write-Host ""
Write-Host "=== Stage 5: Running Linter (Structural + LLM rules) ===" -ForegroundColor Cyan
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

# ------ Stage 6: Gap Analysis (source recommendations from lint results) ---------------------------------
Write-Host ""
Write-Host "=== Stage 6: Gap Analysis (source recommendations) ===" -ForegroundColor Cyan
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

# ------ Stage 7: Remediate (auto-fix + LLM-generated concept pages) ------------------------------------------------
Write-Host ""
Write-Host "=== Stage 7: Remediate (auto-fix + concept generation) ===" -ForegroundColor Cyan
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

# ------ Stage 8: Lint History (trend across builds) ------------------------------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Stage 8: Lint History (trend) ===" -ForegroundColor Cyan
$HistoryArgs = @(
    "uv", "run", "knox",
    "--wiki-dir", $OutputPath,
    "lint", "history",
    "--last", "5",
    "--trend"
)

Write-Host ($HistoryArgs -join " ") -ForegroundColor DarkGray
Write-Host ""

if (-not $Preview) {
    & $HistoryArgs[0] $HistoryArgs[1..($HistoryArgs.Length - 1)]
}

# ------ Load demo content from TOML (learn/capture/write/query phrases) ------------
$DemoContentFile = Join-Path $PSScriptRoot "..\configs\demo-content.toml"
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

# ------ Stage 9: Capture workflow demo (learn + capture + write) ---------------------------
if ($DemoContent) {
    Write-Host ""
    Write-Host "=== Stage 9: Capture workflow demo ===" -ForegroundColor Cyan

    # learn
    if ($DemoContent.learn) {
        foreach ($item in $DemoContent.learn) {
            $cmd = @("uv", "run", "knox", "--wiki-dir", $OutputPath, "learn", $item.text)
            if ($item.tags) {
                foreach ($tag in $item.tags) {
                    $cmd += @("--tags", $tag)
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
                    # Resolve within the vault's sources/ dir; emit a path relative
                    # to cwd ($ProjectRoot) with forward slashes to avoid the
                    # native-command backslash-mangling on Windows.
                    $resolved = Get-ChildItem -Path "$OutputPath/sources" -Filter $src -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
                    if ($resolved) {
                        $rel = $resolved.Replace((Get-Location).Path + [System.IO.Path]::DirectorySeparatorChar, "").Replace("\", "/")
                        $writeArgs += @("--source", $rel)
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

    # ------ Stage 10: Query demo (RAG queries) ------------------------------------------------------------------------------------------
    if ($DemoContent.query) {
        Write-Host ""
        Write-Host "=== Stage 10: Query demo ===" -ForegroundColor Cyan

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

# ------ Stage 11: Status + Trace (diagnostics) ------------------------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Stage 11: Status + Trace ===" -ForegroundColor Cyan

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

$ProvenanceArgs = @(
    "uv", "run", "knox",
    "--wiki-dir", $OutputPath,
    "provenance"
)
Write-Host ""
Write-Host ($ProvenanceArgs -join " ") -ForegroundColor DarkGray
Write-Host ""
if (-not $Preview) {
    & $ProvenanceArgs[0] $ProvenanceArgs[1..($ProvenanceArgs.Length - 1)]
}

# ------ Stage 12: Organize (dry-run source categorisation) ------------------------------------------------------
Write-Host ""
Write-Host "=== Stage 12: Organize (dry-run) ===" -ForegroundColor Cyan
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

# ------ Stage 13: Final Lint Check (score after all remediation + write) ----------
# Re-runs the linter read-only (no --fix) so the post-remediation score is
# visible, then re-prints history to show the improvement trend including
# this final run. New pages from `knox write` / concept generation are
# re-evaluated here too.
Write-Host ""
Write-Host "=== Stage 13: Final Lint Check (post-remediation score) ===" -ForegroundColor Cyan
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

# ------ Stage 14: Pipeline Log (on-demand report: compile + runs + lint trend) ----
# log.md is no longer auto-written; regenerate it on demand from the
# structured stores (runs.jsonl, lint_history.jsonl, last_wiki.json) and
# print the full report so the build ends with a complete picture.
Write-Host ""
Write-Host "=== Stage 14: Pipeline Log (regenerated report) ===" -ForegroundColor Cyan
$LogArgs = @(
    "uv", "run", "knox",
    "--wiki-dir", $OutputPath,
    "log",
    "--last", "15"
)

Write-Host ""
Write-Host ($LogArgs -join " ") -ForegroundColor DarkGray
Write-Host ""

if (-not $Preview) {
    & $LogArgs[0] $LogArgs[1..($LogArgs.Length - 1)]
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
