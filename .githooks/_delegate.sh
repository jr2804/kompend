#!/usr/bin/env sh
# Delegate a git hook to the tool-managed hooks in .beads/hooks.
#
# Git honours exactly one core.hooksPath. This repository owns that path
# (.githooks, set by `mise run install-hooks`) so its own checks run, and
# forwards to the hooks written by other tools — beads (bd hooks run <name>,
# generated between BEGIN/END BEADS markers) and repowise (wiki sync). Those
# files must not be edited: the tools regenerate them.
#
# Usage: _delegate.sh <hook-name> [hook-args...] — called by the .githooks
# wrappers, which forward their own name and arguments.
set -e

hook=$1
shift

root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
target="$root/.beads/hooks/$hook"

[ -x "$target" ] || exit 0
exec "$target" "$@"
