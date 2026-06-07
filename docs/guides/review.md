# Review Queue

Human-in-the-loop approval for compiled wiki pages.

## Storage layout

Inside `<output>/.knox/review/`:

| File | Meaning |
|------|---------|
| `<slug>.md` | Pending page (awaiting review) |
| `<slug>.approved.md` | Approved page |
| `<slug>.rejected.md` | Rejected page |
| `<slug>.locked` | Empty marker — prevents future re-compile |
| `_state.json` | State index (one entry per slug) |

## Workflow

1. Compile with `--review` to send pages to the queue instead of writing
   them directly to the output directory:

   ```bash
   knox compile *.pdf --review
   ```

2. Inspect the queue:

   ```bash
   knox review list            # human-readable
   knox review list --json     # machine-readable
   ```

3. Approve or reject:

   ```bash
   knox review approve my-page
   knox review approve --all
   knox review reject my-page --reason "incomplete"
   ```

4. (Optional) Lock approved pages so a future recompile cannot overwrite
   them:

   ```bash
   knox review lock my-page
   knox review unlock my-page
   ```

## Bypassing the queue

For CI runs where you trust the output:

```bash
knox compile *.pdf --review --bypass-review
```

`--bypass-review` overrides `--review` and writes pages directly to the
output directory.

## CI integration

```bash
# Get a JSON report
knox review list --json > queue.json
test "$(jq 'length' queue.json)" -eq 0  # fail if anything is pending
```

## FAQ

**Q: Where are approved pages written?**
A: They stay in `<output>/.knox/review/<slug>.approved.md`. The
`MarkdownWriter` reads from there when rendering the final wiki.

**Q: How do I batch-approve a CI run?**
A: `knox review approve --all` then `knox compile --bypass-review`.

**Q: Can I edit a page before approving?**
A: Yes — open `<output>/.knox/review/<slug>.md` in your editor, then
`knox review approve <slug>`.

**Q: What happens when a locked page is re-compiled?**
A: The compile writes the new page, but the locked page takes
precedence during render. Unlock with `knox review unlock <slug>` to
accept the new version.
