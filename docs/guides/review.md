# Review Queue

Human-in-the-loop approval for compiled wiki pages.

## Storage layout

Inside `.kompend/review/`:

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
   kompend compile *.pdf --review
   ```

2. Inspect the queue:

   ```bash
   kompend review list            # human-readable
   kompend review list --json     # machine-readable
   ```

3. Approve or reject:

   ```bash
   kompend review approve my-page
   kompend review approve --all
   kompend review reject my-page --reason "incomplete"
   ```

4. (Optional) Lock approved pages so a future recompile cannot overwrite
   them:

   ```bash
   kompend review lock my-page
   kompend review unlock my-page
   ```

## Bypassing the queue

For CI runs where you trust the output:

```bash
kompend compile *.pdf --review --bypass-review
```

`--bypass-review` overrides `--review` and writes pages directly to the
output directory.

## CI integration

```bash
# Get a JSON report
kompend review list --json > queue.json
test "$(jq 'length' queue.json)" -eq 0  # fail if anything is pending
```

## FAQ

**Q: Where are approved pages written?**
A: They stay in `.kompend/review/<slug>.approved.md`. The
`MarkdownWriter` reads from there when rendering the final wiki.

**Q: How do I batch-approve a CI run?**
A: `kompend review approve --all` then `kompend compile --bypass-review`.

**Q: Can I edit a page before approving?**
A: Yes — open `.kompend/review/<slug>.md` in your editor, then
`kompend review approve <slug>`.

**Q: What happens when a locked page is re-compiled?**
A: The compile writes the new page, but the locked page takes
precedence during render. Unlock with `kompend review unlock <slug>` to
accept the new version.
