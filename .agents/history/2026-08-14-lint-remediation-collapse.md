# Lint remediation collapsed to page generation (2026-08-14)

Three overlapping tiers became one: LLM-generate missing concept pages from
cached extractions. Image-embed fixing moved into `run --fix` as the
`image-embed` structural rule; recommendations live only in `kompend lint gaps`;
`RemediationAction.tier` was removed. Same change added the `rumdl` structural
rule (`rumdl check` JSON over the vault's markdown, sources never touched, files
passed explicitly to bypass rumdl's ignore filtering).
