# User Notes

This file is a place for your own voice — context, intent, and notes
that the KOMPEND compiler should keep in mind when producing the wiki.

KOMPEND reads this file and exposes its content to compilation templates
as `{{ user_notes }}`. KOMPEND **never writes to or modifies** this file.
Whatever you write here is preserved verbatim.

## How to use it

- Describe the audience and purpose of the wiki.
- Capture recurring terminology, naming conventions, and abbreviations.
- Note topics that should receive extra attention or that are out of scope.
- Add callouts for things that should *not* be inferred from the sources.

## Example

> This wiki documents an internal API. Use second-person ("you") when
> describing endpoints. Always include a code example for any new
> endpoint. Treat RFC 7807 (Problem Details for HTTP APIs) as the
> canonical error format.
