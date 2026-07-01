"""Event hooks system — placeholder for future use.

The KNOX pipeline, lint, capture, and query commands all previously
fired lifecycle events via a global callback registry, but no hook
was ever registered by any user or pack. The infrastructure was
deleted in favor of direct logging at the call sites.

If a user-space hook mechanism is needed later, add it then.
"""
