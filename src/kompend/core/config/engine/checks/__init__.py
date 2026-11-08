"""Config-check helpers.

``_check.py`` is the entry point; the modules here hold one concern each:

- ``messages.py`` — status vocabulary and report rows
- ``schema.py`` — the live :class:`KompendConfig` schema walk
- ``deprecated.py`` — rows for removed and renamed keys
- ``cross_key.py`` — the pydantic validation pass over the merged config
"""
