"""Technical documentation pack — prompts for API docs, specs, and tutorials."""

from knox.packs.base import Pack

technical_pack = Pack(
    name="technical",
    description="Prompts for technical documentation, API references, specifications, and tutorials.",
    version="1.0.0",
    dependencies=[],
    roles={
        "compile": (
            "You are compiling a knowledge wiki from technical documentation sources. "
            "Extract: API endpoints, functions, parameters, return types, error codes, "
            "configuration options, environment setup steps, and architecture decisions. "
            "Return structured JSON with 'nodes' (components, APIs, concepts) and "
            "'edges' (dependencies, usage relationships, version changes)."
        ),
        "resolve": (
            "Extract technical entities from documentation. Categorise each as: "
            "'api_endpoint', 'function', 'class', 'parameter', 'config_option', "
            "'error_code', 'dependency', 'tool'. "
            "Track version-specific information when available."
        ),
        "query": (
            "Answer the user's technical question based on the compiled wiki. "
            "Include code examples and configuration snippets when relevant. "
            "Mention version compatibility and deprecation notices."
        ),
    },
)
