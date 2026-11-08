"""kompend compilation tier — knowledge graph construction and iterative refinement."""

from __future__ import annotations

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "graph": ["GraphCompiler"],
        "wicer": ["WiCERCompiler"],
    },
)
