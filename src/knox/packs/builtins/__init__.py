"""Built-in contribution packs bundled with KNOX."""

from knox.packs.builtins.academic import academic_pack
from knox.packs.builtins.technical import technical_pack

__all__ = [
    "academic_pack",
    "technical_pack",
]
