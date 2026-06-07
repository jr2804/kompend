"""JSON output writer."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from knox.core.models import CompiledWiki
from knox.output.base import BaseOutputWriter


class JsonWriter(BaseOutputWriter):
    """Write compiled wiki as JSON for downstream consumers."""

    def write(self, wiki: CompiledWiki, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "nodes": [dataclasses.asdict(n) for n in wiki.nodes],
            "edges": wiki.edges,
            "toc": wiki.toc,
            "quality_score": wiki.quality_score,
            "iteration": wiki.iteration,
            "metadata": wiki.metadata,
        }

        output_path = output_dir / "wiki.json"
        output_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return output_path