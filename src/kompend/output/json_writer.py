"""JSON output writer."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import ClassVar

from kompend.core.models import CompiledWiki, WikiField
from kompend.output.base import BaseOutputWriter


class JsonWriter(BaseOutputWriter):
    """Write compiled wiki as JSON for downstream consumers."""

    name: ClassVar[str] = "json"

    def write(self, wiki: CompiledWiki, output_dir: Path) -> Path:  # noqa: PLR6301 — abc override, needs self
        output_dir.mkdir(parents=True, exist_ok=True)

        data = {
            WikiField.NODES: [dataclasses.asdict(n) for n in wiki.nodes],
            WikiField.EDGES: wiki.edges,
            WikiField.TOC: wiki.toc,
            WikiField.QUALITY_SCORE: wiki.quality_score,
            WikiField.ITERATION: wiki.iteration,
            "metadata": wiki.metadata,
        }

        output_file = output_dir / "wiki.json"
        output_file.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return output_file
