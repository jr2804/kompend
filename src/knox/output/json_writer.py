"""JSON output writer."""

from __future__ import annotations

import json
from pathlib import Path

from knox.core.models import CompiledWiki
from knox.output.base import BaseOutputWriter


class JsonWriter(BaseOutputWriter):
    """Write compiled wiki as JSON for downstream consumers."""

    def write(self, wiki: CompiledWiki, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "compiled_wiki.json"
        data = {
            "nodes": [
                {
                    "id": n.id,
                    "title": n.title,
                    "content": n.content,
                    "node_type": n.node_type,
                    "source_chunks": n.source_chunks,
                    "metadata": n.metadata,
                }
                for n in wiki.nodes
            ],
            "edges": wiki.edges,
            "toc": wiki.toc,
            "metadata": wiki.metadata,
        }
        output_path.write_text(json.dumps(data, indent=2))
