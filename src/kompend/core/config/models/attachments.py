"""Attachment lifecycle models (``[attachments]``)."""

from __future__ import annotations

from pydantic import Field

from kompend.core.config.models._base import _KompendBaseModel
from kompend.core.keys import EnvVar


class AttachmentsConfig(_KompendBaseModel):
    """Attachment lifecycle configuration (auto-adoption, vector indexing)."""

    ingest_adopted: bool = Field(
        default=False,
        validation_alias=EnvVar.ATTACHMENTS_INGEST_ADOPTED,
        description=(
            "Global gate for attachment ingestion: auto-adopted attachments "
            "(files under attachments/<slug>/ without a manifest entry) are "
            "registered but NOT ingestible by default. When True, adopted "
            "text/image files become ingestible. Effective ingestibility at "
            "compile is manifest flag AND this gate."
        ),
    )
    embed: bool = Field(
        default=False,
        validation_alias=EnvVar.ATTACHMENTS_EMBED,
        description="Whether ingestible attachments (text/image captions) are included in the vector index. Replaces the deprecated search.embed_attachments.",
    )
