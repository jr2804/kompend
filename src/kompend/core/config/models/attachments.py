"""Attachment lifecycle models (``[attachments]``)."""

from __future__ import annotations

from pydantic import Field, model_validator

from kompend.core.config.enums import AttachmentMode
from kompend.core.config.models._base import _KompendBaseModel
from kompend.core.keys import EnvVar


class AttachmentsConfig(_KompendBaseModel):
    """Attachment lifecycle configuration (mode, auto-adoption, vector indexing)."""

    mode: AttachmentMode = Field(
        default=AttachmentMode.INDEX_ONLY,
        validation_alias=EnvVar.ATTACHMENTS_MODE,
        description=(
            "How far attachments travel through the pipeline: 'off' stores "
            "and mentions them only, 'index-only' parses/indexes supported "
            "attachments without ever compiling them (default), "
            "'full-ingest' runs them through the whole pipeline. A vault "
            "that set ingest_adopted = true before mode existed upgrades to "
            "'full-ingest' automatically; set mode explicitly to opt out."
        ),
    )
    ingest_adopted: bool = Field(
        default=True,
        validation_alias=EnvVar.ATTACHMENTS_INGEST_ADOPTED,
        description=(
            "Whether auto-adopted attachments (files under "
            "attachments/<slug>/ without a manifest entry) become "
            "ingestible. When False, adoption only registers the file — "
            "mark it with 'kompend attachments ingest' to ingest it "
            "anyway. Explicitly marked attachments are processed "
            "regardless of this gate."
        ),
    )
    embed: bool = Field(
        default=True,
        validation_alias=EnvVar.ATTACHMENTS_EMBED,
        description="Whether ingestible attachments (text/image captions) are included in the vector index. Replaces the deprecated search.embed_attachments.",
    )

    @model_validator(mode="before")
    @classmethod
    def _upgrade_ingest_gate(cls, data: object) -> object:
        """Vaults predating ``mode``: ``ingest_adopted = true`` meant full ingestion.

        The old gate turned every ingestible attachment into a full pipeline
        citizen, so a config that says so without a ``mode`` upgrades to
        ``full-ingest``; any config that names ``mode`` (field or env var)
        keeps exactly what it asked for.
        """
        if not isinstance(data, dict):
            return data
        if "mode" in data or EnvVar.ATTACHMENTS_MODE in data:
            return data
        adopted = data.get("ingest_adopted", data.get(EnvVar.ATTACHMENTS_INGEST_ADOPTED, False))
        if adopted:
            data["mode"] = AttachmentMode.FULL_INGEST
        return data
