"""Regression test for offloading sidecar enrichment to a worker thread.

ITEM 9: ``enrich_sidecars_with_surrounding`` reads the whole ``blocks.jsonl``
from disk synchronously (tens-to-hundreds of ms). The analyze path now runs it
through ``asyncio.to_thread`` so it does not block the event loop. This test
spies on ``asyncio.to_thread`` while driving ``analyze_multimodal`` and asserts
the enrichment function is dispatched off the loop.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from lightrag import ROLES, LightRAG, RoleLLMConfig
from lightrag.multimodal_context import enrich_sidecars_with_surrounding
from tests.pipeline.conftest import build_role_rag

pytestmark = pytest.mark.offline


async def _vlm_func(prompt, **kwargs) -> str:
    return json.dumps(
        {"name": "fig-1", "type": "Chart", "description": "concise description"}
    )


def _build_rag(tmp_path: Path) -> LightRAG:
    role_configs = {}
    for spec in ROLES:
        if spec.name == "vlm":
            role_configs[spec.name] = RoleLLMConfig(func=_vlm_func)
        else:
            role_configs[spec.name] = RoleLLMConfig()
    return build_role_rag(
        tmp_path,
        workspace=f"enrich-offload-{tmp_path.name}",
        llm_model_func=_vlm_func,
        role_llm_configs=role_configs,
        vlm_process_enable=True,
    )


def _write_fixtures(tmp_path: Path) -> tuple[str, dict]:
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()

    image_path = parsed_dir / "fig1.png"
    # 64x64 RGBA PNG is enough to pass the min-pixel gate; content is unused
    # because the VLM func is mocked. A trivially valid PNG is fine here.
    from tests.pipeline.test_pipeline_analyze_multimodal import PNG_BYTES

    image_path.write_bytes(PNG_BYTES)

    blocks_path = parsed_dir / "doc.blocks.jsonl"
    blocks_path.write_text(
        json.dumps({"type": "meta", "doc_id": "doc-1"}) + "\n",
        encoding="utf-8",
    )

    sidecar_path = parsed_dir / "doc.drawings.json"
    sidecar_path.write_text(
        json.dumps(
            {"drawings": {"im-001": {"caption": "Figure 1", "path": str(image_path)}}}
        ),
        encoding="utf-8",
    )
    return "doc-1", {"blocks_path": str(blocks_path)}


@pytest.mark.asyncio
async def test_enrich_sidecars_runs_via_to_thread(tmp_path, monkeypatch):
    rag = _build_rag(tmp_path)
    await rag.initialize_storages()
    try:
        doc_id, parsed_data = _write_fixtures(tmp_path)

        offloaded: list = []
        real_to_thread = asyncio.to_thread

        async def _spy_to_thread(func, /, *args, **kwargs):
            offloaded.append(func)
            return await real_to_thread(func, *args, **kwargs)

        monkeypatch.setattr("lightrag.pipeline.asyncio.to_thread", _spy_to_thread)

        await rag.analyze_multimodal(
            doc_id=doc_id,
            file_path="fixture.pdf",
            parsed_data=parsed_data,
            process_options="i",
        )

        # The sidecar enrichment must be dispatched off the event loop.
        assert enrich_sidecars_with_surrounding in offloaded
    finally:
        await rag.finalize_storages()
