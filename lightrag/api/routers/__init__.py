"""
This module contains route factories for the LightRAG API.

Greenfield serving exposes explicit immutable graph planes: the server
mounts only ``plane_routes.create_plane_routes``. The sibling
``document_routes`` module carries no HTTP surface at all; it is retained
purely as internal ingestion machinery (document manager, file enqueue,
file-variant cleanup). Routers are constructed per-app via
``create_*_routes`` factory functions rather than module-level singletons,
which would accumulate duplicate routes across repeated app construction.
"""

__all__: list[str] = []
