"""
This module contains route factories for the LightRAG API.

The document/query/graph routers are intentionally NOT re-exported here:
they are constructed per-app via the `create_*_routes` factory functions
in their respective modules. A module-level singleton would accumulate
duplicate routes if the factory is invoked more than once in the same
process (e.g. across tests), which produced "Duplicate Operation ID"
warnings before the factories were converted to local routers.

Greenfield serving exposes explicit immutable graph planes. Ollama emulation
is not part of the public API and is intentionally not imported here.
"""

__all__: list[str] = []
