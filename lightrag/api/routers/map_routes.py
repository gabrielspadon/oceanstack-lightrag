"""Read-only map-data routes backing the OceanStack maritime map view.

The knowledge-graph nodes carry no coordinates, so the geographic layers are
driven straight from the canonical AIS tables in the separate `oceanstack`
database: `external.world_ports` for ports and `derived.vessel_tracks` for recent
vessel positions. The pool reuses the server's POSTGRES_* credentials with the
database overridden to `oceanstack`; every query is read-only and bounded.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query

from lightrag.utils import logger

from ..utils_api import get_combined_auth_dependency

router = APIRouter(tags=["map"])

_pool: asyncpg.Pool | None = None


async def _get_pool() -> asyncpg.Pool:
    """Return a lazily-created connection pool to the oceanstack AIS database."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            user=os.environ.get("POSTGRES_USER"),
            password=os.environ.get("POSTGRES_PASSWORD") or None,
            database=os.environ.get("OCEANSTACK_MAP_DB", "oceanstack"),
            min_size=1,
            max_size=4,
        )
    return _pool


def create_map_routes(api_key: Optional[str] = None):
    """Build the read-only map-data router (ports + recent vessel positions)."""
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get("/map/ports", dependencies=[Depends(combined_auth)])
    async def get_ports(
        limit: int = Query(5000, ge=1, le=20000),
    ) -> list[dict[str, Any]]:
        """Return world ports with coordinates for the map scatter layer."""
        try:
            pool = await _get_pool()
            rows = await pool.fetch(
                "SELECT port_id, name, country, longitude AS lon, latitude AS lat, harbor_size "
                "FROM external.world_ports "
                "WHERE longitude IS NOT NULL AND latitude IS NOT NULL "
                "LIMIT $1",
                limit,
            )
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Error fetching map ports: {e}")
            raise HTTPException(status_code=500, detail=f"Error fetching ports: {e}")

    @router.get("/map/vessels", dependencies=[Depends(combined_auth)])
    async def get_vessels(
        limit: int = Query(5000, ge=1, le=20000),
    ) -> list[dict[str, Any]]:
        """Return the most recent vessel positions (last track endpoint per track)."""
        try:
            pool = await _get_pool()
            # Bound on start_time (the hypertable partition column) so the planner
            # excludes all but the most recent chunks — a full ORDER BY over 151M
            # tracks would otherwise take ~20s. This yields a recent sample, not a
            # strict per-vessel latest, which is what the map overview needs.
            rows = await pool.fetch(
                "SELECT mmsi, end_lon AS lon, end_lat AS lat, end_time "
                "FROM derived.vessel_tracks "
                "WHERE start_time >= "
                "  (SELECT max(start_time) FROM derived.vessel_tracks) - 86400 * 3 "
                "  AND end_lon IS NOT NULL AND end_lat IS NOT NULL "
                "LIMIT $1",
                limit,
            )
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Error fetching map vessels: {e}")
            raise HTTPException(status_code=500, detail=f"Error fetching vessels: {e}")

    return router
