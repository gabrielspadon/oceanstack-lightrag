"""Read-only map-data routes backing the OceanStack maritime map view.

The knowledge-graph nodes carry no coordinates, so the geographic layers are
driven straight from the canonical AIS tables in the separate `oceanstack`
database: `external.world_ports` for ports and `derived.vessel_tracks` for vessel
positions and tracks, sampled evenly across the full history so the time slider
sweeps every year. The pool reuses the server's POSTGRES_* credentials with the
database overridden to `oceanstack`; every query is read-only and bounded.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query

from lightrag.utils import logger

from ..utils_api import get_combined_auth_dependency

router = APIRouter(tags=["map"])

_pool: asyncpg.Pool | None = None
_pool_lock = asyncio.Lock()

# The map samples a fixed number of equal-width time windows across the full
# vessel_tracks history so the client time slider sweeps every year, not just
# the most recent chunk. Constant per-window bounds let TimescaleDB exclude all
# but the matching chunks, so the whole sweep stays sub-second over 150M+ tracks.
_N_WINDOWS = 60
_time_range: tuple[int, int] | None = None


async def _track_time_range(pool: asyncpg.Pool) -> tuple[int, int] | None:
    """Return the cached (min, max+1) epoch bounds of `derived.vessel_tracks`.

    The history is append-only and the bound only grows, so a single lookup is
    cached for the process. Returns None when the table is empty.
    """
    global _time_range
    if _time_range is None:
        row = await pool.fetchrow(
            "SELECT min(start_time)::bigint AS lo, max(start_time)::bigint AS hi "
            "FROM derived.vessel_tracks"
        )
        if row is None or row["lo"] is None:
            return None
        _time_range = (int(row["lo"]), int(row["hi"]) + 1)
    return _time_range


def _windowed_union(
    select_cols: str, nonnull: str, lo: int, hi: int, per_window: int
) -> tuple[str, list[int]]:
    """Build a UNION ALL that pulls `per_window` rows from each of `_N_WINDOWS`
    equal time slices across [lo, hi).

    Each branch carries constant `start_time` bounds (passed as query params, not
    interpolated) so the planner keeps chunk exclusion and an early LIMIT per
    window — the sample is spread evenly across the whole history without scanning
    it. Returns the SQL and the positional param list for `pool.fetch`.
    """
    step = max(1, (hi - lo) // _N_WINDOWS)
    branches: list[str] = []
    params: list[int] = []
    p = 1
    for i in range(_N_WINDOWS):
        a = lo + i * step
        b = hi if i == _N_WINDOWS - 1 else a + step
        branches.append(
            f"(SELECT {select_cols} FROM derived.vessel_tracks "
            f"WHERE start_time >= ${p} AND start_time < ${p + 1} AND {nonnull} "
            f"LIMIT ${p + 2})"
        )
        params += [a, b, per_window]
        p += 3
    return " UNION ALL ".join(branches), params


async def _get_pool() -> asyncpg.Pool:
    """Return a lazily-created connection pool to the oceanstack AIS database."""
    global _pool
    if _pool is None:
        # Double-checked locking: the first burst of /map/* requests fires
        # concurrently, so without the lock several pools would be created and
        # all but one orphaned (leaking connections to the oceanstack DB).
        async with _pool_lock:
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
        limit: int = Query(120000, ge=1, le=300000),
    ) -> list[dict[str, Any]]:
        """Return vessel positions sampled evenly across the full track history.

        `end_time` is a BIGINT epoch (seconds) — already a JSON number that drives
        the client time slider, so positions reveal as the slider advances.
        """
        try:
            pool = await _get_pool()
            rng = await _track_time_range(pool)
            if rng is None:
                return []
            sql, params = _windowed_union(
                "mmsi, end_lon AS lon, end_lat AS lat, end_time",
                "end_lon IS NOT NULL AND end_lat IS NOT NULL",
                rng[0],
                rng[1],
                max(1, limit // _N_WINDOWS),
            )
            rows = await pool.fetch(sql, *params)
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Error fetching map vessels: {e}")
            raise HTTPException(status_code=500, detail=f"Error fetching vessels: {e}")

    @router.get("/map/tracks", dependencies=[Depends(combined_auth)])
    async def get_tracks(
        limit: int = Query(60000, ge=1, le=150000),
    ) -> list[dict[str, Any]]:
        """Return vessel tracks (start/end coordinate pairs) sampled across history.

        Each track carries `start_time` (BIGINT epoch) so the path layer can reveal
        with the same time slider as the vessel positions instead of drawing static.
        """
        try:
            pool = await _get_pool()
            rng = await _track_time_range(pool)
            if rng is None:
                return []
            sql, params = _windowed_union(
                "mmsi, start_lon, start_lat, end_lon, end_lat, start_time",
                "start_lon IS NOT NULL AND end_lon IS NOT NULL",
                rng[0],
                rng[1],
                max(1, limit // _N_WINDOWS),
            )
            rows = await pool.fetch(sql, *params)
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Error fetching map tracks: {e}")
            raise HTTPException(status_code=500, detail=f"Error fetching tracks: {e}")

    return router
