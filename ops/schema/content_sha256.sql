-- Adds content_sha256 to lightrag_doc_status, indexed for fast diff lookups.
-- Trigger mirrors hash from lightrag_doc_full.content (source of truth for ingested content).
-- Backfill at the end populates current rows.

SET lock_timeout = '5s';
SET statement_timeout = '60s';

BEGIN;

ALTER TABLE lightrag_doc_status
    ADD COLUMN IF NOT EXISTS content_sha256 VARCHAR(64);

CREATE OR REPLACE FUNCTION lightrag_doc_full_sync_sha()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.content IS NOT NULL THEN
        UPDATE lightrag_doc_status
        SET content_sha256 = ENCODE(SHA256(CONVERT_TO(NEW.content, 'UTF8')), 'hex')
        WHERE workspace = NEW.workspace AND id = NEW.id;
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS lightrag_doc_full_sha_sync ON lightrag_doc_full;
CREATE TRIGGER lightrag_doc_full_sha_sync
    AFTER INSERT OR UPDATE OF content ON lightrag_doc_full
    FOR EACH ROW EXECUTE FUNCTION lightrag_doc_full_sync_sha();

UPDATE lightrag_doc_status ds
SET content_sha256 = ENCODE(SHA256(CONVERT_TO(df.content, 'UTF8')), 'hex')
FROM lightrag_doc_full df
WHERE ds.workspace = df.workspace
  AND ds.id = df.id
  AND ds.content_sha256 IS NULL
  AND df.content IS NOT NULL;

COMMIT;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_lightrag_doc_status_workspace_sha
    ON lightrag_doc_status(workspace, content_sha256);
