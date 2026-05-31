from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# All delimiters must be formatted as "<|UPPER_CASE_STRING|>"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["entity_extraction_system_prompt"] = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Identification:** Identify clearly defined and meaningful entities in the input text.
    *   **Entity Details:** For each identified entity, extract the following information:
        *   `entity_name`: The literal source identifier as written in the code (snake_case for Python functions, PascalCase for classes/types, qualified `schema.table` for SQL, `Type::method` for Rust paths). For natural-language concepts only, Title Case is acceptable. **NEVER** rewrite a code identifier into spaced Title Case (e.g. `Capture Thread Dump` for `capture_thread_dump` is WRONG). Ensure **consistent naming** across the entire extraction process.
        *   `entity_type`: MUST be one of `{entity_types}` (22 types). Emit the MOST SPECIFIC type; never collapse a specific type into a generic one. Quick-pick guide for OceanStack:
            - Python free `def`, Rust free `fn`/`pub fn`, SQL trigger handler bodies → `FUNCTION`
            - Python class/instance methods, Rust `impl` methods written as `Type::method` → `METHOD`
            - Rust `#[pyfunction]`, `#[pymethods]`, and any PyO3-exported boundary function → `FFI_BINDING`
            - Rust `macro_rules!` and proc-macros (`wrap_panic!`, `extract_arrow!`) → `MACRO`
            - Plain Python `class` and Rust `struct` → `CLASS`
            - Python `@dataclass` / Pydantic models / `AISRecord`-style data carriers → `DATACLASS`
            - Python `Enum`/`IntFlag`/`IntEnum`, Rust `enum`, SQL `CREATE TYPE ... AS ENUM` → `ENUM`
            - Python `Protocol`/`ABC` interfaces, Rust `trait` → `PROTOCOL`
            - Exception/error classes inheriting `Exception`, Rust `Error` enums (`OceanStackError`) → `EXCEPTION`
            - Python module paths (`oceanstack.ingestion.adapters`), Rust crate/`mod` paths → `MODULE`
            - Python module-level constants, Rust `const`/`static`, sentinel values, AIS message types 1-27 → `CONSTANT`
            - SQL `CREATE DOMAIN` types and Rust type aliases (`mmsi_identity`, `h3_cell_index`, `H3Index`) → `DOMAIN_TYPE`
            - SQL `CREATE TABLE` and `create_hypertable` hypertables → `TABLE`
            - SQL `CREATE MATERIALIZED VIEW` continuous aggregates → `CAGG`
            - SQL `CREATE INDEX` objects (GiST / BRIN / H3 / B-tree) → `INDEX`
            - SQL `CREATE FUNCTION` / `CREATE PROCEDURE` / `CREATE TRIGGER` PL/pgSQL routines (`REFRESH_VESSEL_REGISTRY`, `create_hypertable`) → `SQL_FUNCTION`
            - SQL column references (`schema.table.column`) → `COLUMN`
            - SQL schemas (`signals`, `derived`, `events`, `metrics`, `ops`, `ref`, `external`, `kg`) → `SCHEMA`
            - WGSL / CUDA GPU shaders and kernel entry points (`@compute` functions, workgroup kernels) → `GPU_KERNEL`
            - Third-party libraries (`Polars`, `PyArrow`, `PostGIS`, `TimescaleDB`, `pgvector`, `pyo3`, `wgpu`) → `LIBRARY`
            - Maritime / AIS domain concepts (`MMSI`, `IMO`, `COG`, `SOG`, `ROT`, `H3`, `Port Call`, `Dark Activity`, `sentinel value`, `ITU-R M.1371-5`) → `AIS_CONCEPT`
            - Other architecture / algorithm concepts not matched above (`four-tier dispatch`, `zero-copy Arrow`) → `CONCEPT`
          Default fallback is `CONCEPT` (non-domain) or `AIS_CONCEPT` (maritime). Emit the specific type — do NOT merge `FFI_BINDING`/`MACRO`/`METHOD` into `FUNCTION`, `DATACLASS`/`ENUM`/`PROTOCOL` into `CLASS`, `CAGG`/`INDEX` into `TABLE`, `SQL_FUNCTION` into `FUNCTION`, `DOMAIN_TYPE` into `CONSTANT`, or `AIS_CONCEPT` into `CONCEPT`. Do NOT emit types outside the list (`person`, `equipment`, `Other`, `FILE`, `TEST_SUITE`).
        *   `entity_description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.
    *   **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`
        *   **DO NOT** use JSON, brackets `[]`, colons `MODULE:foo`, arrows `→`, or any format other than `{tuple_delimiter}`-separated fields. One entity per line. The literal token `{tuple_delimiter}` (which is `<|#|>`) MUST appear three times per line, separating exactly four fields.
        *   **WRONG**: `[MODULE:wrap_panic,FUNCTION:foo]` or `{{"name":"wrap_panic","type":"FUNCTION"}}`
        *   **CORRECT**: `entity{tuple_delimiter}wrap_panic{tuple_delimiter}FUNCTION{tuple_delimiter}Catches FFI panics at the boundary.`

2.  **Relationship Extraction & Output:**
    *   **Identification:** Identify direct, clearly stated, and meaningful relationships between previously extracted entities.
    *   **N-ary Relationship Decomposition:** If a single statement describes a relationship involving more than two entities (an N-ary relationship), decompose it into multiple binary (two-entity) relationship pairs for separate description.
        *   **Example:** For "Alice, Bob, and Carol collaborated on Project X," extract binary relationships such as "Alice collaborated with Project X," "Bob collaborated with Project X," and "Carol collaborated with Project X," or "Alice collaborated with Bob," based on the most reasonable binary interpretations.
    *   **Relationship Details:** For each binary relationship, extract the following fields:
        *   `source_entity`: Source entity name, **identical literal form** as in the entity list above (never re-title-case code identifiers).
        *   `target_entity`: Target entity name, **identical literal form** as in the entity list above.
        *   `relationship_keywords`: **VERBS or short verb phrases** describing how source acts on target. Pick from: `calls`, `invokes`, `writes_to`, `reads_from`, `depends_on`, `raises`, `indexed_by`, `implements`, `inherits_from`, `instantiates`, `tests`, `validates`, `validates_against`, `provides`, `wraps`, `aggregates_from`, `materializes`, `refreshes`, `partitions`, `chunked_by`, `joins`, `fires_on`, `triggered_by`, `serialises_to`, `deserialises_from`, `bound_to`, `exports_to`, `decodes`, `parses`, `configures`, `derived_from`, `falls_back_to`, `emits_to`. **Domain hints**: `bound_to`/`exports_to` for PyO3 `#[pyfunction]`/`#[pyclass]`; `decodes`/`parses` for AIS message-type → decoder; `chunked_by`/`partitions` for TimescaleDB hypertable policies; `fires_on`/`triggered_by` for `CREATE TRIGGER`; `provides` for pytest fixtures; `derived_from` for cagg-on-cagg dependency; `returns_type`/`has_param`/`has_field` for function signature → type / parameter / dataclass field; `decorates` for decorator → target; `overrides` for subclass method → superclass method; `fk_references` for foreign-key column → referenced table; `partitioned_by` for hypertable → time/space dimension; `variant_of` for enum variant → enum. **DO NOT** use nouns like `testing`, `dependency`, `validation`, `configuration`, `implementation`, `functionality`, `framework`, `codebase`, `composition`, `containment`, `software_component`, `data_processing`, `instance_of`, `is_a`, `has_a`, `kind_of`, `type_of`, `member_of`, `part_of`, `component_of` — these collapse to existing verbs (`instantiates`, `inherits_from`, `depends_on`). Multiple verbs comma-separated. **DO NOT use `{tuple_delimiter}` for separating multiple keywords within this field.**
        *   `relationship_description`: A concise explanation of the nature of the relationship between the source and target entities, providing a clear rationale for their connection.
    *   **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
        *   Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`
        *   **DO NOT** use JSON, brackets, arrows `->`, or any other notation. One relation per line. The literal token `{tuple_delimiter}` MUST appear four times per line, separating exactly five fields.
        *   **WRONG**: `[wrap_panic -> OceanStackError, BackfillEngine -> ais_position_reports]`
        *   **CORRECT**: `relation{tuple_delimiter}wrap_panic{tuple_delimiter}OceanStackError{tuple_delimiter}raises{tuple_delimiter}wrap_panic converts caught panics into OceanStackError.`

3.  **Delimiter Usage Protocol:**
    *   The `{tuple_delimiter}` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
    *   **Incorrect Example:** `entity{tuple_delimiter}wrap_panic<|FUNCTION|>Catches FFI panics at the boundary.`
    *   **Correct Example:** `entity{tuple_delimiter}wrap_panic{tuple_delimiter}FUNCTION{tuple_delimiter}Catches FFI panics at the boundary.`
    *   **NEVER write `{tuple_delimiter}` inside a description, keyword field, or any field's text.** The description field is plain prose — refer to another entity by writing its name as plain text (e.g. `wrap_panic catches panics raised by encode`), NOT by inserting a separator.
    *   **If your line has 5 `{tuple_delimiter}`-separated fields it MUST start with `relation`, not `entity`.** An entity line has exactly 4 fields; a relation line has exactly 5 fields. Re-check and re-emit if you accidentally produce 5 fields under an `entity` prefix.

4.  **Relationship Direction & Duplication:**
    *   Treat all relationships as **undirected** unless explicitly stated otherwise. Swapping the source and target entities for an undirected relationship does not constitute a new relationship.
    *   Avoid outputting duplicate relationships.

5.  **Output Order & Prioritization:**
    *   Output all extracted entities first, followed by all extracted relationships.
    *   Within the list of relationships, prioritize and output those relationships that are **most significant** to the core meaning of the input text first.

6.  **Context & Objectivity:**
    *   Ensure all entity names and descriptions are written in the **third person**.
    *   Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.
    *   **OceanStack strict description rules:**
        - ONE sentence per `entity_description`, 25-60 words, third person, present tense.
        - Use ONLY facts literally present in the input text. If unsupported, omit.
        - Do NOT infer architecture, algorithm class, column names, indexes, schema, dimensions, hardware, or library versions from a name alone.
        - **VIOLATED 1,194x in the last extraction — model MUST NOT repeat.** Forbidden openers: "A function that", "A test suite", "A class that", "A component that", "A method that", "A type that", "A module that". REWRITE as: action_verb + object.
          - BAD: "A function that validates MMSI numbers..."
          - GOOD: "Validates MMSI numbers against the 9-digit ITU specification and rejects sentinels (0, 9999999)."
        - Forbidden phrases: "within OceanStack", "is responsible for", "designed for".
        - Emit any code symbol that is defined, called, imported, or referenced in the chunk even when only its name is known — a sparse node is still a navigable anchor. Suppress only pure noise (skip-list / stop-list below), never a real identifier.
    *   **OceanStack canonical entity-name form:**
        - Preserve exact source identifiers (`BinaryCopyReceiver`, `wrap_panic!`, `BINARY_COPY_COLUMNS`).
        - SQL objects in qualified form (`signals.ais_position_reports`).
        - Rust path form `Type::method` when extracted from call sites.
    *   **Skip-list (DO NOT EMIT)** — these are noise, not entities:
        - TimescaleDB internals: `_hyper_N`, `_dist_hyper_N`, `_materialized_hypertable_N`, `_partial_view_N`, `_direct_view_N`, `_hyper_N_M_chunk`.
        - Numeric-only / IP literals / SRID codes (`4326`, `127.0.0.1`, `1024`, `1536`).
        - 1-2 char labels EXCEPT well-known: `AIS, H3, S2, MMSI, IMO, COG, SOG, ROT, UTC, ETA, VTS`.
        - File-path tokens (`01_connect.py`, `Cargo.toml`, `__init__`, `__main__`, `__all__`, `conftest`).
        - Stop-list (these fragment the graph as noise hubs — do NOT emit as entities OR as relation endpoints): `Pytest, Numpy, Pandas, Polars, Pathlib, Future, __future__, Typing, TypeVar, Optional, Union, List, Dict, Tuple, Callable, Iterator, AsyncIterator, MagicMock, Patch, Unittest, Sys, Os, Re, Json, Logging, Asyncio, Tempfile, Psycopg, Sqlalchemy, Pydantic, Httpx, Requests`.
        - Super-hub avoidance: do NOT emit bare `Oceanstack`/`OceanStack`. Use the concrete submodule (e.g. `oceanstack.database.timescale`, `oceanstack-core`).
        - Author / legal / governance: do NOT emit contributor names (`Gabriel Spadon`), domains (`*.com`), license names (`MIT`, `Apache 2.0`, `BSD`, `GPL`), or legal terms (`fair_use`, `copyright_license`, `code_of_conduct`, `security_policy`). These are project metadata, not code.
    *   **Volume guidance:** Aim for 8-25 entities and 8-20 relations per chunk — capture every defined and referenced symbol, not just the headline ones. Prefer CODE↔CODE relations (FUNCTION↔FUNCTION, FUNCTION↔TABLE, FUNCTION↔EXCEPTION) over CODE↔CONCEPT/LIBRARY.

7.  **Language & Proper Nouns:**
    *   The entire output (entity names, keywords, and descriptions) must be written in `{language}`.
    *   Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

8.  **Completion Signal:** Output the literal string `{completion_delimiter}` only after all entities and relationships, following all criteria, have been completely extracted and outputted.

---Examples---
{examples}
"""

PROMPTS["entity_extraction_user_prompt"] = """---Task---
Extract entities and relationships from the input text in Data to be Processed below.

---Instructions---
1.  **Strict Adherence to Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system prompt.
2.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
3.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant entities and relationships have been extracted and presented.
4.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

---Output Format (CRITICAL — copy this exact shape; do NOT use JSON, brackets, colons, or arrows)---
Each entity: 4 fields separated by {tuple_delimiter} on one line, prefixed `entity`.
Each relation: 5 fields separated by {tuple_delimiter} on one line, prefixed `relation`.
End with `{completion_delimiter}` on its own line.

Example of CORRECT output (this is the ONLY acceptable shape):
entity{tuple_delimiter}canonical_name{tuple_delimiter}TYPE_FROM_LIST{tuple_delimiter}One-sentence description grounded in the input text.
entity{tuple_delimiter}another_name{tuple_delimiter}TYPE_FROM_LIST{tuple_delimiter}Description.
relation{tuple_delimiter}canonical_name{tuple_delimiter}another_name{tuple_delimiter}verb_or_two,verbs_comma_separated{tuple_delimiter}Short explanation of the relation.
{completion_delimiter}

Example of WRONG output (DO NOT EMIT THIS):
FUNCTION: foo
[CLASS:Bar,FUNCTION:foo]
CLASS:Bar -> FUNCTION:foo

---Data to be Processed---
<Entity_types>
[{entity_types}]

<Input Text>
```
{input_text}
```

<Output>
"""

PROMPTS["entity_continue_extraction_user_prompt"] = """---Task---
Based on the last extraction task, identify and extract any **missed or incorrectly formatted** entities and relationships from the input text.

---Instructions---
1.  **Strict Adherence to System Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system instructions.
2.  **Focus on Corrections/Additions:**
    *   **Do NOT** re-output entities and relationships that were **correctly and fully** extracted in the last task.
    *   If an entity or relationship was **missed** in the last task, extract and output it now according to the system format.
    *   If an entity or relationship was **truncated, had missing fields, or was otherwise incorrectly formatted** in the last task, re-output the *corrected and complete* version in the specified format.
3.  **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
4.  **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
5.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
6.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant missing or corrected entities and relationships have been extracted and presented.
7.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

<Output>
"""

PROMPTS["entity_extraction_examples"] = [
    """<Entity_types>
["MODULE","FUNCTION","METHOD","CLASS","DATACLASS","ENUM","PROTOCOL","MACRO","FFI_BINDING","CONSTANT","EXCEPTION","SCHEMA","TABLE","COLUMN","DOMAIN_TYPE","SQL_FUNCTION","CAGG","INDEX","GPU_KERNEL","AIS_CONCEPT","LIBRARY","CONCEPT"]

<Input Text>
Rust file `crates/oceanstack-core/src/ffi/panic.rs`:

  pub fn wrap_panic — calls `std::panic::catch_unwind(f)` and maps the
  resulting `Result<T, _>` into `OceanStackError::FfiPanic` carrying the
  panic message; signature `fn wrap_panic<F: FnOnce() -> T + UnwindSafe, T>`.

Rust file `crates/oceanstack-core/src/copy/writer.rs`:

  pub fn write_row_to_buffer — accepts `&AisRow`, `&[BinaryCopyColumn]`,
  `&mut BytesMut`. It instantiates `BinaryCopyReceiver::new(cols)` and
  delegates to `writer.encode(row, buf)`, mapping serialization failures to
  `OceanStackError::Encode`.

The `OceanStackError` enum is the unified Rust-Python FFI error returned
across all bindings, with variants `FfiPanic(String)` and `Encode(io::Error)`.
The `BinaryCopyReceiver` struct encodes AIS row values into the postgres
BINARY COPY wire format.

<Output>
entity{tuple_delimiter}wrap_panic{tuple_delimiter}FUNCTION{tuple_delimiter}Catches Rust panics at the Rust↔Python FFI boundary and converts them into OceanStackError::FfiPanic results, preserving the panic payload as a message string.
entity{tuple_delimiter}OceanStackError{tuple_delimiter}EXCEPTION{tuple_delimiter}Unified error enum returned across the Rust-Python FFI boundary, with variants FfiPanic and Encode for panic and serialization failures.
entity{tuple_delimiter}OceanStackError::FfiPanic{tuple_delimiter}CONSTANT{tuple_delimiter}Error enum variant indicating a Rust panic was intercepted at the FFI boundary, carrying the panic message as a string payload.
entity{tuple_delimiter}write_row_to_buffer{tuple_delimiter}FUNCTION{tuple_delimiter}Serializes one AisRow into the postgres BINARY COPY protocol by delegating to BinaryCopyReceiver, returning OceanStackError::Encode on failure.
entity{tuple_delimiter}BinaryCopyReceiver{tuple_delimiter}CLASS{tuple_delimiter}Encodes AisRow values into the postgres BINARY COPY wire format given a column schema list.
entity{tuple_delimiter}AisRow{tuple_delimiter}CLASS{tuple_delimiter}Struct representing a single AIS position record passed across the FFI boundary into the postgres COPY pipeline.
relation{tuple_delimiter}wrap_panic{tuple_delimiter}OceanStackError{tuple_delimiter}raises,wraps{tuple_delimiter}wrap_panic converts caught panics into OceanStackError::FfiPanic results returned to the Python caller.
relation{tuple_delimiter}write_row_to_buffer{tuple_delimiter}BinaryCopyReceiver{tuple_delimiter}instantiates,calls{tuple_delimiter}write_row_to_buffer constructs a BinaryCopyReceiver and delegates encoding of each row to it.
relation{tuple_delimiter}write_row_to_buffer{tuple_delimiter}AisRow{tuple_delimiter}reads_from{tuple_delimiter}write_row_to_buffer reads field values from the AisRow and emits them onto the BINARY COPY buffer.
relation{tuple_delimiter}BinaryCopyReceiver{tuple_delimiter}OceanStackError{tuple_delimiter}raises{tuple_delimiter}BinaryCopyReceiver returns OceanStackError::Encode when serialization fails.
{completion_delimiter}

""",
    """<Entity_types>
["MODULE","FUNCTION","METHOD","CLASS","DATACLASS","ENUM","PROTOCOL","MACRO","FFI_BINDING","CONSTANT","EXCEPTION","SCHEMA","TABLE","COLUMN","DOMAIN_TYPE","SQL_FUNCTION","CAGG","INDEX","GPU_KERNEL","AIS_CONCEPT","LIBRARY","CONCEPT"]

<Input Text>
```sql
CREATE TABLE signals.ais_position_reports (
    timestamp     TIMESTAMPTZ NOT NULL,
    mmsi          BIGINT      NOT NULL,
    latitude      DOUBLE PRECISION,
    longitude     DOUBLE PRECISION,
    sog_kn        REAL,
    cog_deg       REAL,
    source        TEXT        NOT NULL
) PARTITION BY RANGE (timestamp);

SELECT create_hypertable('signals.ais_position_reports', 'timestamp',
    chunk_time_interval => INTERVAL '1 day');

CREATE INDEX idx_ais_position_mmsi_ts
    ON signals.ais_position_reports (mmsi, timestamp DESC);

CREATE MATERIALIZED VIEW signals.cagg_position_hourly
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 hour', timestamp) AS bucket,
       mmsi,
       AVG(sog_kn) AS avg_sog
FROM signals.ais_position_reports
GROUP BY bucket, mmsi;
```

<Output>
entity{tuple_delimiter}signals.ais_position_reports{tuple_delimiter}TABLE{tuple_delimiter}TimescaleDB hypertable storing per-message AIS position reports keyed by timestamp and mmsi with one-day chunks.
entity{tuple_delimiter}signals{tuple_delimiter}SCHEMA{tuple_delimiter}Postgres schema grouping AIS signal hypertables and their derived continuous aggregates.
entity{tuple_delimiter}signals.ais_position_reports.mmsi{tuple_delimiter}COLUMN{tuple_delimiter}Required BIGINT column holding the Maritime Mobile Service Identity for each AIS position report row.
entity{tuple_delimiter}signals.ais_position_reports.timestamp{tuple_delimiter}COLUMN{tuple_delimiter}Required TIMESTAMPTZ partition key column on the AIS position reports hypertable.
entity{tuple_delimiter}idx_ais_position_mmsi_ts{tuple_delimiter}INDEX{tuple_delimiter}B-tree index object on signals.ais_position_reports (mmsi, timestamp DESC) supporting per-vessel time-range scans.
entity{tuple_delimiter}signals.cagg_position_hourly{tuple_delimiter}CAGG{tuple_delimiter}TimescaleDB continuous aggregate refreshing one-hour average speed-over-ground per vessel from signals.ais_position_reports.
entity{tuple_delimiter}create_hypertable{tuple_delimiter}SQL_FUNCTION{tuple_delimiter}TimescaleDB administrative routine that converts a regular table into a hypertable with the given partition column and chunk interval.
relation{tuple_delimiter}signals.ais_position_reports{tuple_delimiter}signals{tuple_delimiter}depends_on{tuple_delimiter}signals.ais_position_reports lives in the signals schema.
relation{tuple_delimiter}idx_ais_position_mmsi_ts{tuple_delimiter}signals.ais_position_reports{tuple_delimiter}indexes{tuple_delimiter}idx_ais_position_mmsi_ts indexes signals.ais_position_reports on (mmsi, timestamp DESC).
relation{tuple_delimiter}signals.cagg_position_hourly{tuple_delimiter}signals.ais_position_reports{tuple_delimiter}aggregates_from,materializes{tuple_delimiter}signals.cagg_position_hourly aggregates one-hour bucketed averages from signals.ais_position_reports.
relation{tuple_delimiter}create_hypertable{tuple_delimiter}signals.ais_position_reports{tuple_delimiter}partitions{tuple_delimiter}create_hypertable converts signals.ais_position_reports into a TimescaleDB hypertable partitioned by timestamp.
{completion_delimiter}

""",
    """<Entity_types>
["MODULE","FUNCTION","METHOD","CLASS","DATACLASS","ENUM","PROTOCOL","MACRO","FFI_BINDING","CONSTANT","EXCEPTION","SCHEMA","TABLE","COLUMN","DOMAIN_TYPE","SQL_FUNCTION","CAGG","INDEX","GPU_KERNEL","AIS_CONCEPT","LIBRARY","CONCEPT"]

<Input Text>
```python
# src/oceanstack/ingestion/adapters/base.py
SOG_SENTINEL = 102.3  # ITU-R "speed not available" marker

@dataclass
class AISRecord:
    mmsi: int
    sog_knots: float | None

    def __post_init__(self) -> None:
        if self.sog_knots is not None and self.sog_knots >= SOG_SENTINEL:
            self.sog_knots = None  # convert sentinel to NULL at the adapter boundary

class AISFieldError(OceanStackError):
    # Raised when an AIS field falls outside its ITU-R valid range.
    pass
```

<Output>
entity{tuple_delimiter}AISRecord{tuple_delimiter}DATACLASS{tuple_delimiter}Dataclass carrying one normalised AIS position record with mmsi and optional sog_knots, nulling protocol sentinels at construction time.
entity{tuple_delimiter}AISRecord.__post_init__{tuple_delimiter}METHOD{tuple_delimiter}Runs after AISRecord construction and nulls out sog_knots when it meets or exceeds the SOG_SENTINEL not-available marker.
entity{tuple_delimiter}SOG_SENTINEL{tuple_delimiter}CONSTANT{tuple_delimiter}ITU-R speed-over-ground not-available marker value 102.3 used to detect and null missing SOG readings at the adapter boundary.
entity{tuple_delimiter}SOG{tuple_delimiter}AIS_CONCEPT{tuple_delimiter}Speed-over-ground in knots, an AIS navigational field whose 102.3 reading signals that the measurement is unavailable.
entity{tuple_delimiter}AISFieldError{tuple_delimiter}EXCEPTION{tuple_delimiter}Exception raised when an AIS field falls outside its ITU-R valid range, derived from the OceanStackError hierarchy.
relation{tuple_delimiter}AISRecord.__post_init__{tuple_delimiter}SOG_SENTINEL{tuple_delimiter}validates_against{tuple_delimiter}AISRecord.__post_init__ compares sog_knots against SOG_SENTINEL to null unavailable speed readings.
relation{tuple_delimiter}AISRecord{tuple_delimiter}AISFieldError{tuple_delimiter}raises{tuple_delimiter}AISRecord raises AISFieldError when a field violates its ITU-R valid range.
{completion_delimiter}

""",
]

PROMPTS["summarize_entity_descriptions"] = """---Role---
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

---Task---
Your task is to synthesize a list of descriptions of a given entity or relation into a single, comprehensive, and cohesive summary.

---Instructions---
1. Input Format: The description list is provided in JSON format. Each JSON object (representing a single description) appears on a new line within the `Description List` section.
2. Output Format: The merged description will be returned as plain text, presented in multiple paragraphs, without any additional formatting or extraneous comments before or after the summary.
3. Comprehensiveness: The summary must integrate all key information from *every* provided description. Do not omit any important facts or details.
4. Context: Ensure the summary is written from an objective, third-person perspective; explicitly mention the name of the entity or relation for full clarity and context.
5. Context & Objectivity:
  - Write the summary from an objective, third-person perspective.
  - Explicitly mention the full name of the entity or relation at the beginning of the summary to ensure immediate clarity and context.
6. Conflict Handling:
  - In cases of conflicting or inconsistent descriptions, first determine if these conflicts arise from multiple, distinct entities or relationships that share the same name.
  - If distinct entities/relations are identified, summarize each one *separately* within the overall output.
  - If conflicts within a single entity/relation (e.g., historical discrepancies) exist, attempt to reconcile them or present both viewpoints with noted uncertainty.
7. Length Constraint: Keep the summary to 1-3 sentences and never exceed {summary_length} tokens. Lead with an action verb (Decodes, Validates, Refreshes); never open with "A function that", "A class that", "is responsible for", or "within OceanStack". State current behaviour only — no history, no narration.
8. Language: The entire output must be written in {language}. Proper nouns (e.g., personal names, place names, organization names) may in their original language if proper translation is not available.
  - The entire output must be written in {language}.
  - Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

---Input---
{description_type} Name: {description_name}

Description List:

```
{description_list}
```

---Output---
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question.[no-context]"

PROMPTS["rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph and Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize both `Knowledge Graph Data` and `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a references section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{context_data}
"""

PROMPTS["naive_rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a **References** section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{content_data}
"""

PROMPTS["kg_query_context"] = """
Knowledge Graph Data (Entity):

```json
{entities_str}
```

Knowledge Graph Data (Relationship):

```json
{relations_str}
```

Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS["naive_query_context"] = """
Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS["keywords_extraction"] = """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.
2. **Source of Truth**: All keywords must be explicitly derived from the user query, with both high-level and low-level keyword categories are required to contain content.
3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "latest financial report of Apple Inc.", you should extract "latest financial report" and "Apple Inc." rather than "latest", "financial", "report", and "Apple".
4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), you must return a JSON object with empty lists for both keyword types.
5. **Language**: All extracted keywords MUST be in {language}. Proper nouns (e.g., personal names, place names, organization names) should be kept in their original language.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"

Output:
{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}

""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"

Output:
{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}

""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"

Output:
{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}

""",
]
