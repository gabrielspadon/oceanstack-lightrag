# OceanStack entity-type extraction profiles

Data-driven entity-extraction customization for the LightRAG KG builder
(upstream v1.5.4 `resolve_entity_extraction_prompt_profile`). These profiles
replace the built-in general taxonomy WITHOUT editing `lightrag/prompt.py`, so
the fork stays close to upstream and survives future prompt refactors.

## Deploy

This profile ships tracked as `prompts/samples/oceanstack.entity_type.yml`.
The runtime dir `prompts/entity_type/` is gitignored (upstream treats it as
user-supplied runtime state), so copy the profile into place at deploy:

    cp prompts/samples/oceanstack.entity_type.yml \
       "$PROMPT_DIR"/entity_type/oceanstack.yml

## Wiring

Set on the deployments that build a CODE / OceanStack graph (code-kg,
oceanstack-kg):

- `PROMPT_DIR` = directory containing this `entity_type/` subdir (defaults to
  `./prompts`).
- `ENTITY_TYPE_PROMPT_FILE=oceanstack.yml` (file name only; loaded from
  `PROMPT_DIR/entity_type/`), or pass
  `addon_params={"entity_type_prompt_file": "oceanstack.yml"}`.

`oceanstack.yml` supplies:
- `entity_types_guidance` — the 22-type OceanStack code/data taxonomy, the
  controlled relationship-verb vocabulary, and code-identifier naming rules.
  Injected verbatim at `{entity_types_guidance}` (delimiter refs are literal
  `<|#|>`, not `{tuple_delimiter}`, because guidance is substituted, not
  re-formatted).
- `entity_extraction_examples` — OceanStack few-shot examples (these DO keep
  `{tuple_delimiter}`/`{completion_delimiter}` placeholders; the resolver
  `.format()`s them).

## Do NOT use for the personal email KG

The XTX Hermes email KG is personal (email/calendar/family), not code. It keeps
upstream's general taxonomy. Only code-kg and oceanstack-kg point
`ENTITY_TYPE_PROMPT_FILE` at `oceanstack.yml`.
