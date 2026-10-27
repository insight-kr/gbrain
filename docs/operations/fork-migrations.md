# Fork migration policy

Fork changes use stable identifiers, starting with
`sinaxis-001-google-discovery-embedding`, rather than upstream version numbers.
Each receipt lives in the existing config table at
`fork_migration.<identifier>` with the exact value `applied`. This avoids adding
a table or changing upstream's `config.version`, registry, latest version or
golden records. Otherwise an unrelated upstream migration reusing the fork's
number could be silently skipped on a persistent brain.

Initialization prepares forward-referenced columns before schema replay, then
runs every pending upstream migration, then fork migrations. The bootstrap only
prepares the discovery column; it never stamps a fork receipt. The cheap pending
probe checks the upstream version, fork receipts and the discovery column/index
shape and registry. An applied receipt still requires `vector(1536)`, its
Google registry descriptor and a valid HNSW cosine index on that column. A fork
receipt is written only after its DDL and index verification succeed. PostgreSQL builds the index
concurrently; PGLite uses ordinary DDL. The engines' unused numeric statement tag
is zero for fork DDL and does not allocate an upstream version.

Discovery uses `vector(1536)` and the Google `gemini-embedding-2` registry
descriptor with 1536 dimensions. Both indexing and queries take their output
dimension from the resolved column descriptor. The upstream Google recipe
default remains independent. Runtime primary-width substitution changes only
`embedding`; Discovery remains 1536 on both engines.

No persistent brain had received this fork when its initial 768-dimensional
definition was replaced. An experimental 768-dimensional brain, including one
with an `applied` receipt, fails schema verification explicitly; its vectors,
registry and receipt are not converted or regenerated automatically. A deployed
width change would require a separate migration.

<a id="discovery_schema_mismatch"></a>

The refusal code is `discovery_schema_mismatch`, including the startup safety
notice. CLI startup retains its general migration warning policy, allowing
unrelated Wiki text operations. Discovery chunk writes (inline and segmented),
vector reads and query/indexing dimension resolution check the same read-only
probe at use time. Text and image Discovery queries refuse before calling a
provider; no Wiki-vector fallback is allowed for a mismatched Discovery space.
Backfill readiness also checks its active column before any provider health
probe, including a drain blocked on an unsealed projection.

Every fork DDL is additive and repeatable (`IF NOT EXISTS`, registry merge that
preserves existing entries). A missing receipt can safely replay it. If upstream
adopts the same DDL under its own number, that migration runs normally and finds
the objects already present. Keep the old fork receipt as history; on the one-time
sync, retire the fork runner entry only after the adopted migration covers the
same schema and registry behavior. An unrelated upstream migration always keeps
its own number and runs normally. Never rewrite an upstream version to claim
fork completion. This policy does not automatically repair brains previously
stamped with an experimental fork occupying an upstream number.

Only the exact known receipt and exact 1536-dimensional discovery registry seed
count as system seed during graduation. Edited values, unknown receipt keys and custom registry
entries remain user data. Engine migration does not copy fork receipts: each
target earns its own receipt by running initialization.

Fork failures surface in the CLI notice, advisor and apply-migrations output.
Doctor schema_version and readiness inspect only the upstream version, so they
do not certify that fork migrations succeeded.

**Say to your agent:** *"Check that my fork receipts are applied without advancing
the upstream schema version."*
