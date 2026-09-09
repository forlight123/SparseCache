# Local reference import, 2026-09-09

These files were imported from the adjacent, user-owned CacheDraft workspace.
They are clean-room research implementations, not author-released KVShot code.
Source workspace HEAD: `6b84f8c7e109eb0f626f28635fc075a85a24b99f`.
The hashes below identify the actual working-tree files used (HEAD alone does
not establish that the source worktree was clean).

| File | Source SHA-256 |
|---|---|
| `experiments/kvshot/model.py` | `286de3e119dbb0c42c5596e2466abcdde9f30e1563fe071196c9e8dfe0974309` |
| `experiments/kvshot/chat_data.py` | `918bde03cd75bf58f0d73a3c664511ff047f2b49d291a3e9f7208f95c5edab4d` |
| `experiments/blockdraft/model.py` | `02de24a077ed7b2e7ef2f7c52fdf46b6d8cb51593eec3e4d5a6d6e42d50ae787` |

The import normalized one extra trailing blank line in `kvshot/model.py`;
its imported hash is
`2ed298d78a8fe44f75251c93d3a1b2fd1ef50c1616dd76d3061bce3f555dbe80`.
No executable content differs. The other two imported hashes match their sources.

The KVShot reference retains its historical 32K vocabulary and optional
EAGLE weight initialization for reproducibility. It does NOT take target
hidden states at inference. The existing checkpoint named `dense4` actually
has TWO draft layers: its name refers to training density, not model depth.

The block reference supports several legacy variants. SparseCache rejects the
seed-hidden/EAGLE-style runtime input. Its inherited checkpoint uses five KV
layers, a DFlash-initialized block backbone, full target vocabulary, and a
KV-to-memory projection. SparseCache adds a causal correction GRU that consumes
only prior proposal-token embeddings and sparse-KV block features; it still
receives no target hidden history. Its runtime input is KV and token embeddings,
and the internal projected memory is NOT a target hidden-history input.

Model weights and datasets remain external assets and are never copied into
Git. The imported implementations now run from SparseCache's own `.venv`.
Sparse positions and arrival-stage training are implemented separately in
`experiments/lossless_pd/pilot.py`, preserving the imported references.
