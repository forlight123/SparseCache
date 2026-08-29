# Datasets

- `hotpotqa/` contains the local HotpotQA-E-style multi-passage JSONL used by
  the initial experiment.
- `musique/` contains the four official MuSiQue answer/full train/dev JSONL
  files that were already present in the workspace.
- `cases/` contains normalized, immutable experiment inputs. The initial case
  is HotpotQA-E row zero and has exactly ten full document chunks.

Dataset documents are treated as the output of a coarse RAG retriever. The
SparseCache experiments evaluate fine-grained KV reuse after document
retrieval; they do not measure retriever recall.
