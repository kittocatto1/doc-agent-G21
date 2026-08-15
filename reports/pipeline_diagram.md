# Knowledge-Base Pipeline Diagram

## BYTE Retriever — Knowledge-Base Construction Pipeline

```text
                    ┌──────────────────────────┐
                    │   BYTE Magazine Pages    │
                    │  (scanned page images)   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │         Ingest           │
                    │  Load ordered page data  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       Preprocess         │
                    │  Basic image preparation │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                
                    ┌──────────────────────────┐
                    │      Layout Detection    │
                    │ DocLayout-YOLO /         │
                    │     DocStructBench       │
                    │                          │
                    │ score_thr = 0.25         │
                    └────────────┬─────────────┘
                                 │
                                 │ detected regions
                                 │ + reading order
                                 ▼
                    ┌──────────────────────────┐
                    │           OCR            │
                    │ Qwen2-VL-2B-Instruct     │
                    │                          │
                    │ Pretrained / zero-shot   │
                    │ No fine-tuning           │
                    └────────────┬─────────────┘
                                 │
                                 │ region-level
                                 │ Chunk objects
                                 ▼
                    ┌──────────────────────────┐
                    │          Chunk           │
                    │ Rule-based re-chunking   │
                    │                          │
                    │ 256 words/chunk          │
                    │ 32-word overlap          │
                    │                          │
                    │ Grouped by doc_id        │
                    │ Reading order preserved  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │          Embed           │
                    │ BAAI/bge-small-en-v1.5   │
                    │                          │
                    │ 384-dimensional vectors  │
                    │ Pretrained / zero-shot   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
             ┌──────────────────────────────────────┐
             │             Vector Store             │
             │                                      │
             │        FAISS IndexHNSWFlat           │
             │                                      │
             │ M = 32                               │
             │ efConstruction = 200                 │
             │ efSearch = 64                        │
             │                                      │
             │ Incremental indexing                 │
             │ + doc_id duplicate protection        │
             └────────────────────┬─────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
          ┌───────────────────┐       ┌────────────────────┐
          │    index.faiss    │       │  chunks_meta.json  │
          │                   │       │                    │
          │ FAISS vectors +   │       │ Chunk text, doc_id │
          │ HNSW graph        │       │ page_ids, etc.     │
          └───────────────────┘       └────────────────────┘