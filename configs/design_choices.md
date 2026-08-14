# Per-stage design choices (A2 deliverable). Fill every cell.
| Stage | Problem statement | Data | Model | Methods | Design | Development | Deployment | MLOps |
|---|---|---|---|---|---|---|---|---|
| 0 Frame | Answer grounded questions about scanned BYTE Magazine issues for a retro-computing researcher/enthusiast, citing the source page |Public-domain BYTE Magazine scans (Internet Archive), multi-column layout as our data speciality  |N/A (framing stage)  |RAG over an agentic pipeline  |Scalability is our primary NFR (p95 ≤1.5s/query) — this constrains every downstream model/index choice toward speed |TBD  |	Deferred to A4 |Deferred to A4|
| 1 Ingest+Enhance |Load raw PDF pages into a consistent Page schema for downstream stages |15 BYTE issues, 6,603 pages, page dimensions vary ~7% (Section 3 finding) |N/A — no enhancement model used|fitz/PyMuPDF page rendering; enhancement deliberately not implemented (enhance: {enabled: false})  |Enhancement skipped to preserve latency budget under our scalability NFR — trading robustness to noisy scans for speed |TBD |Deferred to A4 |Deferred to A4 |
| 2 Layout |  |  |  |  |  |  |  |  |
| 3 OCR |  |  |  |  |  |  |  |  |
| 4 Index |  |  |  |  |  |  |  |  |
| 5 Retrieval |  |  |  |  |  |  |  |  |
| 6 Agent |  |  |  |  |  |  |  |  |
| 7 RL/RLVR |  |  |  |  |  |  |  |  |
| 8 Serving |  |  |  |  |  |  |  |  |
| 9 Eval |  |  |  |  |  |  |  |  |
