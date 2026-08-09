# Corpus provenance (A1 — fill this)

- Source (URL): https://archive.org/details/byte-magazine
  Individual issues fetched via Internet Archive identifiers of the form `byte-magazine-YYYY-MM` (e.g. `byte-magazine-1993-06`), using the `ia` CLI in scripts/get_data.sh.

- Licence / usage rights: Permitted for Educational purposes
  No explicit license/rights statement is published on the Internet Archive item pages for this collection (checked directly — no CC tag, no public-domain notice). Treated as all-rights-reserved. We do not redistributescans in this repo — only a fetch script pointing to archive.org is committed; anyone reproducing this project re-downloads directly from IA. 

- Pages: [RUN: `find data/raw -name '*.pdf' | xargs -I{} pdfinfo {} | grep Pages | awk '{s+=$2} END {print s}'`]
  Words: [RUN: after OCR/text extraction — word count is only knowable once text is extracted; log an estimate here for now and update post-OCR]
  Size on disk: [RUN: `du -sh data/raw/`]

- Scan/script difficulty notes:
  BYTE magazine pages use a dense 2-3 column layout throughout, with sidebars, boxed code listings, and full-page ads that interrupt normal reading order. Naive left-to-right/top-to-bottom OCR would scramble text across columns. Print quality varies significantly across the ~1975-1993 span in our subset — early issues (1975-1980) show more visible print/scan artifacts (halftone dot patterns in images, lower contrast body text) than later issues. This is why a layout/region-detection stage runs before OCR (see vision/layout.py) rather than OCR-ing raw pages directly.

- Split policy (by document):
  Not applicable in the traditional train/test sense — per A1, this projectuses pretrained models only (no fine-tuning), so there is no training split. Instead: `grading_kit/heldout_pages/` holds a small set of pages set aside and never used to tune thresholds/prompts/parameters — these are reserved solely for honest OCR/retrieval evaluation (see labels.jsonl).