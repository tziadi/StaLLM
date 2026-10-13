# StaLLM manuscript — visual and editorial audit

Compared documents (2026-10-10):
- **Reference:** *Is a Microservice a Feature?* (`Microservices_SPL_Journal (37) (1).pdf`, 22 pages, elsarticle 5p).
- **Target:** `Publications/StaLLM___Journal/00main.pdf` before redesign (17 pages, elsarticle 5p).
  A copy is kept as `before/00main_before.pdf` in the session scratchpad.

Decision (user, 2026-10-10): keep the Elsevier `elsarticle` 5p/Times layout of the
reference paper; the target journal is chosen later.

## What makes the reference look polished

| Aspect | Reference | StaLLM (before) |
|---|---|---|
| Body font | Times (Nimbus) throughout | **Latin Modern** body (p. 1–17) with Times only in places: the `times` option is ignored because the local engine (Tectonic = XeTeX) uses Unicode fonts |
| Process figure | p. 4, Fig. 3: input → three numbered stages → output, an observation band, a legend, one palette | p. 5, Fig. 2: raster image, title inside the figure, content out of date (Snyk, DACOSX, "multi-oracle"), no controls |
| Figure fonts | figures share the document's sans/serif style | Fig. 3 (RQ4 plot) uses DejaVu Sans from matplotlib; Fig. 1 has tiny raster text |
| Figure width | wide figures span both columns; small diagrams stay in one column | Fig. 1 (p. 3) is a dense workflow squeezed into one column |
| Tables | booktabs, no redundant columns, 8–9 pt, short headers | Table 1 (p. 5) repeats the system name in a "Codebase" column; RQ3 table (p. 10) mixes 9 pt values with 5 pt intervals; RQ1/RQ2 headers "Place/File" are cryptic |
| Code/records | none; examples are in figures | Listing 1 (p. 5) breaks across columns and pages, 20 lines of key–value text |
| RQs | italic run-in paragraphs, numbered, each answered by an "Answer to RQ" paragraph | RQs listed once (p. 4); results use paragraph headings and bold-numbered "Key findings" lists of different length per RQ |
| Captions | 2–4 sentences, self-contained | some one-liners ("Overview of the empirical study design."), some long paragraphs (RQ3 table) |
| Emphasis | italics for terms, no boxes | grey `mdframed` prompt box (p. 3), bold inline headings, mixed `\texttt`/`\textsf` for features |
| Page composition | floats at page tops, no float-only pages | p. 6–7: three full-width tables in a row; p. 17 one column of references |

## Page-by-page findings (before)

- **p. 1** Abstract and keywords fine. Body text Latin Modern (font issue above).
- **p. 3** Fig. 1 (localization workflow) raster, labels at about 4 pt when printed. The "Prompt Example" box uses a grey background that the reference never uses.
- **p. 5** Fig. 2 raster and outdated; Table 1 redundant column; Listing 1 broken over two columns.
- **p. 6** Table 2 (oracle families) and Table 3 (controls) both full width and adjacent; Table 3 is the methodological core but reads like a dense list.
- **p. 7** Table 4 (study overview) duplicates the RQ list and parts of Table 3.
- **p. 9** RQ1 and RQ2 tables: header group "F1 at random" with sub-columns "Place/File" needs the caption to decode.
- **p. 10** RQ3 table: confidence intervals in `\scriptsize` brackets, inconsistent decimal style (".21" vs "0.332").
- **p. 11** Fig. 3 legend overlaps the plotting area at column width; DejaVu font.
- **p. 11–12** RQ4 table: "MRR / P@10" pairs in one cell are hard to scan.
- **p. 12** RQ5 table: CIs only in one column.
- **p. 16–17** References: last page half empty (acceptable).

## Planned changes (by priority)

1. **Fonts:** load Times explicitly for every engine (TeX Gyre Termes + newtxmath under XeTeX/LuaTeX, newtx under pdfLaTeX) so that Tectonic and Overleaf give the same result.
2. **Fig. 2 → new vector TikZ architecture figure:** inputs (task and data, prompts and models, oracles) → numbered pipeline (execute, normalize, match, measure) → outputs, with a band for built-in controls and a band for reproducibility records; same palette as the controls figure.
3. **New Fig. "four controls":** four panels (A chance placement, B oracle substitution, C candidate baselines and masking, D held-out selection), principle only, no results. It replaces nothing but makes Table 3 readable.
4. **Fig. 1:** redraw as a compact TikZ diagram or drop (the new architecture figure covers localization). Decision: drop the raster, keep the text.
5. **Listing 1 → table of oracle records** (one row per oracle family, merged with Table 2) to remove the broken listing.
6. **Tables:** remove redundant columns, consistent 3-decimal values, CIs as `[lo, hi]` in the same size, clearer headers (e.g. "Random lines (same file)"), RQ4 split into MRR and P@10 column groups.
7. **RQs:** a restrained `rq` environment (bold label, italic question, thin rules) used once in Section 2 and recalled at the head of each results subsection; one "Answer to RQn" paragraph per RQ in the reference's style, replacing the heterogeneous "Key findings" lists.
8. **Prompt box:** plain framed `verbatim`-style figure without grey fill.
9. **Fig. 3 (RQ4 plot):** regenerate with a serif font matching the text and the legend outside the axes.
10. **Captions:** 2–3 sentences, self-contained, consistent terms ("random placement", "metric oracle", "candidate list").

Constraints kept: no change to numbers, RQs, or claims; no change to the benchmark code.
