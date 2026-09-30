# OCR knowledge-tree study

`scripts/run_ocr_study.py` is the front end of the OCR knowledge-tree study. It runs
extraction engines over PDF corpus pages, builds knowledge trees from the transcripts,
compares runs within a family, and writes every artifact, run and report under one study
root. It touches no database, no queue and no object storage, and it never modifies the
corpus.

Nothing leaves the machine unless a flag says so: a paid engine needs `--allow-upload`,
a live detail merge needs `--allow-merge-upload`. Both are refused before any client is
constructed, and a study that never names the paid engine never reads a key.

Everything is declared in JSON: corpus manifests and gold references. Adding a document
or a reference is a data edit, not a code change.

## Running it

From the repository root, with the project environment:

```bash
.venv/bin/python scripts/run_ocr_study.py --help
```

Commands print JSON on stdout. Errors print `error: ...` on stderr and exit 1; argparse
usage errors exit 2. `--help` lists the engines, profiles and families this build knows.

## Concepts

* **Corpus manifest** — `corpus_version: 1`. One entry per document: PDF path, sha256,
  the 0-based physical pages in scope, declared modality, permission note, and the gold
  path as a path or an explicit `null`.
* **Engine** — turns a page selection into an extraction artifact.
* **Artifact** — content-addressed extraction result stored write-once at
  `<root>/<doc_id>/artifacts/<digest>.json`; the digest is the sha256 of the canonical
  artifact JSON.
* **Profile** — pairs an engine with a tree builder and a merge policy.
* **Family** — the unit of comparison: `pipeline`, `ocr` or `chunk`.
* **Run** — a directory under `<root>/<doc_id>/runs/<run_id>/`. Extraction run ids are
  `{doc_id}.{engine}.r{n}`; tree run ids are `{doc_id}.{profile}.r{n}.{llm_mode}`.
* **Gold** — one hand-written JSON file per document: the transcript a run should have
  produced, page by page. It is taken as human-checked; see
  [the gold file](#the-gold-file).

## Engines

| Engine | What it is | Requirements |
| --- | --- | --- |
| `vision` | Paid hosted model; renders every page and uploads one image request per page | `--allow-upload`; `VISION_API_BASE` (HTTPS), `VISION_API_KEY`, `VISION_MODEL`; the endpoint host must be listed in `VISION_ALLOWED_HOSTS`; some hosts additionally need a session header in `VISION_HEADERS` |
| `marker` | Real local Marker (`marker-pdf` distribution) | `pip install marker-pdf` **and pin `transformers==4.51.3`** (see [Engine prerequisites](#engine-prerequisites)); the first run downloads model weights and a render font |
| `docling` | Real local Docling (`docling` distribution), mapping v1 | `pip install docling` into an interpreter of its own, named in `OCR_RESEARCH_ENGINE_PYTHON`; Docling is not an application dependency (see [Engines in their own interpreter](#engines-in-their-own-interpreter)) |
| `native` | PDF text layer only (`pypdfium2`) | a scanned page is `unsupported_modality` |

Environment variable names only; values belong in the environment or `.env`, never in a
manifest. `--dpi` (default 200) controls vision rendering. `--offset` is the page number
the first physical page should have for the engine (`page_index + 1 + offset`); it is
recorded in the artifact and used to map tree page numbers back to physical indices.

## Profiles

| Profile | Engine | Builder | Merge |
| --- | --- | --- | --- |
| `vision_direct` | vision | vision_groups | off |
| `marker_llm` | marker | marker_structure | real |
| `vision_atomic_merge` | vision | atomic | real |
| `vision_atomic` | vision | atomic | off |
| `marker_atomic` | marker | atomic | off |
| `docling_atomic` | docling | atomic | off |
| `native_atomic` | native | atomic | off |

`run --profile all-compatible` builds every profile the artifact's own engine declares for
the chosen `--llm-mode`; under the default `none` that is the merge-off profiles.

Families: `pipeline` compares `vision_direct` with `marker_llm` on matched documents
and repeats. `ocr` compares transcripts and says nothing about the trees built
from them; `chunk` compares `vision_direct` with `vision_atomic_merge` on the same artifact.
`pipeline` and `chunk` are fixed pairs and pair nothing else: two arms that measure an
engine against itself are not the comparison those families name.

## Commands

Every command takes `--corpus PATH` and `--root PATH` (default
`data/research/ocr_pipeline`). `--doc DOC_ID` is repeatable and defaults to every document.
`--pages N-M,...` is a comma-separated list of 0-based physical pages, a single `N` or an
inclusive `N-M`, e.g. `0-9,20`; the default is the document's own selection.

### `corpus validate`

```
corpus validate --corpus CORPUS [--root ROOT] [--doc DOC_ID]
```

Hashes every selected PDF, checks that page indices are in range and strictly increasing,
re-derives the text layer over the selection and refuses a `modality` that contradicts it,
then probes all four engines. Probes are read-only: they never upload and never download
model weights. Document checks follow `--doc` like the other commands; the engine probe
list always covers all four engines. JSON: `corpus`, `documents[]`, `engines[]`.

### `render`

```
render --corpus CORPUS [--root ROOT] [--doc DOC_ID] [--pages N-M,...]
       --out OUT [--dpi DPI] [--offset OFFSET]
```

Renders pages to `<out>/<doc_id>/page-NNNN.png` using 0-based physical indices; existing
files are left alone. These images are what a human reads while authoring or correcting a
gold reference — do not author a reference from engine output.

### `gold check`

```
gold check --corpus CORPUS [--root ROOT] --doc DOC --file FILE
```

Reads one gold file against one document — the file is named on the command line, so a gold
can be checked before it is declared in the manifest — and reports how much of the page
selection it covers: `requested_pages`, `covered_pages`, `missing_pages`,
`out_of_scope_pages` and `complete`. A page left out of `pages` is missing, not empty; a
page outside the document's selection is reported rather than silently ignored. The file's
`doc_id` must match `--doc`, so the wrong gold cannot be checked silently. JSON: `doc_id`,
`gold_path`, `gold_digest`, the four page lists and `complete`.

### `extract`

```
extract --corpus CORPUS [--root ROOT] [--doc DOC_ID] [--pages N-M,...]
        --engine {vision,marker,docling,native}
        [--repeats REPEATS] [--dpi DPI] [--offset OFFSET]
        [--allow-upload] [--resume] [--label LABEL] [--cold | --warm]
```

Runs one engine `--repeats` times (default 1), storing one run and one artifact per repeat.
`vision` without `--allow-upload` is refused before a client is built. An engine that is
not installed is recorded `not_available`; a failure mid-run is recorded `failed`; neither
hides the other documents. A vision run aborts that document's run on the first failed
page, so no partial artifact is stored. `--resume` reuses an existing run only when the
identical request key already succeeded; otherwise a fresh attempt directory
(`<run_id>-a2`, `-a3`, ...) is allocated so the earlier attempt survives. `--cold`,
`--warm` and `--label` are recorded provenance, not control flow. Exit status 0 when every
run this invocation stored is `ok`, otherwise 1.

### `runs`

```
runs --corpus CORPUS [--root ROOT] [--doc DOC_ID] [--kind {extract,tree}]
```

Lists the run manifests on disk with `run_id`, `kind`, `doc_id`, `engine_id`, `profile`,
`status`, `rankable`, `artifact_digest`, `created_at` and `path`.

### `run`

```
run --corpus CORPUS [--root ROOT] [--doc DOC_ID] --profile NAME
    [--extract-run DOC_ID:RUN_ID] [--artifact DIGEST]
    [--llm-mode {none,live}] [--allow-merge-upload]
    [--merge-concurrency N] [--resume] [--label LABEL]
```

Builds trees from artifacts already on disk; no engine is called, so a replay is free.
The source is one `--extract-run` per document (repeatable) or one `--artifact` digest for
exactly one document. `--profile all-compatible` expands to the engine's merge-off
profiles.

`--llm-mode none` builds a merge profile without its merge and records
`quality: not_measured`, `quality_reason: merge_not_run`, `rankable: false`.
`--llm-mode live` calls the configured LLM gateway and requires `--allow-merge-upload`;
there is no offline merge mode, because a merge that reached no model would be recorded
under a name it did not earn. Research live merges use four concurrent section calls by
default; `--merge-concurrency N` changes that limit (minimum 1) and changes the cache
request identity. Independent child sections can overlap, but a parent waits for its
children; only the LLM invocation holds a concurrency slot. Exit status as in `extract`.

Resuming a run that asked for a merge adds one demand to the identical request key: the
reused record's counters must show at least one section whose details the merge rebuilt
(`sections_merged > sections_fallback`). A merge that reached no model keeps every
original, so its record holds the tree the merge ran over under the merge's name — the
same tree the merge-off profile builds — and a resume walks past it to a later attempt
that really merged, or allocates a fresh one. The stored `quality` is not what decides:
it was computed by the code of the day, while the counters are the fact. The demand
follows the request's resolved mode, so a merge profile asked for no merge
(`--llm-mode none`) is still answered by its zero counters.

### `compare`

```
compare --corpus CORPUS [--root ROOT] --family {pipeline,ocr,chunk}
        --run DOC_ID:RUN_ID [--run ...] [--gold DOC_ID=PATH]
        [--out OUT] [--label LABEL] [--dry-run]
```

At least one `--run`. `--gold doc=path` supplies or overrides gold for a document that
declares none. Relative paths resolve against the corpus manifest directory. The command
prints the report and writes `report.json`, `report.csv`, `report.md` and `manifest.json`
into `--out` or `<root>/comparisons/<family>.<UTC stamp>`. `--dry-run` prints the report
and writes nothing. A run with no artifact keeps its row as not measured; a document with
no gold is listed in the report's `limitations` and is not scored; a pair gets a delta only
when both runs are `ok`, `rankable` and on the same document (and, for `chunk`, the same
artifact digest). Exit status 0 once the report is produced, even if some rows are
unrankable.

### `study`

```
study --corpus CORPUS [--root ROOT] [--doc DOC_ID] [--pages N-M,...]
      --families FAMILIES --engines ENGINES [--profile NAME]
      [--repeats REPEATS] [--gold DOC_ID=PATH]
      [--llm-mode {none,live}] [--allow-upload]
      [--allow-merge-upload] [--merge-concurrency N]
      [--label LABEL] [--dry-run] [--resume | --fresh]
```

Plans the work, validates the corpus, then extracts and compares each requested family
for every document and repeat. A study with only `--families ocr` builds **no trees** and
runs **no merge**; `--profile` is rejected because there are no trees to select. Studies
including `pipeline` or `chunk` also build trees. Engine order alternates between repeats
so a warm engine cannot masquerade as an engine difference. `--families` and `--engines`
are comma-separated; for tree studies, `--profile` is repeatable and narrows the trees
built. With `--llm-mode none` (the default) only merge-off profiles are built; a merge
profile needs `live` plus `--allow-merge-upload`. `--dry-run` prints the plan, including
planned limitations such as a refused paid engine or a missing engine package, and runs
nothing; it needs no permission flag.

The JSON summary reports the plan, corpus checks, extraction rows, tree rows, comparison
counts, the expected `grid` cell by cell, the corpus-wide `summary` roll-up, and `skipped`
families with reasons. Individual run failures are visible there; the command exits 1 when
any extraction or tree failed or was unavailable. Configuration and corpus errors also
exit 1.

### Exit status

| Code | Meaning |
| --- | --- |
| 0 | Completed; for `extract`, `run`, and `study`, every run they recorded is `ok` (a dry run records none) |
| 1 | Refused or failed: `error: ...` on stderr, or an extraction/tree run recorded by `extract`, `run`, or `study` was not `ok` |
| 2 | argparse usage error (missing `--corpus`, unknown engine choice) |

`compare` exits 0 when it produces a report, even if some rows are unrankable:
that missingness is data in the report. `study` exits nonzero for recorded failed
or unavailable runs; inspect its summary, `runs`, and reports for the reason.

## Permissions

* `--allow-upload` is required by `vision`: rendered page images leave the machine and the
  endpoint may charge. Refusal happens before any client or key is read.
* `--allow-merge-upload` is required by `--llm-mode live`: merge prompts leave the machine.
  Without it a run that would merge is refused before any client is built. A run that only
  replays a record already on disk never merges, so it never needs the permission.
* `corpus validate`, `render`, `gold check`, `runs` and `compare` never post anything.
* `permission_note` is mandatory on every document and is copied into run manifests. It
  records the human permission decision; the tool does not enforce it.
* Reports never carry keys, headers, full URLs or raw transcript text: `public_config` is
  an allowlist (DPI, model name, endpoint host, request shape) and the report builder
  rejects text-like keys and long strings.
* Manifest validation forbids unknown keys, so a credential or endpoint cannot be smuggled
  into a tracked corpus file.

## What the numbers mean

* **Run status** — `ok`, `failed` or `not_available`. Failure entries carry an error class
  and code, never a provider body.
* **Page status** — `ok` or `unsupported_modality` (for example a `native` page with no
  text layer). A page that could not be read stays in the denominator with its status.
* **Metric status** — `measured`, `not_measured` or `empty_reference`, always with a
  reason (`hypothesis_missing`, `empty_reference`, `empty_denominator`, `no_measured_pages`,
  `no_measured_documents`, `no_page_totals`, `not_provided`). A missing hypothesis is
  not 100% error; a produced empty-string hypothesis is a full deletion and does score as
  0. Character error rate is computed over NFC-normalized tokens; `cer` and `wer` are
  reported per run under `text_metrics.runs.<run_id>.rates`, micro and macro.
* **Structure status** — heading and boundary placement is scored per gold page under
  `structure_metrics`. Two gates stand in front of a page: only a gold page that lists
  `headings` is asked, and only a page whose text lines up with the gold at or above
  `confidence` (0.90, Dice over tokens) is scored. A page is skipped by its own name —
  `hypothesis_missing`, the page's own status such as `unsupported_modality`,
  `empty_candidate_text`, `self_scoring` (the candidate is the gold, so a copy would score
  perfectly and say nothing) or `alignment_below_threshold` — and the reasons are counted
  in `skipped`. Marks are compared as single character offsets, never as spans.
* **Tokens** — a run's own counters: `attempts`, `prompt_tokens`, `completion_tokens`.
  A local engine reports none, which is a typed miss (`no_documents`, `usage_missing`),
  never a zero. A pair's two sides are compared under `paired_usage.pairs[].usage`, with a
  `delta` only when both sides reported: one side missing makes the delta
  `not_measured`/`incomplete_pairing`. Tokens are the only thing a pair compares; nothing
  here converts them to money.
* **Rankable / quality** — an unmerged merge profile (`merge_not_run`) and a failed build
  (`build_failed`) are stored but not rankable. Paired deltas exist only when both sides
  are compatible.
* **Denominators** count what was asked for, not what worked. Every report also carries a
  `limitations` list naming what could not be measured and why.

## Safe synthetic example

The two files below are invented, contain no third-party text and no credentials, and are
safe to commit or share. They only need a real two-page digital PDF you own (or generate
yourself) at `sample.pdf`, plus the real `sha256sum sample.pdf`; the all-zero checksum is a
deliberate placeholder and `corpus validate` refuses it until replaced. Relative paths in
the manifest resolve against the manifest's own directory.

`sample.corpus.json`

```json
{
  "corpus_version": 1,
  "documents": [
    {
      "doc_id": "sample",
      "pdf_path": "sample.pdf",
      "pdf_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "page_indices": [0, 1],
      "modality": "digital",
      "permission_note": "invented pages created for this example; no third-party content",
      "gold_path": "sample.gold.json"
    }
  ]
}
```

`sample.gold.json` — page 1 is deliberately left out, so it is missing rather than blank:

```json
{
  "doc_id": "sample",
  "pages": [
    {
      "page_index": 0,
      "text": "Sample heading\nBody line one.\nBody line two.",
      "headings": ["Sample heading"]
    }
  ]
}
```

Run steps (offline engines only; nothing is uploaded):

```bash
# 1. Validate the corpus (all four engines are probed).
.venv/bin/python scripts/run_ocr_study.py corpus validate --corpus sample.corpus.json

# 2. Check the gold against the document's page selection.
.venv/bin/python scripts/run_ocr_study.py gold check --corpus sample.corpus.json --doc sample

# 3. Extract with a local engine, twice.
.venv/bin/python scripts/run_ocr_study.py extract --corpus sample.corpus.json \
    --engine native --repeats 2

# 4. List what exists and compare the two transcripts.
.venv/bin/python scripts/run_ocr_study.py runs --corpus sample.corpus.json
.venv/bin/python scripts/run_ocr_study.py compare --corpus sample.corpus.json \
    --family ocr --run sample:sample.native.r1 --run sample:sample.native.r2

# 5. Plan a paid study without spending anything or granting upload.
.venv/bin/python scripts/run_ocr_study.py study --corpus sample.corpus.json \
    --families pipeline,ocr --engines vision,native --dry-run
```

What to look at: `corpus validate` reports each engine's availability; `gold check` reports
`missing_pages: [1]` and `complete: false`; the `ocr` report scores page 0 against the gold
and leaves page 1 without a reference; the native runs' `usage` is `not_measured` with
reason `no_documents`, because a local engine reports no token usage, so a pair with a
vision run reports an `incomplete_pairing` delta rather than a difference; `runs` shows
both repeat directories; the study `--dry-run` plan names the paid engine and the
permission it would need.

## Real corpus drop-in

The repository ships no corpus and no gold. Bring your own PDFs, a manifest and, for
scoring, gold transcripts.

### Files and fields

**Manifest** (`corpus_version: 1`) — per document:

| Field | Required | Notes |
| --- | --- | --- |
| `doc_id` | yes | Slug; used in run ids and directories |
| `pdf_path` | yes | Resolved against the manifest's directory; PDFs stay out of git |
| `pdf_sha256` | yes | 64 lowercase hex; `sha256sum path/to/file.pdf` |
| `page_indices` | yes | 0-based physical pages, strictly increasing, at least one |
| `modality` | yes | `scan`, `digital` or `mixed`, checked against the real text layer over the selection |
| `permission_note` | yes | Non-blank; why this document may be used |
| `gold_path` | yes | Path or explicit `null`; must be declared so "no gold yet" is not implied |

### The gold file

One JSON document per document, written by hand and taken as checked:

```json
{
  "doc_id": "sample",
  "pages": [
    {"page_index": 0, "text": "…", "headings": ["BÀI 1", "I. Điện trường"]}
  ]
}
```

`doc_id` names the document the transcripts were read from, so a gold file cannot be
attached to another document by accident. `text` is the page's transcript. `headings` is
optional: the heading strings on that page, in reading order, each of which must occur in
that page's own `text` — a heading that cannot be found is an error at load time rather
than a scored miss, because gold that disagrees with itself would measure the search
instead of the run. A page with nothing worth transcribing is left out of `pages`
altogether; `gold check` names it as missing. Unknown keys are refused (`extra="forbid"`).

There is no separate annotations file: the headings in the gold are what the boundary
metrics are scored against, and the alignment locates them.

### Preparing a document

1. Put the PDF outside version control or under `data/research/` (both are gitignored),
   compute `sha256sum`, choose the page selection, and declare the modality honestly.
2. Run `corpus validate`; fix any checksum, range or modality error before extracting.
3. For gold: `render` the pages, read them yourself, and write the transcripts by hand.
   Confirm with `gold check` that every page in scope is covered.

### Engine prerequisites

| Engine | Install / configure |
| --- | --- |
| `vision` | Set `VISION_API_BASE`, `VISION_API_KEY`, `VISION_MODEL`, and list the host in `VISION_ALLOWED_HOSTS`; add a session header to `VISION_HEADERS` if the host requires one (for example `x-opencode-session`) |
| `marker` | `pip install marker-pdf "transformers==4.51.3"`; the first run downloads weights and a font |
| `docling` | `pip install docling` into a separate interpreter (its Torch and Transformers pins cannot share one with the application), then `OCR_RESEARCH_ENGINE_PYTHON=docling=/path/to/that/python` |
| `native` | `pypdfium2` (already a project dependency) |

The `transformers` pin under `marker` is required, not cosmetic. `marker-pdf 1.6.1` declares
`transformers>=4.45.2,<5.0.0` and `surya-ocr 0.13.1` declares `transformers>=4.41.0,<5.0.0`,
but surya's recognition config is a composition config (`is_composition = True`) that reads
`encoder` and `decoder` from its `__init__` kwargs and never sets
`has_no_defaults_at_init = True`. Two things then break on newer transformers, both while the
recognition model loads:

* from `4.52`, `to_diff_dict` instantiates the config class bare unless that flag is set, so
  `SuryaOCRConfig.from_pretrained` dies inside `logger.info(f"Model config {config}")` with
  `KeyError: 'encoder'`;
* from `4.54`, `_init_weights` also calls `config.get_text_config()`, which raises
  `ValueError: Multiple valid text configs were found ... ['text_encoder', 'decoder']`.

`4.51.3` is the highest release with neither problem, and is the version a real Marker run
through this pipeline was verified against. Both declared ranges are too permissive to
express the constraint, and the constraint belongs to the environment rather than to the
adapter, so no code enforces it: an environment that upgrades `transformers` past 4.51 fails
its first Marker run.

#### Engines in their own interpreter

An engine can be run by an interpreter other than the study's own:

```bash
OCR_RESEARCH_ENGINE_PYTHON=docling=/srv/venvs/docling/bin/python \
.venv/bin/python scripts/run_ocr_study.py extract --corpus corp/corpus.json --engine docling
```

The value is comma-separated `engine=path` pairs, so two engines can be routed at once:

```bash
OCR_RESEARCH_ENGINE_PYTHON=docling=/srv/venvs/docling/bin/python,marker=/srv/venvs/marker/bin/python
```

An engine with no pair runs in the study's own interpreter, which is what `vision`, `native`
and a Marker you installed into the application venv do. A path that is not a file raises
before any work starts, rather than becoming a worker that cannot start.

Docling is not an application dependency, and it is routed away from the application venv
because the two cannot share one interpreter. The environment used for this repository's
Docling runs holds `transformers 5.17.0` and `torch 2.14.0`; the application venv must hold
`transformers 4.51.3` for Marker and carries `torch 2.9.1+cu128`. Installing Docling into the
application venv would give one of the two the wrong Transformers, so the engine keeps its
own. Nothing in the application imports Docling, and the study reaches it only through this
variable.

What the child sees: the engine's interpreter answers first for everything it has, and this
interpreter's import path is appended after it. The engine therefore runs on its own pinned
packages while `research.ocr_pipeline.*` and `workflows.*` stay importable, so the worker
imports the code that started it. The child is a fresh process either way, so routing an
engine changes which packages answer, not how the run is measured.

#### What a run costs in memory

A local engine holds its whole model set resident for the length of a run, so the run needs
the room before it starts, and a run that does not get it is killed rather than slowed. The
recorded `rss_peak` is a sampled estimate of the worker process tree (`estimate: true`),
not a continuous measurement or an additive peak across independent stages.

A report attributes a reading to the run that measured it. A tree replay ran no engine, so
the extraction it names contributes its stages and its resources to the report under that
run's own id, and the tree's row carries them as its arm's while naming the run they came
from (`resource_run_id`). The linkage is that run's id, never a digest two runs may share.
A hosted engine's backend memory is not observable from here: that arm's reading is
`not_available`/`hosted_backend`, and a pair with one on either side reports a typed miss
rather than a difference. No peak is ever summed across stages.

| Engine | Measured peak |
| --- | --- |
| `marker` | Approximately 5.4 GB while loading its predictors; one-page run: 7.03 GB (decimal) sampled peak |
| `docling` | Ten-page run: 3.05 GB (decimal), approximately 2.84 GiB; earlier 2.91 GB figure was from a different run/context |

These are historical observations, not memory requirements measured for every document.
Marker's model set stays resident through a run. On this machine a one-page run completed;
a separate two-page direct run was kernel-killed near 7.43 GB. That OOM observation came
from the direct-run/kernel log, **not** a stored two-page run manifest. `marker` therefore needs roughly 8 GB of headroom to be safe, and a study that asks
for more pages than the machine can hold fails with `engine_failed` rather than degrading.
When a failed worker produced resource samples, its run retains the sampled
`rss_peak`; that estimate alone cannot diagnose OOM without the exit status or kernel log. `create_model_dict`
takes no argument for loading a subset of the models, so the floor is not tunable — scope the
run with `--pages` instead.

**A study on a machine without that headroom leaves `marker` out of `--engines`.** The
`pipeline` family's Marker arm is then a named gap in the report rather than a substitute:
this build has no frozen-dump engine to stand in for a live run, because a replay's numbers
are not measurements of Marker.

### Running a real study

```bash
.venv/bin/python scripts/run_ocr_study.py corpus validate --corpus corp/corpus.json

# Local engines only.
.venv/bin/python scripts/run_ocr_study.py study --corpus corp/corpus.json \
    --families pipeline,ocr,chunk --engines native,marker,docling --repeats 2

# Add vision only if you accept paid uploads.
.venv/bin/python scripts/run_ocr_study.py study --corpus corp/corpus.json \
    --families pipeline,ocr,chunk --engines vision,native --repeats 2 --allow-upload
```

No code changes are needed for a new corpus or a new gold. A new engine, profile or family
is code.

### Data hygiene

The default root `data/research/ocr_pipeline` is gitignored, as are `*.pdf` and `.env`.
Keep real book text, PDFs and gold transcripts out of git unless you hold the rights; keep
credentials in the environment or `.env`, never in a manifest. A manifest cannot carry an
endpoint or key anyway (`extra="forbid"`), and `public_config` in every run keeps only an
allowlisted shape.

## What executes, and what its numbers mean

| Path | What executes | What its numbers mean |
| --- | --- | --- |
| `vision` | Paid hosted model, one request per rendered page | Real transcript and reported usage. Uploads leave the machine |
| `marker` | Real local Marker over the PDF | Real transcript; a local engine reports no tokens, so its usage is a typed miss |
| `docling` | Real local Docling, mapping v1 | Real transcript; usage not reported |
| `native` | PDF text layer | Real transcript where a text layer exists; scan pages are `unsupported_modality` |
| `--llm-mode live` | Gateway merge over a real tree | Real merge; rankable when the build succeeds |
| `--llm-mode none` | Skips the merge for a merge profile | `merge_not_run`, `not_measured`, unrankable |

The unit tests inject in-memory engine doubles; the CLI itself never selects a fake engine,
and no mode replays a merge.

## Where things are written

```text
<root>/<doc_id>/artifacts/<digest>.json          content-addressed extraction
<root>/<doc_id>/runs/<run_id>/manifest.json      request, outcome, digests
<root>/<doc_id>/runs/<run_id>/tree.json          the built tree (tree runs)
<root>/<doc_id>/runs/<run_id>/metrics.json       fidelity and chunk metrics
<root>/<doc_id>/runs/<run_id>/signature.json     semantic and page signatures
<root>/comparisons/<family>.<stamp>/report.json  one report dict, three renderings
<root>/comparisons/<family>.<stamp>/report.csv
<root>/comparisons/<family>.<stamp>/report.md
<root>/comparisons/<family>.<stamp>/manifest.json
```

The default root is `data/research/ocr_pipeline`; point `--root` anywhere writable.
Artifacts are write-once per content digest; runs are never overwritten (a repeat gets an
attempt suffix). Reports carry counts, statuses and digests; page text stays in the
artifacts. The three report formats are rendered from the same dict, so their counts agree
by construction.

## Limitations

* Heading/boundary scores abstain when the gold lists no headings for a page, when the run
  did not read the page, or when the alignment is not confident enough; the reason is
  counted per page, never folded into a score.
* A document with no gold, or a gold that does not cover a page, is not scored: it is
  `not_measured` and named in `limitations`.
* The first `marker` run downloads model weights and a font: expect a large network
  download, and plan for it on an offline machine.
* A vision run inside a study uses the study's fixed retry policy (2 retries, 300 s
  timeout) rather than per-study options.
* A merge that reached no model is unrankable; the report says `merge_rebuilt_nothing`
  rather than reporting the merge-off tree under the merge's name.
* The `docling` mapping is v1 and only as faithful as its heading model.
* The reconstructed Marker heading parser intentionally accepts lowercase Roman and
  uppercase lettered headings via `re.IGNORECASE`; the historical parser did not.
  `tests/test_builders.py` records this divergence instead of claiming absolute parity.
