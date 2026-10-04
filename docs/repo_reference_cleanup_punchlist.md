# Repo-reference cleanup punch-list (`paper/main.tex`)

**Status:** post-mortem of the cleanup pass landed in commit `db90fad`
(*paper(main): reduce repo-internal file references to a minimum*).
Kept here as a checklist for future passes — same rules apply if a new
revision round adds back path-like `\texttt{...}` clutter.

The aim is a manuscript that reads as a research paper rather than as
a tour of the producer codebase. Repo-internal artefacts (scripts,
source-tree paths, internal artefact paths, Make targets, notebook
filenames, internal tracker pointers, release-versioning chrome) do
not belong in the body text; they belong in the **Code and data
availability** section, which carries the public-repo URL once.

---

## Rule of thumb

A path-like `\texttt{...}` is allowed in the manuscript body if and
only if it is one of the following:

1. A **third-party data-product file name** that the reader needs in
   order to cross-reference an external archive
   (e.g. `LUMIERE-MRinfo.csv` next to its figshare URL).
2. A **cohort-pinning manifest** cited once with a `cf.\ Code and
   data availability` pointer (e.g.
   `configs/cohort_ucsfpdgm_n50.yaml`).
3. A **dependency manifest** named in the *Code and data availability*
   section itself (e.g. `pyproject.toml`, `environment.yml`).

Everything else — producer scripts, source-tree paths, internal
artefact paths under `outputs/` or `data/derivatives_*`, Make targets,
notebook filenames, internal tracker pointers (`paper/revision_tracker.md`,
`PR-CLEANUP` markers), and release-versioning chrome (e.g.
`v2.0-revised`, `Zenodo DOI to be minted`) — should be **removed** or
**softened to prose** ("the producer chain in the public repository",
"the per-subject Hit-Plot bar renderer in the public repository",
"the outlier-audit notebook in the public repository").

---

## What was done in `db90fad`

Reduced 65 path-like `\texttt{...}` references to **7 deliberate
residuals** (line-count basis), all matching the rule above.

### Kept (7 lines)

| Where | What | Why |
| --- | --- | --- |
| Methods, cohort sentence | `configs/cohort_ucsfpdgm_n50.yaml` | cohort-pinning manifest, with `cf.\ Code and data availability` pointer |
| Methods, LUMIERE figure | `LUMIERE-MRinfo.csv` | third-party dataset file, cross-referenced with figshare URL |
| Code and data availability | `pyproject.toml`, `environment.yml`, `configs/cohort_ucsfpdgm_n50.yaml`, public-repo URL, REK contact email | consolidated into one paragraph; section title renamed `Data availability` → `Code and data availability` (in `\rev{}` blue) |

### Removed or softened (~58 lines)

* **Producer scripts** (`scripts/build_*`, `scripts/bench_*`,
  `scripts/render_*`, `scripts/compute_*`) → "the producer chain in
  the public repository" or specific functional names ("the
  per-subject Hit-Plot bar renderer in the public repository").
  All ~22 references gone or replaced.
* **Source-tree paths** (`src/hpgs/segment/unified.py`,
  `src/hpgs/parcellate/lobes.py`, `src/hpgs/viz/qc.py`,
  `src/hpgs/metrics/`, `src/hpgs/hitplot/`,
  `src/hpgs/segment/baselines.py`) → "in the public repository" or
  removed.
* **Internal artefact paths** under `outputs/`,
  `data/derivatives_cohort50/`, `metrics/`, `seg_dl/`, `parcellation/`,
  `hitplot/` → all ~20 references removed; aggregators are described
  in prose.
* **Make targets** (`make all`, `make bench-runtime`,
  `make lobe-heatmap`) → all 4 dropped.
* **Notebooks** (`notebooks/qc_segmentation_outliers.ipynb`,
  `notebooks/06-BGO-12-analyze-segmentations.ipynb`) → "the
  outlier-audit notebook in the public repository", "the BGO
  analysis notebook in the public repository".
* **Internal tracker pointers** (`paper/revision_tracker.md`,
  `PR-CLEANUP` markers in BGO figure provenance) → removed.
* **Versioning chrome** (`v2.0-revised` tag,
  `Zenodo DOI to be minted` parentheticals in the Contributions
  paragraph and the Runtime narrative) → removed; the *Code and data
  availability* section carries the public-repo URL once.

### Mechanics

* All edits land inside existing `\rev{...}` blocks, so the diff
  renders blue in `main.pdf`. Only one new `\rev{}` block
  was added (the augmented *Code and data availability* paragraph).
* `latexmk`: clean rebuild, 45 pages, 0 errors, 0 undefined
  references.
* Net diff: 105 insertions, 140 deletions; −35 net lines.

---

## Tooling

`scripts/list_rev_blocks.py` enumerates every `\rev{...}` block in
`paper/main.tex` (markdown table or fixed-width). Useful as
a generic inventory tool for any future cleanup pass:

```bash
uv run python scripts/list_rev_blocks.py --markdown
```

The `has_revision_marker` heuristic flags blocks whose body still
leaks revision-context language (*"the present revision"*, *"in this
revision"*, *"of the original submission"*, *"the new \emph{...}"*,
`v2.0-revised`, `Zenodo DOI to be minted`, `PR-CLEANUP`,
`cf.\ PR-`). Cheap to compute, useful for spotting leftover round-1
phrasings during any de-revision pass before resubmission. The
checker is informational only; exit code is always 0.

For path-like `\texttt{...}` references specifically, a quick audit
is `rg '\\texttt\{[a-z][^}]*[/.][^}]*\}' paper/main.tex`
(grep for `texttt` bodies that contain a slash or a dot, i.e. paths
or filenames). Hand-classify each match against the rule above.
