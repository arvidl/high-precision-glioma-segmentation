# `paper/` — manuscript

This subfolder is the manuscript half of the integrated paper-and-code repo. The
pipeline that produces every figure and table lives under `../src/hpgs` and
`../notebooks`; figure outputs are synced into `figs/` from `../outputs/figures/`.
The published journal PDF is
[`../article/Lundervold_etal_High_Precision_Segmentation_of_Glioma_JMET_2026.pdf`](../article/Lundervold_etal_High_Precision_Segmentation_of_Glioma_JMET_2026.pdf).

## Layout

```
paper/
├── main.tex          # author manuscript (no revision markup)
├── references.bib    # the 107 cited references
├── abbrv.bst
├── ORCID.png
├── refs/             # additional BibTeX styles
├── figs/             # figures and table fragments used by main.tex
├── Makefile
├── latexmkrc
└── README.md
```

## Build

The typeset PDF is **not** tracked in git (see [What to commit](#what-to-commit-what-to-keep-out) below). After pulling, rebuild locally:

```bash
cd paper && make paper
```

That produces `main.pdf` from `main.tex` and `references.bib`. The result is a 49-page A4 author manuscript. The journal PDF in `../article/` is the version of record and uses the publisher's layout.

> The full reference for the project's `make` targets — top-level orchestrator
> and `paper/Makefile` — is in [`../docs/build.md`](../docs/build.md). The
> sub-sections below cover the LaTeX-specific workflow only.

### Preferred local workflow — TeXShop (macOS)

The maintainer (Arvid) compiles the manuscript locally with **TeXShop** (MacTeX
distribution). Open `main.tex` in TeXShop and use `Typeset` (⌘T) — the
default engine is `pdflatex`; switch to `LaTeX → BibTeX → LaTeX → LaTeX` (or
press `Tools → Show TeXShop Console` and run BibTeX manually) to refresh the
bibliography on first build and after adding new `\cite{}` keys.

Recommended TeXShop preferences for this project:

- **Typesetting → Default Command:** `pdflatex` (LaTeX)
- **Typesetting → BibTeX engine:** `bibtex` (we use `.bst` files in `refs/`, not
  biber)
- **Source → Editor → Soft wrap:** ON (long sentences are common in the abstract
  and discussion)
- **Preview → Magnification:** Fit to width
- **Source → SyncTeX:** ON (⌘-click in the PDF jumps to source)

The bibliography lookup paths are also configured for `latexmk` via
`latexmkrc`, but TeXShop honours its own `TEXINPUTS`/`BIBINPUTS`/`BSTINPUTS`
environment. If TeXShop cannot find a `.bib` or `.bst`, add to
`~/Library/TeXShop/bin/path` or set them globally in `~/.zshrc`:

```bash
export TEXINPUTS=".:./figs:"
export BIBINPUTS=".:./refs:"
export BSTINPUTS=".:./refs:"
```

### Alternative — `latexmk` from the shell (used by CI)

```bash
make paper        # → main.pdf
make all          # manuscript PDF
make watch        # continuous rebuild on save
```

Requires `latexmk` and a TeX distribution (TeX Live or MacTeX). On macOS:
`brew install --cask mactex-no-gui` (≈4 GB) or `brew install basictex` for a
smaller install — then `tlmgr install latexmk biber tcolorbox enumitem` etc.

The CI `build-paper` job (`.github/workflows/ci.yml`) uses `latexmk` so the PDF
is rebuilt on every PR regardless of which editor was used to author the source.

## Author manuscript and journal PDF

`main.tex` is the corrected camera-ready author source: ordinary black text, the figures in `figs/`, and `references.bib` (107 cited works). `make paper` compiles it to `main.pdf`.

[`../article/Lundervold_etal_High_Precision_Segmentation_of_Glioma_JMET_2026.pdf`](../article/Lundervold_etal_High_Precision_Segmentation_of_Glioma_JMET_2026.pdf) is the published journal article. Its page design is the publisher's, so it will not match `main.pdf` page for page.

## Figure sync

Code-generated figures live under `../outputs/figures/` (git-ignored). Run

```bash
python ../scripts/sync_figs_to_paper.py
```

to mirror them into `paper/figs/` and `git add` only the ones the manuscript
references. The sync script preserves filenames so `\includegraphics{figs/foo}`
keeps working.

## What to commit, what to keep out

- **Commit:** `main.tex`, `references.bib`, `.bst`, `Makefile`, `latexmkrc`, and the figures in `figs/` that `main.tex` includes.
- **Don't commit:** build artefacts (`*.aux`, `*.log`, `*.out`, `*.bbl`, `*.blg`, `*.synctex.gz`) or `main.pdf`. Rebuild with `cd paper && make paper`.

The repo `.gitignore` already covers these patterns.
