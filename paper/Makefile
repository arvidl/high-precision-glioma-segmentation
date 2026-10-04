# paper/Makefile
# Build the author manuscript PDF from main.tex.
# Requires: latexmk + a TeX distribution (TeX Live or MacTeX).

LATEXMK ?= latexmk
ENGINE  ?= -pdf
JOBS    ?=
MAIN    := main
LATEXMK_JOBS := $(if $(strip $(JOBS)),-j$(JOBS),)

.PHONY: all paper clean veryclean watch

all: paper

paper: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib abbrv.bst ORCID.png $(wildcard figs/*)
	$(LATEXMK) $(ENGINE) -interaction=nonstopmode -halt-on-error -file-line-error \
	    -bibtex $(LATEXMK_JOBS) $(MAIN).tex

watch:
	$(LATEXMK) $(ENGINE) -pvc -interaction=nonstopmode $(MAIN).tex

clean:
	$(LATEXMK) -c $(MAIN).tex

veryclean:
	$(LATEXMK) -C $(MAIN).tex
	$(RM) *.bbl *.run.xml
