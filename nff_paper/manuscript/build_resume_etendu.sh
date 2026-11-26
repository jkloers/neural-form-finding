#!/bin/bash
# Compile the standalone French resume etendu (Guide du PFE 2.2.5) in a scratch copy.
#
# Why a scratch copy: paper/ is written by more than one agent at a time and is
# two-way Overleaf-synced. A build launched in paper/ races those writers. Same
# discipline as nff_paper/manuscript/build_paper_isolated.sh.
#
# Why it builds the memoire first: resume_etendu.tex uses xr + \externaldocument
# to resolve every \ref into the MEMOIRE's numbering. Without a current
# sn-article.aux beside it, every figure, table, section and hypothesis number
# prints ?? and the document is silently wrong rather than broken.
#
# Result is copied back to paper/resume_etendu.pdf, which nothing else writes.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$REPO/paper"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/rebuild.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

export PATH="/Library/TeX/texbin:$PATH"

# A zero-filled .bbl is an Overleaf sync artefact, not a document fault, and it
# would silently empty the reference list. Refuse rather than ship that.
if [ ! -s "$SRC/sn-article.bbl" ] || [ "$(wc -c < "$SRC/sn-article.bbl")" -lt 1000 ]; then
  echo "REFUSING: paper/sn-article.bbl is missing or suspiciously small."
  echo "  Overleaf has zero-filled it before. Re-run bibtex on the memoire first."
  exit 1
fi

cp "$SRC"/*.tex "$SRC"/*.bst "$SRC"/*.cls "$SRC"/*.bib "$WORK"/ 2>/dev/null || true
cp "$SRC"/sn-article.bbl "$WORK"/ 2>/dev/null || true
cp "$SRC"/*.eps "$WORK"/ 2>/dev/null || true
cp -R "$SRC"/Images "$WORK"/ 2>/dev/null || true

cd "$WORK"

# 1. the memoire, purely to produce the .aux that xr reads.
for pass in 1 2; do
  pdflatex -interaction=nonstopmode -file-line-error sn-article.tex > "mem$pass.log" 2>&1 || true
done
if [ ! -f sn-article.aux ]; then
  echo "BUILD FAILED -- the memoire produced no .aux, so xr has nothing to resolve against."
  exit 1
fi

# 2. the standalone: pdflatex, bibtex for its own reference list, then two more
#    passes to settle the citations and this document's own page breaks.
pdflatex -interaction=nonstopmode -file-line-error resume_etendu.tex > res1.log 2>&1 || true
bibtex resume_etendu > res_bib.log 2>&1 || true
for pass in 2 3; do
  pdflatex -interaction=nonstopmode -file-line-error resume_etendu.tex > "res$pass.log" 2>&1 || true
done
cp res3.log res2.log

if grep -qE '^!' res2.log; then
  echo "BUILD FAILED -- real errors in the document:"
  grep -E '^!' -A4 res2.log | head -40
  exit 1
fi

cp resume_etendu.pdf "$SRC/resume_etendu.pdf"
echo "OK  $(grep -o '([0-9]* pages' res2.log | tail -1 | tr -d '(')  ->  paper/resume_etendu.pdf"
echo "undefined refs: $(grep -c 'Reference.*undefined' res2.log)   undefined cites: $(grep -c 'Citation.*undefined' res2.log)"
echo "Overfull: $(grep -c 'Overfull' res2.log)"
