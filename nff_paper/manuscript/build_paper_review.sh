#!/bin/bash
# Freeze paper/sn-article.tex and its build products into nff_paper/review/build/.
#
# Why a snapshot rather than the live tree: paper/ is Overleaf-synced two-way and
# clobbers partially. A reviewer reading the live .tex gets line numbers that drift
# under it mid-run, and a sync landing overnight overwrites the bytes it is quoting.
# Every review agent reads nff_paper/review/build/sn-article.snapshot.tex instead, so line
# numbers hold for the whole night and no agent is ever pointed at a writable source.
#
# Unlike build_paper_isolated.sh this runs the full bibtex cycle, so the .blg and the
# undefined-citation warnings are real rather than inherited from a stale .bbl.
#
# Writes only to nff_paper/review/build/. Never touches paper/.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$REPO/paper"
OUT="$REPO/nff_paper/review/build"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/snreview.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

export PATH="/Library/TeX/texbin:/opt/homebrew/bin:$PATH"

cp "$SRC"/*.tex "$SRC"/*.bst "$SRC"/*.cls "$SRC"/*.bib "$WORK"/ 2>/dev/null || true
cp "$SRC"/*.eps "$WORK"/ 2>/dev/null || true
cp -R "$SRC"/Images "$WORK"/ 2>/dev/null || true

STAMP="$(date +%Y%m%d-%H%M)"
echo "snapshot: $(wc -c < "$WORK/sn-article.tex") bytes  md5 $(md5 -q "$WORK/sn-article.tex")  taken $STAMP"

cd "$WORK"
pdflatex -interaction=nonstopmode -file-line-error sn-article.tex > pass1.log 2>&1 || true
bibtex sn-article                                                 > bibtex.log 2>&1 || true
pdflatex -interaction=nonstopmode -file-line-error sn-article.tex > pass2.log 2>&1 || true
pdflatex -interaction=nonstopmode -file-line-error sn-article.tex > pass3.log 2>&1 || true

if grep -qE '^!' pass3.log; then
  echo "BUILD FAILED -- real errors in the document:"
  grep -E '^!' -A4 pass3.log | head -40
  exit 1
fi

# A bibtex error does NOT show up as '^!' in the pdflatex log: pdflatex simply reads
# whatever .bbl bibtex managed to emit and reports nothing. Missed on 2026-08-20, when a
# '%'-commented .bib entry left a stray '@' that bibtex choked on -- the isolated build
# stayed green while latexmk refused to run at all. Check the .blg directly.
if grep -qiE 'error message|I found no|Illegal|missing a field name' sn-article.blg; then
  echo "BUILD FAILED -- bibtex errors:"
  grep -iE -B2 -A2 'error|missing a field name' sn-article.blg | head -30
  exit 1
fi

# Canonical names: the review prompts speak of main.pdf / main.log / main.aux.
mkdir -p "$OUT/pages"
rm -f "$OUT"/pages/*.png
cp sn-article.tex "$OUT/sn-article.snapshot.tex"
cp sn-article.pdf "$OUT/main.pdf"
cp pass3.log      "$OUT/main.log"
cp sn-article.aux "$OUT/main.aux"
cp sn-article.bbl "$OUT/main.bbl" 2>/dev/null || true
cp sn-article.blg "$OUT/main.blg" 2>/dev/null || true
cp sn-article.toc "$OUT/main.toc" 2>/dev/null || true
cp "$SRC"/sn-bibliography.bib "$OUT/" 2>/dev/null || true
rm -rf "$OUT/Images"; cp -R "$SRC"/Images "$OUT/Images" 2>/dev/null || true

# Rasterise once here so the FIGURES agent can look at the paper instead of
# spending its context rebuilding it.
pdftoppm -r 150 -png "$OUT/main.pdf" "$OUT/pages/page"

# The section map every agent is handed, so a 391 KB file is swept section by
# section rather than read whole.
grep -nE '^\\(section|subsection|subsubsection|appendix|section\*)' sn-article.tex \
  > "$OUT/section-map.txt" || true

PAGES="$(pdfinfo "$OUT/main.pdf" | awk '/^Pages:/ {print $2}')"
cat > "$OUT/BUILD-INFO.txt" <<EOF
snapshot taken : $STAMP
source         : $SRC/sn-article.tex
source bytes   : $(wc -c < "$OUT/sn-article.snapshot.tex")
source md5     : $(md5 -q "$OUT/sn-article.snapshot.tex")
pages          : $PAGES
undefined refs : $(grep -c 'Reference.*undefined' pass3.log || true)
undefined cites: $(grep -c 'Citation.*undefined' pass3.log || true)
overfull hbox  : $(grep -c 'Overfull \\hbox' pass3.log || true)
underfull hbox : $(grep -c 'Underfull \\hbox' pass3.log || true)
EOF

echo "OK  $PAGES pages  ->  nff_paper/review/build/"
cat "$OUT/BUILD-INFO.txt"
echo "rasterised: $(ls "$OUT"/pages/*.png | wc -l | tr -d ' ') page PNGs"
echo "section map: $(wc -l < "$OUT/section-map.txt" | tr -d ' ') headings"
