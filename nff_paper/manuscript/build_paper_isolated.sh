#!/bin/bash
# Compile paper/sn-article.tex in a private scratch copy.
#
# Why: paper/ is written by more than one agent at a time. A build launched in
# paper/ races those writers -- pdftex reads a torn .tex or .aux and reports a
# brace error at a line that is provably fine, or latexmk finds the bytes
# reverted to what it last recorded and answers "Nothing to do". Neither is a
# fault in the document. Building a snapshot removes the race.
#
# The result is copied back to paper/sn-article-isolated.pdf, which nothing
# else writes. paper/sn-article.pdf is left alone.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$REPO/paper"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/snbuild.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

export PATH="/Library/TeX/texbin:$PATH"

cp "$SRC"/*.tex "$SRC"/*.bst "$SRC"/*.cls "$SRC"/*.bib "$WORK"/ 2>/dev/null || true
cp "$SRC"/sn-article.bbl "$WORK"/ 2>/dev/null || true
cp "$SRC"/*.eps "$WORK"/ 2>/dev/null || true
cp -R "$SRC"/Images "$WORK"/ 2>/dev/null || true

echo "snapshot: $(wc -c < "$WORK/sn-article.tex") bytes  md5 $(md5 -q "$WORK/sn-article.tex")"

cd "$WORK"
for pass in 1 2; do
  pdflatex -interaction=nonstopmode -file-line-error sn-article.tex > "pass$pass.log" 2>&1 || true
done

if grep -qE '^!' pass2.log; then
  echo "BUILD FAILED -- real errors in the document:"
  grep -E '^!' -A4 pass2.log | head -40
  exit 1
fi

cp sn-article.pdf "$SRC/sn-article-isolated.pdf"
echo "OK  $(grep -o '([0-9]* pages' pass2.log | tail -1 | tr -d '(')  ->  paper/sn-article-isolated.pdf"
echo "undefined refs: $(grep -c 'Reference.*undefined' pass2.log)   undefined cites: $(grep -c 'Citation.*undefined' pass2.log)"
