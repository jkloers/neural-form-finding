# The deck needs XeTeX: it sets EB Garamond and Hiragino Sans through fontspec, and pdflatex
# stops dead at fontspec.sty. Without this a bare `latexmk` here picks pdflatex and fails with
# "The fontspec package requires either XeTeX or LuaTeX", leaving no PDF behind.
#
# 5 is xelatex via xdvipdfmx, which is what slides/build.sh runs.
$pdf_mode = 5;
