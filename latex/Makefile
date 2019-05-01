export SHELL=/bin/bash
.PHONY: FORCE default clean
FILE=report
.PRECIOUS: $(FILE).pdf

default: $(FILE).pdf

%.pdf: %.tex FORCE
	pdflatex $(FILE).tex
	bibtex $(FILE).aux
	pdflatex $(FILE).tex
	pdflatex $(FILE).tex

clean:
	$(RM) *.{dvi,aux,log,toc,lof,lol,lot,dlog,bbl,blg,idx,out,tpt,svn,synctex.gz,pdf}
	$(RM) *.{nav,snm,vrb,fls,fdb_latexmk,eps} *~ *.bak

