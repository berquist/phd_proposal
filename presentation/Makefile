default:
	latexmk -pdf -shell-escape presentation.tex
	gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dNOPAUSE -dBATCH -sOutputFile=proposal_presentation.pdf presentation.pdf

clean:
	latexmk -c
