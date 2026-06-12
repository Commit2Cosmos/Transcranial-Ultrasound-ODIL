LATEX_DIR = latex
BUILD_DIR = build
NAME = ld2022-report

.PHONY: report clean

report:
	latexmk -pdf -cd -jobname=$(NAME) -outdir=../$(BUILD_DIR) $(LATEX_DIR)/$(NAME).tex
	mv ./$(BUILD_DIR)/$(NAME).pdf ../$(NAME).pdf

clean:
	rm -rf ./$(BUILD_DIR)/
