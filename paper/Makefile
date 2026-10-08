# A standalone folder: upload its LaTeX/assets to Overleaf, or run make here.
PY ?= python3
TECTONIC ?= $(if $(wildcard ../../envs/tex/bin/tectonic),../../envs/tex/bin/tectonic,tectonic)
export TECTONIC_CACHE_DIR ?= $(abspath ../../cache/tectonic)

.PHONY: paper check submission
paper:
	$(TECTONIC) -X compile --keep-logs main.tex
	$(PY) check_arr.py

check:
	$(PY) check_arr.py

submission: paper
	$(PY) check_arr.py --strict
