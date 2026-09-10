# Convenience targets. Everything here is safe to run on the login node except `test-full`,
# which needs a compute node (the login node caps virtual memory at 8 GB — see CLAUDE.md §1).

PY ?= /work/jvl210002/migration/envs/bidir-cu129/bin/python
export PYTHONPATH := $(CURDIR)/src:/work/jvl210002/migration/obtune/src

.PHONY: check test test-full dry data status clean-scratch

## tests that fit in the login node's memory cap, plus every pipeline dry-run
check: test dry
	@echo "OK"

test:
	$(PY) -m pytest tests/ -q --ignore=tests/test_train_path.py

## the full suite, including the tokenizer-dependent tests. Needs a compute node.
test-full:
	sbatch -p dev -t 00:30:00 -c 8 --mem=32G -J bidir_tests \
	  -o /work/jvl210002/migration/tmp/bidir_tests_%j.out \
	  --wrap 'source $(CURDIR)/scripts/env.sh && cd $(CURDIR) && python -m pytest tests/ -q'

## a broken pipeline is invisible until submission day; dry-run them all
dry:
	@for t in small small_zh large ladder exec relearn fullft base_replicate; do \
	  $(PY) scripts/slurm/pipeline_grid.py --tier $$t --dry-run >/dev/null || exit 1; done
	@$(PY) scripts/slurm/pipeline_gate.py --dry-run >/dev/null
	@$(PY) scripts/slurm/pipeline_mech.py --all --dry-run >/dev/null
	@$(PY) scripts/slurm/pipeline_attrib.py --dry-run >/dev/null
	@echo "all pipelines dry-run clean"

## build every registered domain cell (CPU; a few minutes)
data:
	@for d in code mt_en-de mt_de-en mt_en-zh mt_zh-en sql d2t fmt exec fmt_novel; do \
	  $(PY) scripts/10_build_domain.py --domain $$d || exit 1; done
	@for s in 10 50 100; do $(PY) scripts/10_build_domain.py --domain fmt_lossy$$s || exit 1; done

status:
	@$(PY) scripts/00_status.py

## temporary adapters written by the mechanism experiments
clean-scratch:
	rm -rf $${TMPDIR:-/work/jvl210002/migration/tmp}/alpha_* $${TMPDIR:-/work/jvl210002/migration/tmp}/ablate_*
