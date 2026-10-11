# Thin wrappers around `lang2act`. Run `make help`.
PY      ?= python
SUITE   ?= suites/core_v1.yaml
MODEL   ?= anthropic:claude-haiku-5-5
MODEL_B ?= anthropic:claude-sonnet-5-5
WORKERS ?= 6
RUNS    := runs/core_v1
tag      = $(subst :,_,$(1))

.PHONY: help install assets test lint ceiling synthetic exp-verifier exp-models exp-prompts report demo

help:
	@echo "install        pip install -e '.[anthropic,viz,dev]'"
	@echo "assets         fetch the pinned Franka Panda meshes (~33 MB)"
	@echo "test / lint    pytest / ruff"
	@echo "ceiling        oracle planner on the suite: the execution ceiling"
	@echo "synthetic      noisy synthetic planner x verifier none/symbolic/twin (pipeline check, no API)"
	@echo "exp-verifier   MODEL with verifier none vs symbolic vs twin     (needs an API key)"
	@echo "exp-models     MODEL vs MODEL_B, symbolic verifier              (needs an API key)"
	@echo "exp-prompts    MODEL with basic vs fewshot prompt               (needs an API key)"
	@echo "report         markdown report + chart over every run in $(RUNS)"
	@echo "demo           render docs/media/demo.gif"

install:
	$(PY) -m pip install -e ".[anthropic,viz,dev]"

assets:
	lang2act assets

test:
	pytest

lint:
	ruff check src tests

ceiling:
	lang2act run --suite $(SUITE) --planner oracle --out $(RUNS)/oracle-symbolic --workers $(WORKERS)

synthetic:
	for v in none symbolic twin; do \
	  lang2act run --suite $(SUITE) --planner noisy:0.5 --verifier $$v --out $(RUNS)/noisy_0.5-$$v --workers $(WORKERS); \
	done
	lang2act compare $(RUNS)/noisy_0.5-none $(RUNS)/noisy_0.5-symbolic

exp-verifier:
	for v in none symbolic twin; do \
	  lang2act run --suite $(SUITE) --planner $(MODEL) --verifier $$v --out $(RUNS)/$(call tag,$(MODEL))-$$v-basic --workers $(WORKERS); \
	done
	lang2act compare $(RUNS)/$(call tag,$(MODEL))-none-basic $(RUNS)/$(call tag,$(MODEL))-symbolic-basic

exp-models:
	for m in $(MODEL) $(MODEL_B); do \
	  lang2act run --suite $(SUITE) --planner $$m --verifier symbolic --out $(RUNS)/$$(echo $$m | tr : _)-symbolic-basic --workers $(WORKERS); \
	done
	lang2act compare $(RUNS)/$(call tag,$(MODEL))-symbolic-basic $(RUNS)/$(call tag,$(MODEL_B))-symbolic-basic

exp-prompts:
	for p in basic fewshot; do \
	  lang2act run --suite $(SUITE) --planner $(MODEL) --verifier symbolic --prompt $$p --out $(RUNS)/$(call tag,$(MODEL))-symbolic-$$p --workers $(WORKERS); \
	done
	lang2act compare $(RUNS)/$(call tag,$(MODEL))-symbolic-basic $(RUNS)/$(call tag,$(MODEL))-symbolic-fewshot

report:
	lang2act report $$(ls -d $(RUNS)/*/ | sort) --out results/report.md --plot

demo:
	lang2act demo --template bowl_then_stack --seed 0 --planner oracle --out docs/media/demo.gif --every 3
