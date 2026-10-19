all: run analyze

run:
	.venv/bin/python -m lbrep.run

h0:
	.venv/bin/python -m lbrep.synthetic

analyze:
	.venv/bin/python -m lbrep.analyze

selftest:
	.venv/bin/python -m lbrep.analyze --selftest

canary:
	.venv/bin/python -m lbrep.run llama2-7b llama3-8b

.PHONY: all run h0 analyze selftest canary
