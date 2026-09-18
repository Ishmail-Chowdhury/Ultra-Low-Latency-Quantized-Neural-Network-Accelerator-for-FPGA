PYTHON ?= python3

.PHONY: test test-python test-rtl vectors
test: test-python test-rtl

test-python:
	$(PYTHON) -m unittest discover -s tests -v

test-rtl:
	$(PYTHON) -m unittest discover -s tb -t . -v

vectors:
	$(PYTHON) -m models.quantization.dense_reference --output build/fixture
