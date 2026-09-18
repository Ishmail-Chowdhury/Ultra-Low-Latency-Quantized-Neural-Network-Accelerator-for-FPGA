PYTHON ?= python3

.PHONY: test test-python vectors
test: test-python

test-python:
	$(PYTHON) -m unittest discover -s tests -v

vectors:
	$(PYTHON) -m models.quantization.dense_reference --output build/fixture
