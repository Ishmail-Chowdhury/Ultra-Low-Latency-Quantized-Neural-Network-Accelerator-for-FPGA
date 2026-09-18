PYTHON ?= python3

.PHONY: test test-python test-rtl vectors train verify-model benchmark
test: test-python test-rtl

test-python:
	$(PYTHON) -m unittest discover -s tests -v

test-rtl:
	$(PYTHON) -m unittest discover -s tb -t . -v

vectors:
	$(PYTHON) -m models.quantization.dense_reference --output build/fixture

train:
	$(PYTHON) -m models.training.train_digits --output build/digits

verify-model:
	$(PYTHON) -m scripts.verify_export --model build/digits

benchmark: train
	$(MAKE) verify-model
