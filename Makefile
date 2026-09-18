PYTHON ?= python3

.PHONY: test test-python test-rtl vectors train verify-model benchmark lob-data train-lob verify-lob predict-lob lob-benchmark validate-software
test: test-python test-rtl

validate-software:
	$(PYTHON) -m scripts.validate_software

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

lob-data:
	$(PYTHON) -m scripts.generate_lob
	$(PYTHON) -m scripts.prepare_lob --input build/lob-data/snapshots.csv --kind synthetic --source "Local pressure-driven toy generator"

train-lob:
	$(PYTHON) -m models.training.train_lob

verify-lob:
	$(PYTHON) -m scripts.verify_export --model build/lob
	$(PYTHON) -m scripts.verify_lob_replay

predict-lob:
	$(PYTHON) -m scripts.predict_lob

lob-benchmark: lob-data
	$(MAKE) train-lob
	$(MAKE) verify-lob
	$(MAKE) predict-lob
