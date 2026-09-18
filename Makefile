PYTHON ?= python3

.PHONY: test test-python test-rtl vectors train verify-model benchmark lob-data train-lob verify-lob predict-lob lob-benchmark validate-software walk-forward improve-lob improve-rtl
test: test-python test-rtl

# Reuse the saved model from improve-lob; no training or board is required.
improve-rtl:
	$(PYTHON) -m scripts.benchmark_rtl

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

walk-forward:
	$(PYTHON) -m scripts.generate_lob --seed 2027 --output build/lob-walk-forward/snapshots.csv
	$(PYTHON) -m models.training.walk_forward_lob --input build/lob-walk-forward/snapshots.csv --kind synthetic --source "Local pressure-driven toy generator, fresh seed 2027"
	$(PYTHON) -m scripts.verify_lob_replay --model build/lob-walk-forward/model --input build/lob-walk-forward/snapshots.csv
	$(PYTHON) -m scripts.verify_export --model build/lob-walk-forward/model
	$(PYTHON) -m scripts.predict_lob --model build/lob-walk-forward/model --input build/lob-walk-forward/snapshots.csv --output build/lob-walk-forward/predictions.csv

improve-lob:
	$(PYTHON) -m scripts.generate_lob --seed 2028 --output build/lob-improved/snapshots.csv
	$(PYTHON) -m models.training.walk_forward_lob --input build/lob-improved/snapshots.csv --kind synthetic --source "Local pressure-driven toy generator, fresh seed 2028" --seed 2028 --profile stationary --output build/lob-improved/model
	$(PYTHON) -m scripts.verify_lob_replay --model build/lob-improved/model --input build/lob-improved/snapshots.csv
	$(PYTHON) -m scripts.verify_export --model build/lob-improved/model
	$(PYTHON) -m scripts.predict_lob --model build/lob-improved/model --input build/lob-improved/snapshots.csv --output build/lob-improved/predictions.csv
