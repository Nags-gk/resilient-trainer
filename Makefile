.PHONY: install lint test test-fast chaos rehearsal image kind-up e2e kind-down

install:
	pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
	pip install -e ".[dev]"

lint:
	ruff check . && ruff format --check .

test-fast:
	pytest tests/test_units.py

test:
	pytest

chaos:
	python scripts/chaos_bench.py

rehearsal:
	python scripts/multinode_rehearsal.py

image:
	docker build -t resilient-trainer:dev .

kind-up:
	mkdir -p /tmp/rtrain-ckpt && chmod 777 /tmp/rtrain-ckpt
	kind create cluster --name rtrain --config deploy/kind/cluster.yaml

e2e: image
	kind load docker-image resilient-trainer:dev --name rtrain
	./hack/e2e.sh

kind-down:
	kind delete cluster --name rtrain
