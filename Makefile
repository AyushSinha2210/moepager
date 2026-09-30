# moepager — top-level entry points.
CARGO ?= cargo
PY ?= python3
RUFF ?= $(shell command -v ruff 2>/dev/null || echo "uvx ruff")
OUT ?= out
BIN := target/release/moepager
DAEMON := target/release/moepagerd

.PHONY: all build test test-rust test-py lint fmt fixtures demo bench microbench clean

all: build

build:
	$(CARGO) build --release

test: test-rust test-py

test-rust: fixtures
	$(CARGO) test --workspace

test-py:
	cd python && $(PY) -m pytest -q

lint:
	$(CARGO) fmt --all -- --check
	$(CARGO) clippy --workspace --all-targets -- -D warnings
	cd python && $(RUFF) check .

fmt:
	$(CARGO) fmt --all
	cd python && $(RUFF) format .

fixtures:
	$(PY) python/moepager_tools/gguf_fixtures.py fixtures/generated --big

# Everything runnable without a model, root or llama.cpp (synthetic routing).
demo: build fixtures
	bash scripts/demo.sh

# Real-hardware baseline matrix (UNTESTED-ON-HW). Needs LLAMA_CLI and MODEL:
#   make bench LLAMA_CLI=~/llama.cpp/build/bin/llama-cli MODEL=~/models/Qwen3-30B-A3B-Q4_K_M.gguf
bench: build
	@if [ -z "$(LLAMA_CLI)" ] || [ -z "$(MODEL)" ]; then \
	  echo "make bench needs LLAMA_CLI=<llama-cli> MODEL=<model.gguf> (see BENCHMARKS.md)"; exit 2; fi
	LLAMA_CLI="$(LLAMA_CLI)" MODEL="$(MODEL)" bash bench/run_llama.sh

# Experiment C: fault-driven vs bulk expert reads. MODEL= a big file on the target disk.
microbench: build
	@if [ -z "$(MODEL)" ]; then echo "make microbench needs MODEL=<large file>"; exit 2; fi
	$(BIN) fault-io --file "$(MODEL)" --units 48 --threads 6

clean:
	$(CARGO) clean
	rm -rf $(OUT) fixtures/generated
