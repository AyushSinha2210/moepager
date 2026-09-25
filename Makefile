# moepager — top-level entry points.
CARGO ?= cargo
PY ?= python3
RUFF ?= $(shell command -v ruff 2>/dev/null || echo "uvx ruff")
OUT ?= out
BIN := target/release/moepager
DAEMON := target/release/moepagerd

.PHONY: all build test test-rust test-py lint fmt fixtures demo bench clean

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
	$(PY) python/moepager_tools/gguf_fixtures.py fixtures/generated

clean:
	$(CARGO) clean
	rm -rf $(OUT) fixtures/generated
