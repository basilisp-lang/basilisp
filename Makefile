DOCSOURCEDIR = "./docs"
DOCBUILDDIR = "./docs/_build"

.PHONY: clean-docs
clean-docs:
	@rm -rf ./docs/_build


.PHONY: docs
docs:
	@poetry run sphinx-build -M html "$(DOCSOURCEDIR)" "$(DOCBUILDDIR)"


.PHONY: livedocs
livedocs:
	@poetry run sphinx-autobuild "$(DOCSOURCEDIR)" "$(DOCBUILDDIR)" -b html --watch "./src"


.PHONY: format
format:
	@poetry run sh -c 'isort . && black .'
	@cargo fmt --manifest-path rust/Cargo.toml


.PHONY: compile
compile:
	@maturin develop


.PHONY: check
check: compile
	@rm -f .coverage*
	@TOX_SKIP_ENV='pypy3|bandit|coverage' poetry run tox run-parallel -p auto


.PHONY: rust-check
rust-check:  # Check Rust code for lints and formatting
	@cargo fmt --manifest-path rust/Cargo.toml --check
	@cargo clippy --manifest-path rust/Cargo.toml


.PHONY: lint
lint: rust-check
	@poetry run tox run-parallel -m lint


.PHONY: repl
repl:
	@BASILISP_USE_DEV_LOGGER=true poetry run basilisp repl


LOGLEVEL ?= INFO
.PHONY: nrepl-server
nrepl-server:
	@BASILISP_USE_DEV_LOGGER=true BASILISP_LOGGING_LEVEL=$(LOGLEVEL) poetry run basilisp nrepl-server


.PHONY: test
test: compile
	@rm -f .coverage*
	@TOX_SKIP_ENV='pypy3' poetry run tox run-parallel -m test


.PHONY: type-check
type-check:
	@poetry run tox run-parallel -m mypy


lispcore.py:
	@BASILISP_DO_NOT_CACHE_NAMESPACES=true \
		poetry run basilisp run -c \
			'(spit "lispcore.py" @#'"'"'basilisp.core/*generated-python*)'
	@poetry run black lispcore.py


.PHONY: clean
clean:
	@rm -rf ./lispcore.py
	@rm -rf rust/target/


.PHONY: pypy-shell
pypy-shell:
	@docker run -it \
		--mount src=`pwd`,target=/usr/src/app,type=bind \
		--mount src="${HOME}/.local/share/basilisp",target="/root/.local/share/basilisp",type=bind \
		--workdir /usr/src/app \
		pypy:3.1-7.3-slim \
		/bin/sh -c 'pip install -e . && basilisp repl'
