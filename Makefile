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


.PHONY: check
check:
	@rm -f .coverage*
	@TOX_SKIP_ENV='pypy3|bandit|coverage' poetry run tox run-parallel -p auto


.PHONY: lint
lint:
	@poetry run tox run-parallel -m lint


.PHONY: repl
repl:
	@BASILISP_USE_DEV_LOGGER=true poetry run basilisp repl


LOGLEVEL ?= INFO
.PHONY: nrepl-server
nrepl-server:
	@BASILISP_USE_DEV_LOGGER=true BASILISP_LOGGING_LEVEL=$(LOGLEVEL) poetry run basilisp nrepl-server

.PHONY: test
test:
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


.PHONY: pypy-shell
pypy-shell:
	@docker run -it \
		--mount src=`pwd`,target=/usr/src/app,type=bind \
		--workdir /usr/src/app \
		pypy:3.10-7.3-slim-buster \
		/bin/sh -c 'pip install -e . && basilisp repl'
