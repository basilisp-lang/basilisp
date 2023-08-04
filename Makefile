.PHONY: docs
docs:
	@poetry run sphinx-build -M html "./docs" "./docs/_build"


.PHONY: format
format:
	@poetry run sh -c 'isort . && black .'


.PHONY: repl
repl:
	@BASILISP_USE_DEV_LOGGER=true poetry run basilisp repl


.PHONY: test
test:
	@rm -f .coverage*
	@TOX_SKIP_ENV='pypy3|safety|coverage' poetry run tox run-parallel -p 4


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
