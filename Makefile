.PHONY: release
release:
	@rm -rf ./build
	@rm -rf ./dist
	@poetry build
	@poetry publish --username chrisrink10


.PHONY: docs
docs:
	@poetry run sphinx-build -M html "./docs" "./docs/_build"


.PHONY: format
format:
	@poetry run black .


.PHONY: repl
repl:
	@BASILISP_USE_DEV_LOGGER=true \
	 BASILISP_LOGGING_LEVEL=DEBUG \
	 BASILISP_DO_NOT_CACHE_NAMESPACES=true \
	 poetry run basilisp repl


.PHONY: pypy-shell
pypy-shell:
	@docker run -it \
		--mount src=`pwd`,target=/usr/src/app,type=bind \
		--workdir /usr/src/app \
		pypy:3.6-7.0-slim-jessie \
		/bin/sh -c 'pip install -e . && basilisp repl'


.PHONY: test
test:
	@rm -f .coverage*
	@TOX_SKIP_ENV='pypy3|safety|coverage' poetry run tox -p 4
