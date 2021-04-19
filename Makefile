.PHONY: release
release:
	@rm -rf ./build
	@rm -rf ./dist
	@poetry run python setup.py sdist bdist_wheel --universal
	@poetry run twine upload dist/*


.PHONY: docs
docs:
	@poetry run sphinx-build -M html "./docs" "./docs/_build"


.PHONY: format
format:
	@poetry run sh -c 'isort --profile black . && black .'


lispcore.py:
	@BASILISP_DO_NOT_CACHE_NAMESPACES=true \
		poetry run basilisp run -c \
		'(with [f (python/open "lispcore.py" "w")] (.write f basilisp.core/*generated-python*))'
	@poetry run black lispcore.py


.PHONY: clean
clean:
	@rm -rf ./lispcore.py


.PHONY: repl
repl:
	@BASILISP_USE_DEV_LOGGER=true poetry run basilisp repl


.PHONY: pypy-shell
pypy-shell:
	@docker run -it \
		--mount src=`pwd`,target=/usr/src/app,type=bind \
		--workdir /usr/src/app \
		pypy:3.6-7.3-slim-buster \
		/bin/sh -c 'pip install -e . && basilisp repl'


.PHONY: test
test:
	@rm -f .coverage*
	@TOX_SKIP_ENV='pypy3|safety|coverage' poetry run tox -p 4
