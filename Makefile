.PHONY: setup-dev
setup-dev:
	@pipenv install --dev
	@pipenv install -e .


.PHONY: release
release:
	@rm -rf ./build
	@rm -rf ./dist
	@pipenv run python setup.py sdist bdist_wheel --universal
	@pipenv run twine upload dist/*


.PHONY: docs
docs:
	@pipenv run sphinx-build -M html "./docs" "./docs/_build"


.PHONY: format
format:
	@pipenv run sh -c 'isort --profile black . && black .'


lispcore.py:
	@BASILISP_DO_NOT_CACHE_NAMESPACES=true \
		pipenv run basilisp run -c \
		'(with [f (python/open "lispcore.py" "w")] (.write f basilisp.core/*generated-python*))'
	@pipenv run black lispcore.py


.PHONY: clean
clean:
	@rm -rf ./lispcore.py


.PHONY: repl
repl:
	@BASILISP_USE_DEV_LOGGER=true pipenv run basilisp repl


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
	@TOX_SKIP_ENV='pypy3|safety|coverage' pipenv run tox -p 4
