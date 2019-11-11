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
	@pipenv run black .


.PHONY: repl
repl:
	@BASILISP_USE_DEV_LOGGER=true pipenv run basilisp repl


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
	@TOX_SKIP_ENV='pypy3|safety' pipenv run tox -p 4
