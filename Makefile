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


# Run PyPy tests inside a Docker container for the moment since
# Pyenv on MacOS still doesn't have PyPy 3.6-7.0.0.
.PHONY: test-pypy
test-pypy:
	@docker run \
		--mount src=`pwd`,target=/usr/src/app,type=bind \
		pypy:3.6-7.0-slim-jessie \
		/bin/sh -c 'cd /usr/src/app && pip install tox && tox -e pypy3'


.PHONY: pypy-shell
pypy-shell:
	@docker run -it \
		--mount src=`pwd`,target=/usr/src/app,type=bind \
		pypy:3.6-7.0-slim-jessie \
		/bin/sh -c 'cd /usr/src/app && pip install -e . && basilisp repl'


.PHONY: test
test:
	@pipenv run tox -p 4
