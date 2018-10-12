.PHONY: setup-dev
setup-dev:
	@pipenv install --dev
	@pipenv install -e .


.PHONY: release
release:
	@rm -rf ./dist
	@pipenv run python setup.py sdist bdist_wheel --universal
	@pipenv run twine upload dist/*


.PHONY: docs
docs:
	@pipenv run sphinx-build -M html "./docs" "./docs/_build"


.PHONY: repl
repl:
	@BASILISP_USE_DEV_LOGGER=true pipenv run basilisp repl


.PHONY: test
test:
	@pipenv run tox
