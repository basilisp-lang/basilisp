.PHONY: setup-dev
setup-dev:
	@pipenv install --dev
	@pipenv install -e .


.PHONY: docs
docs:
	@pipenv run sphinx-build -M html "./docs" "./docs/_build"


.PHONY: repl
repl:
	@pipenv run basilisp repl


.PHONY: test
test:
	@pipenv run tox
