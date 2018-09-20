.PHONY: repl
repl:
	@pipenv install
	@pipenv run python -m basilisp.cli repl


.PHONY: test
test:
	@pipenv install --dev
	@pipenv run tox
