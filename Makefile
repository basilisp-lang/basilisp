.PHONY: setup-dev
setup-dev:
	@pipenv install --dev
	@pipenv install -e .

.PHONY: repl
repl:
	@pipenv run basilisp repl


.PHONY: test
test:
	@pipenv run tox
