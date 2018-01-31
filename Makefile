.PHONY: format
format:
	@pipenv install --dev
	@pipenv run yapf --recursive --in-place ./apylisp/* --exclude *.lpy

.PHONY: lint
lint:
	@pipenv install --dev
	@pipenv run python -m pyflakes .


.PHONY: repl
repl:
	@pipenv install
	@pipenv run python -m apylisp.main


.PHONY: test
test:
	@pipenv install --dev
	@pipenv run python -m pytest


.PHONY: typecheck
typecheck:
	@pipenv run mypy --ignore-missing-imports --follow-imports=skip apylisp
