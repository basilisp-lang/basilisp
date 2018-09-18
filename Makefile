.PHONY: format
format:
	@pipenv install --dev
	@pipenv run yapf --recursive --in-place ./basilisp/* --exclude *.lpy


.PHONY: lint
lint:
	@pipenv install --dev
	@pipenv run python -m prospector > lintout.txt


.PHONY: repl
repl:
	@pipenv install
	@pipenv run python -m basilisp.cli repl


.PHONY: coverage
coverage:
	@pipenv install --dev
	@pipenv run python -m pytest --cov=basilisp --cov-report html


.PHONY: test-with-coverage
test-with-coverage: coverage
	@pipenv run coveralls


.PHONY: test
test:
	@pipenv install --dev
	@pipenv run python -m pytest --disable-warnings


.PHONY: typecheck
typecheck:
	@pipenv install --dev
	@pipenv run mypy --ignore-missing-imports --follow-imports=skip basilisp
