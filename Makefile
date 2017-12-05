.PHONY: test
test:
	@pipenv install --dev
	@pipenv run python -m pytest
