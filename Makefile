.PHONY: setup-dev
setup-dev:
	@pipenv install --dev
	@pipenv install -e .


.PHONY: release
release:
	@rm -rf ./dist
	@pipenv run python setup.py sdist bdist_wheel --universal
	@pipenv run twine upload dist/*


.PHONY: repl
repl:
	@pipenv run basilisp repl


.PHONY: test
test:
	@pipenv run tox
