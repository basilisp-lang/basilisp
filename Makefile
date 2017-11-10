ifndef VIRTUAL_ENV
    $(error VIRTUAL_ENV is undefined)
endif

.PHONY: dev-install
dev-install:
	pip install -e .

.PHONY: test
test: dev-install
	pytest
