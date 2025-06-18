.PHONY: install install-dev build clean

install:
	pip install poetry==1.8.4
	poetry install

###################
# Developer tools #
###################

install-dev:
	pip install poetry==1.8.4
	poetry install --with dev

build:
	poetry build

clean:
	rm -rf build dist *.egg-info