.PHONY: install clean lint test wheel

install:
	pip install -e .

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .tox/
	find . -name \*.pyc -delete
	find . -name \*.pyo -delete
	find . -name \__pycache__ -delete

lint:
	flake8 npml/ tests/ --show-source --statistics

test:
	pip install tox
	tox

wheel:
	python setup.py sdist bdist_wheel
