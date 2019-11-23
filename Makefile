.PHONY: install clean lint test

install:
	#python setup.py install --force
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
	flake8 npml/ tests/

test:
	pip install tox
	tox

wheel:
	python setup.py sdist bdist_wheel
