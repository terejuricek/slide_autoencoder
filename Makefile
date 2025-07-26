.PHONY: help install clean test format

help:
	@echo "Available commands:"
	@echo "  install      Install package in development mode"
	@echo "  clean        Clean build artifacts"
	@echo "  test         Run tests"
	@echo "  format       Format code (black)"

install:
	pip install -e .

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

test:
	python tests/overview.py

format:
	black src/ tests/ scripts/

# Training shortcuts
train-basic:
	python scripts/quick_start_real_data.py

prepare-data:
	python -m src.training.prepare_data

visualize:
	python -m src.visualization.model_visualization
