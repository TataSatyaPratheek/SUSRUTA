#!/bin/bash
# Run all tests with coverage reporting

echo "Running SUSRUTA test suite with coverage..."

# Install development dependencies if needed
pip install pytest pytest-cov

# Run tests with coverage
python -m pytest --cov=susruta -v

# Generate coverage report
python -m pytest --cov=susruta --cov-report=html

echo "Test suite completed. HTML coverage report available in htmlcov/ directory."