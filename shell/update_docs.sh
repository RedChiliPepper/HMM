#!/bin/bash

rm -rf docs/build docs/source/autoapi/

# Generate documentation for all components
#sphinx-apidoc -o docs/source/ src/attractor_analysis/ -f  # Python package
#sphinx-apidoc -o docs/source/ test/ -f -M -t docs/_templates/empty  # Tests
#sphinx-apidoc -o docs/source/ data/ -f -M -t docs/_templates/empty  # Data
#sphinx-apidoc -o docs/source/ logs/ -f -M -t docs/_templates/empty  # Logs
#sphinx-apidoc -o docs/source/ experiments/ -f -M -t docs/_templates/empty  # Experiments
#sphinx-apidoc -o docs/source/ scripts/ -f -M -t docs/_templates/empty  # Scripts


# Clean existing docs
rm -rf docs/build docs/source/autoapi/

# Generate API docs
sphinx-apidoc -o docs/source/ src/attractor_analysis/ -f --separate

# Generate docs for other components
sphinx-apidoc -o docs/source/ test/ -f -M -t docs/_templates/empty
sphinx-apidoc -o docs/source/ scripts/ -f -M -t docs/_templates/empty

# Build HTML docs
cd docs && make clean && make html
