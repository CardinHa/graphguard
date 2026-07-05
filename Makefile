# GraphGuard developer commands
# Usage: make <target>

SAMPLE := examples/sample_project

.PHONY: help install test analyze train report dashboard api clean lint

help:           ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:        ## Install package + all extras + dev dependencies
	pip install -r requirements.txt
	pip install -e ".[dev,gnn,serve,dash,git]"

test:           ## Run the full test suite
	pytest tests/ -v

cov:            ## Run tests with coverage report
	pytest tests/ -v --cov=graphguard --cov-report=term-missing

analyze:        ## Parse the sample project and build its graph
	python -m graphguard.cli analyze $(SAMPLE)

train:          ## Train the GNN on the sample project
	python -m graphguard.cli train $(SAMPLE) --epochs 200

report:         ## Print the risk report for the sample project
	python -m graphguard.cli report $(SAMPLE) --top-n 20

dashboard:      ## Launch the Streamlit dashboard
	python -m graphguard.cli dashboard $(SAMPLE)

api:            ## Launch the FastAPI server
	python -m graphguard.cli api

clean:          ## Remove generated outputs and caches
	rm -rf $(SAMPLE)/outputs/* outputs/* .pytest_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
