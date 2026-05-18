.PHONY: install run-cases refs-cases test snake clean

## -----------------------------------------------------------------------
## Setup (uv manages the venv transparently)
## -----------------------------------------------------------------------

install:
	uv sync

## -----------------------------------------------------------------------
## Run all test-case scripts (writes <name>.csv next to each <name>.py)
## -----------------------------------------------------------------------

run-cases: install
	for f in tests/cases/*.py; do uv run python "$$f"; done

## -----------------------------------------------------------------------
## Regenerate all .ref.csv oracles from the current scripts
## (run once after a deliberate physics change, then commit the result)
## -----------------------------------------------------------------------

refs-cases: install
	for f in tests/cases/*.py; do \
		uv run python "$$f"; \
		stem="$${f%.py}"; \
		mv "$${stem}.csv" "$${stem}.ref.csv"; \
		echo "updated $${stem}.ref.csv"; \
	done

## -----------------------------------------------------------------------
## Tests
## -----------------------------------------------------------------------

test: install
	uv run pytest tests/ -v

## -----------------------------------------------------------------------
## Snakemake parametric study
## -----------------------------------------------------------------------

snake: install
	uv run snakemake --cores all

## -----------------------------------------------------------------------
## Cleanup
## -----------------------------------------------------------------------

clean:
	rm -rf results/parametric/ .snakemake/ tests/cases/*.csv
	find . -type d -name __pycache__ -exec rm -rf {} +
