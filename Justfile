set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# Create the virtual environment and install the `qual` command
venv:
    uv sync

# Run the offline test suite (no API calls)
test:
    uv run pytest -q

# Lint and format
lint:
    uv run ruff check
    uv run ruff format

# Run pre-commit hooks on all files
pre-commit:
    uv run pre-commit run --all-files
