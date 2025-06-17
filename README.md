# Water Polo Tracking

A computer vision system for tracking water polo players, ball, and goals.

## Setup with Poetry

1. Install Poetry (if you haven't already):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

4. Run the project:
   ```bash
   python main.py
   ```

## Development

- Format code:
  ```bash
  poetry run black .
  poetry run isort .
  ```

- Run tests:
  ```bash
  poetry run pytest
  ```

## Project Structure

- `data/`: Raw and processed data
- `models/`: Trained models
- `utils/`: Utility functions
- `trackers/`: Tracking algorithms
- `tests/`: Test files
- `output/`: Output files and videos 