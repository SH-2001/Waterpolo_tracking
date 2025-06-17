# Water Polo Tracking

A computer vision system for tracking water polo players and the ball during matches. This project uses YOLOv8 for object detection and implements custom tracking algorithms for water polo-specific scenarios.

## Features

- Player detection and tracking
- Ball detection and tracking
- Dataset creation and management tools
- Data augmentation pipeline
- Visual review interface for annotations

## Project Structure

```
Waterpolo_Tracking/
├── data/               # Dataset storage (ignored by git)
├── models/            # Trained YOLO models
├── output/            # Output files and results (ignored by git)
├── pipelines/         # Data processing pipelines
├── tests/             # Test files
├── trackers/          # Custom tracking algorithms
├── utils/             # Utility functions
│   └── data_pipeline.py  # Dataset creation and management
├── config.yaml        # Configuration file
├── main.py           # Main application entry point
└── pyproject.toml    # Project dependencies
```

## Setup

1. Install dependencies using Poetry:
```bash
poetry install
```

2. Activate the virtual environment:
```bash
poetry shell
```

## Development

- Use `poetry` for dependency management
- Follow PEP 8 style guide
- Write tests for new features
- Update documentation as needed
