# Data Lineage Tracker

A Python tool for tracking and visualizing data transformations in data processing pipelines. This project helps you understand how your data changes through various processing steps by creating a visual graph of data lineage.

## Features

- **Data Pipeline Tracking**: Automatically tracks transformations applied to your data
- **Hash-based Data Identification**: Uniquely identifies datasets using SHA256 hashing
- **Visualization**: Creates interactive graphs showing data transformation flow
- **Metadata Capture**: Records detailed information about each transformation step
- **Error Handling**: Robust error handling and reporting throughout the pipeline

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Example

The project comes with a sample pipeline using Chicago ridesharing vehicles dataset:

```python
from lineage_tracker import run_pipeline

# Run the sample pipeline
lineage_data, initial_label = run_pipeline()
```

### Pipeline Steps

The sample pipeline includes these transformation steps:

1. **Data Cleaning**: Removes duplicates and handles missing values
2. **Filtering**: Filters vehicles based on number of trips (>200)
3. **Aggregation**: Groups data by model year and sums the trips

### Visualization

The pipeline automatically generates a visualization showing:
- Dataset nodes (blue squares)
- Transformation nodes (green circles)
- Data flow connections between nodes

## Data Requirements

The tool expects a CSV file named 'chicago-ridesharing-vehicles.csv' in the project directory. The dataset should include these columns:
- NUMBER_OF_TRIPS
- MODEL_YEAR

## How It Works

1. **Data Hashing**: Each dataset is uniquely identified using a SHA256 hash
2. **Metadata Tracking**: Records transformation parameters, timestamps, and data hashes
3. **Graph Building**: Constructs a directed graph showing data lineage
4. **Visualization**: Uses NetworkX and Matplotlib for clear visual representation

## Error Handling

The system includes robust error handling for:
- Missing input data
- Invalid transformations
- Data type mismatches
- Graph visualization issues

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.