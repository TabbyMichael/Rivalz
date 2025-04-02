# LineageLens

LineageLens is a powerful Python library for tracking and visualizing data transformations in data analysis pipelines. It provides comprehensive tools for monitoring data lineage, quality metrics, and transformation history, specifically designed for analyzing Chicago ridesharing vehicle data.

## Features

- **Data Lineage Tracking**: Automatically tracks and visualizes the complete transformation history of your data
- **Data Quality Monitoring**: Tracks missing values and duplicates before and after each transformation
- **Transformation Pipeline**: Supports various data transformations including:
  - Cleaning missing values
  - Filtering by year
  - Aggregating by make
  - Enriching trip data
  - Converting data types
- **Visualization Tools**:
  - Data lineage graphs
  - Distribution plots
  - Count plots
  - Scatter plots
  - Data difference analysis
- **Metadata Export**: Exports detailed transformation metadata in JSON format

## Installation

1. Clone the repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- pandas
- networkx
- matplotlib
- seaborn
- numpy

## Usage

### Basic Usage

```python
# Initialize LineageLens
lens = LineageLens()

# Load your data
initial_data = lens.load_data('chicago-ridesharing-vehicles.csv')

# Apply transformations
clean_data = lens.transform_data(initial_data, 'clean_missing')
filtered_data = lens.transform_data(clean_data, 'filter_year', {'value': 2015})
enriched_data = lens.transform_data(filtered_data, 'enrich_trips')

# Visualize the data lineage
lens.visualize_lineage()

# Export transformation metadata
lens.export_metadata()
```

### Visualization Examples

```python
# Distribution plot
lens.visualize_distribution(initial_data, 'NUMBER_OF_TRIPS')

# Count plot
lens.visualize_counts(filtered_data, 'MAKE')

# Scatter plot
lens.visualize_scatter(initial_data, 'MODEL_YEAR', 'NUMBER_OF_TRIPS')

# Compare datasets
lens.visualize_data_diff(clean_data, filtered_data)
```

## Data Quality Features

LineageLens automatically tracks data quality metrics throughout the transformation pipeline:
- Missing value counts
- Duplicate record counts
- Data hashes for version tracking
- Transformation timestamps

## Metadata Tracking

For each transformation, LineageLens records:
- Input and output dataset IDs
- Transformation type and parameters
- Data quality metrics before and after transformation
- Timestamps
- Data hashes

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.