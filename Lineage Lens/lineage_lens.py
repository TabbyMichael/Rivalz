import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import json
import uuid
import datetime
import numpy as np

class LineageLens:
    def __init__(self):
        self.pipeline = []
        self.datasets = {}
        self.metadata = []
        self.graph = nx.DiGraph()

    def load_data(self, filename='chicago-ridesharing-vehicles.csv'):
        df = pd.read_csv(filename)
        dataset_id = str(uuid.uuid4())
        self.datasets[dataset_id] = df
        self.graph.add_node(dataset_id, label="Original Data")
        return dataset_id

    def calculate_hash(self, df):
        return hashlib.sha256(pd.DataFrame.to_numpy(df).tobytes()).hexdigest()

    def add_transformation(self, input_id, transformation_type, params=None, output_id=None):
        self.pipeline.append((input_id, transformation_type, params))
        if output_id:
            self.graph.add_edge(input_id, output_id)

    def transform_data(self, input_id, transformation_type, params=None):
        df = self.datasets[input_id].copy()
        input_hash = self.calculate_hash(df)
        output_id = str(uuid.uuid4())

        # Data Quality Checks
        missing_before = df.isnull().sum().sum()
        duplicates_before = df.duplicated().sum()

        if transformation_type == 'clean_missing':
            df = df.dropna()
        elif transformation_type == 'filter_year':
            if params and 'value' in params:
                df = df[df['MODEL_YEAR'].notna()]
                df = df[df['MODEL_YEAR'] > params['value']]
            else:
                raise ValueError("Filter transformation requires 'value' parameter")
        elif transformation_type == 'aggregate_make':
            df = df.groupby('MAKE').size().reset_index(name='count')
        elif transformation_type == 'enrich_trips':
            df['TRIPS_PER_YEAR'] = df['NUMBER_OF_TRIPS'] * 12
        elif transformation_type == 'convert_reported_month':
            df['REPORTED_MONTH'] = df['REPORTED_MONTH'].astype(str)

        output_hash = self.calculate_hash(df)
        self.datasets[output_id] = df
        self.graph.add_node(output_id, label=transformation_type)
        self.add_transformation(input_id, transformation_type, params, output_id)

        # Data Quality Checks After
        missing_after = df.isnull().sum().sum()
        duplicates_after = df.duplicated().sum()

        metadata_entry = {
            'input_id': input_id,
            'output_id': output_id,
            'transformation_type': transformation_type,
            'params': params,
            'input_hash': input_hash,
            'output_hash': output_hash,
            'timestamp': datetime.datetime.now().isoformat(),
            'missing_before': missing_before,
            'missing_after': missing_after,
            'duplicates_before': duplicates_before,
            'duplicates_after': duplicates_after,
        }
        self.metadata.append(metadata_entry)
        return output_id

    def visualize_lineage(self):
        pos = nx.spring_layout(self.graph)
        labels = nx.get_node_attributes(self.graph, 'label')
        nx.draw(self.graph, pos, with_labels=True, labels=labels, node_size=1500, node_color='lightblue', font_size=10)
        plt.show()

    def export_metadata(self, filename='metadata.json'):
        # Convert numpy types to native Python types for JSON serialization
        metadata_copy = []
        for entry in self.metadata:
            entry_copy = entry.copy()
            for key, value in entry_copy.items():
                if isinstance(value, (np.int64, np.int32)):
                    entry_copy[key] = int(value)
            metadata_copy.append(entry_copy)
            
        with open(filename, 'w') as f:
            json.dump(metadata_copy, f, indent=4)

    def visualize_distribution(self, dataset_id, column_name, plot_type='hist'):
        df = self.datasets[dataset_id][column_name].dropna()
        if plot_type == 'hist':
            plt.hist(df)
        elif plot_type == 'box':
            sns.boxplot(x=df)
        plt.title(f'Distribution of {column_name} in {dataset_id}')
        plt.show()

    def visualize_counts(self, dataset_id, column_name):
        sns.countplot(x=self.datasets[dataset_id][column_name])
        plt.title(f'Counts of {column_name} in {dataset_id}')
        plt.show()

    def visualize_scatter(self, dataset_id, col1, col2):
        sns.scatterplot(x=col1, y=col2, data=self.datasets[dataset_id])
        plt.title(f'Scatter plot of {col1} vs {col2} in {dataset_id}')
        plt.xticks(rotation=90)
        plt.show()

    def visualize_data_diff(self, before_id, after_id):
        df_before = self.datasets[before_id]
        df_after = self.datasets[after_id]
        diff = pd.concat([df_before, df_after]).drop_duplicates(keep=False)
        print("Data Differences:")
        print(diff)

    def compare_datasets(self, dataset_id1, dataset_id2):
        df1 = self.datasets[dataset_id1]
        df2 = self.datasets[dataset_id2]
        diff = pd.concat([df1, df2]).drop_duplicates(keep=False)
        return diff

# Example Usage:
lens = LineageLens()
initial_data = lens.load_data()

data1 = lens.transform_data(initial_data, 'clean_missing')
data2 = lens.transform_data(data1, 'filter_year', {'value': 2015})
data3 = lens.transform_data(data2, 'enrich_trips')
data4 = lens.transform_data(data3, 'convert_reported_month')
data5 = lens.transform_data(data4, 'aggregate_make')

lens.visualize_lineage()
lens.export_metadata()

# Visualizations
lens.visualize_distribution(initial_data, 'NUMBER_OF_TRIPS')
lens.visualize_counts(data5, 'MAKE')
lens.visualize_scatter(initial_data, 'MODEL_YEAR', 'NUMBER_OF_TRIPS')
lens.visualize_data_diff(data1, data2)

# Dataset Comparison
diff_df = lens.compare_datasets(data2, data3)
print("\nDataset Differences:")
print(diff_df)