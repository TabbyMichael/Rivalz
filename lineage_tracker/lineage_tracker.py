# lineage_tracker.py

import pandas as pd
import numpy as np
import hashlib
import datetime
import networkx as nx
import matplotlib.pyplot as plt
import json # For storing parameters cleanly in metadata
import os   # Added for potential future use (like saving graphs)
import sys  # Added for potential future use

# --- 1. Helper Function: Calculate DataFrame Hash ---
# We need a consistent way to hash DataFrames for identification
def calculate_df_hash(df):
    """Calculates a SHA256 hash for a pandas DataFrame."""
    # Ensure df is a DataFrame before proceeding
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    # Sort dataframe by columns and index to ensure consistency
    # Handle potential mixed data types that prevent sorting gracefully
    try:
        df_sorted = df.sort_index().sort_index(axis=1)
    except TypeError:
         # If sorting fails (e.g., mixed types), hash based on a stable representation
         df_sorted = df.copy() # Work on a copy

    # Use pandas utility to hash the dataframe object, then hash the result for fixed length
    # Convert to string representation for hashing robustness across types
    try:
        pd_hash = str(pd.util.hash_pandas_object(df_sorted.astype(str)).sum())
    except Exception as e:
        print(f"Warning: Hashing encountered an issue: {e}. Falling back to basic hash.")
        # Fallback: hash the string representation of the dataframe
        pd_hash = str(df_sorted)

    return hashlib.sha256(pd_hash.encode('utf-8')).hexdigest()

# --- 2. Data Generation ---
def generate_initial_data():
    """Loads the Chicago ridesharing vehicles dataset."""
    file_path = 'chicago-ridesharing-vehicles.csv'
    if not os.path.exists(file_path):
        print(f"Error: Dataset file not found at '{file_path}'")
        print("Please download the dataset and place it in the same directory.")
        # You might want to download it automatically here if you add requests/urllib
        return None
    try:
        # Specify low_memory=False to potentially handle mixed type warnings
        df = pd.read_csv(file_path, low_memory=False)
        print("Loaded Chicago Ridesharing Data:")
        print(df.head())
        # Basic check for expected columns (optional but good practice)
        expected_cols = ['NUMBER_OF_TRIPS', 'MODEL_YEAR'] # Add others if needed
        if not all(col in df.columns for col in expected_cols):
             print(f"Warning: Expected columns {expected_cols} not all found in the dataset.")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# --- 3. Transformation Functions ---
# Each function takes a DataFrame and returns a modified DataFrame
# They also include parameters used in the transformation

def clean_data(df):
    """Simulates data cleaning: drops duplicates and rows with NaNs in key columns."""
    print("\nStep: Cleaning Data (Dropping NaNs and Duplicates)")
    # Specify subset for dropna if you only care about NaNs in certain columns
    df_cleaned = df.dropna().drop_duplicates()
    print(f"Shape before cleaning: {df.shape}, Shape after cleaning: {df_cleaned.shape}")
    return df_cleaned, {} # No specific parameters for this simple version

def filter_data(df, column_name, threshold):
    """Simulates filtering data based on a numeric column value."""
    print(f"\nStep: Filtering Data (Column '{column_name}' > {threshold})")
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found for filtering.")
        return df, {'column_name': column_name, 'threshold': threshold, 'error': 'Column not found'}
    # Ensure the column is numeric before comparison
    if pd.api.types.is_numeric_dtype(df[column_name]):
        df_filtered = df[df[column_name] > threshold].copy() # Use .copy()
        print(f"Shape before filtering: {df.shape}, Shape after filtering: {df_filtered.shape}")
        params = {'column_name': column_name, 'threshold': threshold}
    else:
        print(f"Warning: Column '{column_name}' is not numeric. Cannot filter by threshold {threshold}.")
        df_filtered = df.copy() # Return original df if column not numeric
        params = {'column_name': column_name, 'threshold': threshold, 'warning': 'Column not numeric'}
    return df_filtered, params

def aggregate_data(df, group_by_col, agg_col, agg_func='mean'):
    """Simulates aggregating data."""
    print(f"\nStep: Aggregating Data (Group by '{group_by_col}', Aggregate '{agg_col}' with '{agg_func}')")
    if group_by_col not in df.columns or agg_col not in df.columns:
         print(f"Error: Group-by column '{group_by_col}' or Aggregation column '{agg_col}' not found.")
         return df, {'group_by_col': group_by_col, 'agg_col': agg_col, 'agg_func': agg_func, 'error':'Column(s) not found'}

    try:
        df_agg = df.groupby(group_by_col, observed=True)[agg_col].agg(agg_func).reset_index() # Use observed=True for potential category types
        print("Aggregated Data:")
        print(df_agg.head())
        params = {'group_by_col': group_by_col, 'agg_col': agg_col, 'agg_func': agg_func}
    except Exception as e:
        print(f"Error during aggregation: {e}")
        df_agg = pd.DataFrame() # Return empty df on error
        params = {'group_by_col': group_by_col, 'agg_col': agg_col, 'agg_func': agg_func, 'error': str(e)}

    return df_agg, params

# --- 4. Main Pipeline Execution & Metadata Capture ---
def run_pipeline():
    """Executes the data pipeline and captures lineage metadata."""
    lineage_metadata = []
    pipeline_steps = [
        {'func': clean_data, 'params': {}},
        {'func': filter_data, 'params': {'column_name': 'NUMBER_OF_TRIPS', 'threshold': 200}},  # Filter vehicles with >200 trips
        {'func': aggregate_data, 'params': {'group_by_col': 'MODEL_YEAR',  # Group by model year
                                              'agg_col': 'NUMBER_OF_TRIPS',  # Sum trips per year
                                              'agg_func': 'sum'}},
        # Add more steps as needed
    ]

    # Initial Data
    current_df = generate_initial_data()
    if current_df is None:
        print("Failed to load initial data. Aborting pipeline.")
        return [], None # Return empty metadata and no label

    try:
        current_hash = calculate_df_hash(current_df)
        start_node_label = f"Initial Data\n({current_hash[:8]})" # Use short hash for label
    except Exception as e:
        print(f"Error hashing initial dataframe: {e}")
        return [], None

    # Store metadata about the initial state (optional but good for graph)
    initial_metadata = {
        'step_name': 'Initial Data',
        'function_name': 'generate_initial_data',
        'parameters': {},
        'input_hash': None, # No input for generation
        'output_hash': current_hash,
        'timestamp': datetime.datetime.now().isoformat(),
        'user': 'system_user' # Hardcoded for this example
    }
    lineage_metadata.append(initial_metadata)


    previous_hash = current_hash # Hash of the dataset *entering* the next step

    # Execute Pipeline Steps
    for i, step_info in enumerate(pipeline_steps):
        step_func = step_info['func']
        step_params = step_info['params']
        step_name = f"Step {i+1}: {step_func.__name__}"

        print("-" * 30)
        print(f"Executing: {step_name} with params: {step_params}")

        # Execute the transformation
        try:
            output_df, actual_params_used = step_func(current_df, **step_params) # Pass params using **
            if output_df is None or output_df.empty:
                 print(f"Warning: Step '{step_name}' produced an empty or None DataFrame.")
                 # Decide how to handle: stop pipeline, skip step, etc.
                 # For now, let's stop if it's empty after aggregation, continue otherwise
                 if step_func == aggregate_data and output_df.empty:
                     print("Stopping pipeline due to empty result after aggregation.")
                     break
                 elif output_df is None:
                     print("Stopping pipeline due to None result.")
                     break


            # Calculate output hash
            output_hash = calculate_df_hash(output_df)

        except Exception as e:
            print(f"Error during step '{step_name}': {e}")
            # Log error in metadata maybe? For now, break the pipeline
            break


        # Capture metadata for this specific transformation step
        step_metadata = {
            'step_name': step_name,
            'function_name': step_func.__name__,
            'parameters': actual_params_used, # Store the parameters actually used
            'input_hash': previous_hash,
            'output_hash': output_hash,
            'timestamp': datetime.datetime.now().isoformat(),
            'user': 'system_user' # Hardcoded
        }
        lineage_metadata.append(step_metadata)
        print(f"Input Hash: {previous_hash[:8]}... Output Hash: {output_hash[:8]}...")


        # Update for the next iteration
        current_df = output_df
        previous_hash = output_hash # The output of this step is the input for the next

    print("-" * 30)
    print("\nPipeline execution finished.")
    if current_df is not None:
        print(f"Final data hash: {previous_hash[:8]}...")
    return lineage_metadata, start_node_label # Return metadata and the label for the first data node

# --- 5. Graph Building ---
def build_lineage_graph(metadata_list, start_node_label):
    """Builds a NetworkX graph from the lineage metadata."""
    if not metadata_list: # Handle case where pipeline failed early
        return nx.DiGraph(), {}

    G = nx.DiGraph()
    node_labels = {} # For cleaner visualization labels

    # Find the initial data hash from the first entry
    initial_data_hash = None
    initial_data_timestamp = None
    for item in metadata_list:
        if item['step_name'] == 'Initial Data':
            initial_data_hash = item['output_hash']
            initial_data_timestamp = item['timestamp']
            break

    if initial_data_hash:
         G.add_node(initial_data_hash, type='dataset', timestamp=initial_data_timestamp)
         node_labels[initial_data_hash] = start_node_label # Use the label passed in


    # Process transformation steps
    for item in metadata_list:
        if item['input_hash'] is None: # Skip the initial data entry here
             continue

        step_node_id = f"{item['step_name']}_{item['timestamp']}" # Unique ID for step node
        input_hash = item['input_hash']
        output_hash = item['output_hash']
        step_name = item['step_name']
        func_name = item['function_name']
        params = item['parameters']
        timestamp = item['timestamp']
        user = item['user']


        # Add Transformation Node
        G.add_node(step_node_id, type='transformation', function=func_name, params=json.dumps(params, indent=2), user=user, timestamp=timestamp)
        node_labels[step_node_id] = step_name

        # Add Output Dataset Node (if not already added by a previous step's output)
        if output_hash not in G:
            G.add_node(output_hash, type='dataset', timestamp=timestamp) # Timestamp when created
            node_labels[output_hash] = f"Data\n({output_hash[:8]})"

        # Add Edges
        # Edge from Input Dataset -> Transformation
        if input_hash in G: # Ensure the input dataset node exists
            G.add_edge(input_hash, step_node_id)
        else:
            # This might happen if the pipeline was interrupted after hashing initial data
            # but before the first step completed metadata saving properly.
            print(f"Warning: Input hash {input_hash[:8]}... not found as a node for step {step_name}. Graph may be incomplete.")


        # Edge from Transformation -> Output Dataset
        G.add_edge(step_node_id, output_hash)

    return G, node_labels

# --- 6. Visualization (Enhanced Matplotlib/NetworkX) ---
def visualize_lineage(G, node_labels):
    """Visualizes the lineage graph using Matplotlib with enhancements."""
    if not G: # Handle empty graph if pipeline failed
         print("Graph is empty, skipping visualization.")
         return

    plt.figure(figsize=(16, 10)) # Increase figure size for more space

    # Try a different layout algorithm
    try:
        # kamada_kawai_layout often looks good for lineage-like flows
        # Increase distance between nodes using scale or dist parameter if available
        pos = nx.kamada_kawai_layout(G) # Scale parameter might not exist directly, layout adjusts spacing
    except Exception as layout_error: # Fallback if kamada_kawai fails (e.g., disconnected graph)
        print(f"Kamada-Kawai layout failed: {layout_error}. Falling back to spring layout.")
        try:
            pos = nx.spring_layout(G, k=1.5/np.sqrt(len(G.nodes())), iterations=50) # Adjust k for spacing based on node count
        except Exception as spring_error:
            print(f"Spring layout also failed: {spring_error}. Using random layout.")
            pos = nx.random_layout(G)


    # Separate nodes by type for different styling
    dataset_nodes = [node for node, attr in G.nodes(data=True) if attr.get('type') == 'dataset']
    transformation_nodes = [node for node, attr in G.nodes(data=True) if attr.get('type') == 'transformation']

    # Draw dataset nodes (squares)
    nx.draw_networkx_nodes(G, pos,
                           nodelist=dataset_nodes,
                           node_shape='s', # Square shape
                           node_size=3500,
                           node_color='skyblue',
                           label='Dataset')

    # Draw transformation nodes (circles)
    nx.draw_networkx_nodes(G, pos,
                           nodelist=transformation_nodes,
                           node_shape='o', # Circle shape
                           node_size=3500,
                           node_color='lightgreen',
                           label='Transformation')

    # Draw edges
    nx.draw_networkx_edges(G, pos,
                           arrowstyle='-|>', # Nicer arrow style '-|>' or '->'
                           arrowsize=25,    # Increase arrow size
                           edge_color='gray',
                           node_size=3500) # Make sure node_size matches node drawing for edge clipping

    # Draw labels
    nx.draw_networkx_labels(G, pos,
                            labels=node_labels,
                            font_size=10, # Increase font size
                            font_weight='bold')

    plt.title("Data Lineage Visualization (Enhanced)", fontsize=16)
    # Manually create legend elements as drawing nodes separately doesn't auto-legend well
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', label='Dataset', markerfacecolor='skyblue', markersize=12),
        plt.Line2D([0], [0], marker='o', color='w', label='Transformation', markerfacecolor='lightgreen', markersize=12)
    ]
    # Place legend smartly, avoid overlapping graph if possible
    plt.legend(handles=legend_elements, loc='best', fontsize=12)

    plt.margins(0.1) # Add some margin around the graph
    # plt.tight_layout() # Often causes issues/warnings with complex graphs, margins might be better
    plt.axis('off') # Hide the axes
    plt.show()


# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. Run the pipeline and get metadata
    lineage_data, initial_label = run_pipeline()

    # Only proceed if pipeline ran successfully and produced metadata
    if lineage_data and initial_label:
        # Optional: Print the collected metadata
        print("\nCollected Lineage Metadata:")
        # Use safe string conversion for parameters before dumping to JSON
        for item in lineage_data:
             if 'parameters' in item and item['parameters']:
                  item['parameters'] = {k: str(v) for k, v in item['parameters'].items()}
        try:
            print(json.dumps(lineage_data, indent=2))
        except TypeError as e:
             print(f"Could not serialize metadata to JSON: {e}")
             # print(lineage_data) # Print raw if JSON fails


        # 2. Build the graph
        lineage_graph, graph_labels = build_lineage_graph(lineage_data, initial_label)

        # 3. Visualize the graph
        visualize_lineage(lineage_graph, graph_labels)

        print("\nGraph visualization displayed.")
    else:
        print("\nPipeline execution failed or produced no metadata. Skipping graph generation.")