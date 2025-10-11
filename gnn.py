# gnn_imputation.py - Imputation using a Graph Neural Network (GNN)

# --- Installation ---
# This script requires PyTorch and PyTorch Geometric.
#
# 1. Install PyTorch:
#    Follow instructions from the official website: https://pytorch.org/
#
# 2. Install PyTorch Geometric (PyG):
#    The installation command depends on your PyTorch and CUDA versions.
#    Find the correct command here: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
#
# Example CPU-only installation:
# pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
# pip install torch-geometric

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import utility functions from your existing project
from simplified_utils import (
    load_and_preprocess_data,
    add_temporal_features,
    create_contiguous_segment_gaps,
    evaluate_metrics
)

# --- 1. Graph Construction ---

def create_graph_dataset(df_discharge_features, df_contrib):
    """
    Converts the time-series DataFrame into a graph structure for the GNN.

    Args:
        df_discharge_features (pd.DataFrame): DataFrame with discharge and temporal features.
        df_contrib (pd.DataFrame): DataFrame defining the station connectivity.

    Returns:
        tuple: A tuple containing:
            - data (Data): A PyG Data object representing the graph.
            - station_map (dict): A mapping from station names to integer node indices.
            - scaler (StandardScaler): The scaler fitted on the feature data.
    """
    print("--- Building Graph Structure ---")
    
    # Get the list of stations (nodes)
    discharge_cols = [col for col in df_discharge_features.columns if not col.startswith('day_of_year_')]
    stations = sorted(discharge_cols)
    station_map = {name: i for i, name in enumerate(stations)}
    
    # Create edges from contributor data
    # An edge goes from a "contributor" (source) to a "station" (target)
    source_nodes = [station_map[s] for s in df_contrib['contributor'] if s in station_map]
    target_nodes = [station_map[s] for s in df_contrib['station'] if s in station_map]
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    
    # --- Feature Engineering for Nodes ---
    # The features for each node at each timestep will be:
    # [discharge_value, day_of_year_sin, day_of_year_cos]
    
    # We create a large tensor of shape [num_nodes, num_timesteps, num_features]
    num_nodes = len(stations)
    num_timesteps = len(df_discharge_features)
    num_features = 3 # discharge, sin, cos
    
    node_features = np.zeros((num_nodes, num_timesteps, num_features))
    
    # Fill the tensor with data from the DataFrame
    for i, station in enumerate(stations):
        node_features[i, :, 0] = df_discharge_features[station].values
        node_features[i, :, 1] = df_discharge_features['day_of_year_sin'].values
        node_features[i, :, 2] = df_discharge_features['day_of_year_cos'].values

    # Reshape for scaling: [num_nodes * num_timesteps, num_features]
    features_reshaped = node_features.reshape(-1, num_features)
    
    # Scale features
    scaler = StandardScaler()
    # We only scale the temporal features, not the discharge itself
    features_reshaped[:, 1:] = scaler.fit_transform(features_reshaped[:, 1:])

    # Reshape back to the original shape
    scaled_features = features_reshaped.reshape(num_nodes, num_timesteps, num_features)
    
    # The features for the GNN will be a tensor of shape [num_timesteps, num_nodes, num_features]
    # We transpose from [nodes, timesteps, features] to [timesteps, nodes, features]
    x = torch.tensor(scaled_features, dtype=torch.float).permute(1, 0, 2)
    
    # The target `y` is the true discharge value
    y = torch.tensor(node_features[:, :, 0], dtype=torch.float).T # Shape [timesteps, nodes]

    # Create the PyG Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    
    print(f"✓ Graph created with {data.num_nodes} nodes and {data.num_edges} edges.")
    print(f"  Feature tensor shape (timesteps, nodes, features): {data.x.shape}")
    
    return data, station_map, scaler

# --- 2. GNN Model Definition ---

class GCN(torch.nn.Module):
    """A simple Graph Convolutional Network for imputation."""
    def __init__(self, num_node_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1) # Output is a single value (discharge)

    def forward(self, x, edge_index):
        # x shape: [num_nodes, num_features]
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# --- 3. Training and Evaluation ---

def train(model, data, optimizer, criterion, train_mask):
    """Training loop for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    total_loss = 0
    
    # Pre-calculate column means from the training data for filling missing inputs
    col_means = torch.nanmean(data.y[train_mask], axis=0)
    
    # Iterate over each timestep in the training data
    num_train_steps = np.sum(train_mask)
    for t in np.where(train_mask)[0]:
        # Get the graph state at timestep t
        x_t = data.x[t] # Features for all nodes at time t
        y_t = data.y[t] # True discharge for all nodes at time t
        
        # Create a mask for valid (non-NaN) training labels at this timestep
        valid_mask = ~torch.isnan(y_t)
        if valid_mask.sum() == 0:
            continue # Skip if no observed data at this step

        # --- FIX: Handle NaNs in input features ---
        temp_x_t = x_t.clone()
        input_missing_mask = torch.isnan(temp_x_t[:, 0])
        
        # Fill missing input features with pre-calculated training data means
        for i, is_missing in enumerate(input_missing_mask):
            if is_missing:
                mean_val = col_means[i] if not torch.isnan(col_means[i]) else 0
                temp_x_t[i, 0] = mean_val
        # --- END FIX ---

        # Forward pass using the cleaned input tensor
        out = model(temp_x_t, data.edge_index).squeeze()
        
        # Calculate loss only on the valid (observed) nodes
        loss = criterion(out[valid_mask], y_t[valid_mask])
        
        # Safeguard against potential NaN loss from other sources
        if not torch.isnan(loss):
            loss.backward()
            total_loss += loss.item()
    
    optimizer.step()
    return total_loss / num_train_steps if num_train_steps > 0 else 0

def impute(model, data, test_mask):
    """Use the trained model to impute missing values."""
    model.eval()
    imputed_values = data.y.clone()

    with torch.no_grad():
        # Calculate means from the training data, to be used for filling test set gaps
        train_mask = ~test_mask
        col_means = torch.nanmean(data.y[train_mask], axis=0)

        for t in np.where(test_mask)[0]:
            x_t = data.x[t]
            y_t = data.y[t] # The original data with NaNs
            
            # Identify which values are missing
            missing_mask = torch.isnan(y_t)
            
            # The model needs discharge values as input features.
            # For missing ones, we'll make a temporary guess (e.g., column mean).
            temp_x = x_t.clone()
            for i, is_missing in enumerate(missing_mask):
                if is_missing:
                    # If the mean is also NaN (column is all NaN), use 0
                    mean_val = col_means[i] if not torch.isnan(col_means[i]) else 0
                    temp_x[i, 0] = mean_val

            # Get predictions for all nodes
            predictions = model(temp_x, data.edge_index).squeeze()
            
            # Fill in the missing values with the predictions
            imputed_values[t, missing_mask] = predictions[missing_mask]
            
    return imputed_values

# --- 4. Main Execution ---

if __name__ == '__main__':
    # Configuration
    discharge_path = 'discharge_data_cleaned.csv'
    lat_long_path = 'lat_long_discharge.csv'
    contrib_path = 'mahanadi_contribs.csv'
    
    # --- Load Data ---
    df_original, df_contrib, _, _, _ = load_and_preprocess_data(
        discharge_path, lat_long_path, contrib_path
    )
    df_with_features = add_temporal_features(df_original)

    # --- Filter Data to a smaller time range for faster execution ---
    print("\n--- Filtering data from 2010-2014 ---")
    df_with_features = df_with_features.loc['2010-01-01':'2014-12-31']
    print(f"  New data shape: {df_with_features.shape}")

    # --- Create Artificial Gaps for Evaluation ---
    print("\n--- Creating Artificial Gaps for Evaluation ---")
    discharge_cols = [col for col in df_original.columns if not col.startswith('day_of_year_')]
    gap_info = create_contiguous_segment_gaps(
        df_with_features, 
        discharge_cols, 
        gap_lengths=[30], # Evaluate on 30-day gaps
        num_intervals_per_column=1
    )
    df_gapped = gap_info[30]['gapped_data']
    true_values_map = gap_info[30]['true_values']
    
    # --- Build Graph ---
    graph_data, station_map, scaler = create_graph_dataset(df_gapped, df_contrib)
    
    # --- Prepare for Training ---
    # Split data into training (2010-2013) and testing (2014) periods
    print("\n--- Splitting data into train (2010-2013) and test (2014) sets ---")
    # We get the years from the index of the dataframe used to create the graph
    years = df_gapped.index.year
    train_mask = (years >= 2010) & (years <= 2013)
    test_mask = (years == 2014)

    print(f"  Training timesteps: {np.sum(train_mask)}")
    print(f"  Testing timesteps: {np.sum(test_mask)}")
    
    # --- Model Initialization ---
    model = GCN(num_node_features=graph_data.num_node_features, hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()

    # --- Train Model ---
    print("\n--- Training GNN Model ---")
    for epoch in range(200):
        loss = train(model, graph_data, optimizer, criterion, train_mask)
        if (epoch + 1) % 20 == 0:
            print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}')

    # --- Impute and Evaluate ---
    print("\n--- Imputing and Evaluating ---")
    imputed_tensor = impute(model, graph_data, test_mask)
    
    # Convert results back to a DataFrame for easier evaluation
    imputed_df = pd.DataFrame(
        imputed_tensor.numpy(), 
        index=df_with_features.index, 
        columns=sorted(station_map.keys())
    )
    
    # Gather true and predicted values for the gapped sections in the test set
    y_true_eval, y_pred_eval = [], []
    
    test_indices = np.where(test_mask)[0]
    
    # Get the original, complete data for the test period to get ground truth
    original_test_data = df_with_features.iloc[test_indices]
    gapped_test_data = df_gapped.iloc[test_indices]
    imputed_test_data = imputed_df.iloc[test_indices]

    for station in imputed_df.columns:
        # Find where the artificial gaps were created in the test set
        gap_mask_in_test = gapped_test_data[station].isnull()
        
        if gap_mask_in_test.sum() > 0:
            # Get the predicted values from our model at these specific gap locations
            predicted_vals = imputed_test_data[station][gap_mask_in_test].values
            
            # Get the ground truth values from the original, non-gapped data at the same locations
            true_vals_in_test = original_test_data[station][gap_mask_in_test].values
            
            y_pred_eval.extend(predicted_vals)
            y_true_eval.extend(true_vals_in_test)

    # --- Final Metrics ---
    if y_true_eval:
        metrics = evaluate_metrics(np.array(y_true_eval), np.array(y_pred_eval))
        print("\n--- GNN Imputation Performance on Artificial Gaps ---")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
    else:
        print("Could not find any artificial gaps in the test period to evaluate.")


