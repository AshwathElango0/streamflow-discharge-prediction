import pandas as pd
import numpy as np
from missforest_imputer import ModifiedMissForest

def train_full_model(df_train_masked, distance_matrix, connectivity_matrix, all_feature_cols):
    """
    Trains the ModifiedMissForest model using all available features,
    including spatial (distance & connectivity) and temporal features.
    """
    print("--- Training Full Modified MissForest Model ---")
    
    mf_imputer = ModifiedMissForest(
        distance_matrix=distance_matrix,
        connectivity=connectivity_matrix,
        max_iter=10,
        n_estimators=100,
        random_state=42,
        distance_weighting_type='inverse',
        temporal_feature_columns=all_feature_cols
    )
    
    try:
        mf_imputer.fit(df_train_masked)
        print("Full Modified MissForest model trained.")
        return mf_imputer
    except Exception as e:
        print(f"ERROR: Full model training failed. Error: {e}")
        return None

def train_no_temporal_model(df_train_masked, distance_matrix, connectivity_matrix, all_feature_cols):
    """
    Trains the ModifiedMissForest model without temporal features.
    """
    print("--- Training Modified MissForest Model (No Temporal Features) ---")
    
    mf_imputer = ModifiedMissForest(
        distance_matrix=distance_matrix,
        connectivity=connectivity_matrix,
        max_iter=10,
        n_estimators=100,
        random_state=42,
        distance_weighting_type='inverse',
        temporal_feature_columns=[] # Exclude temporal features
    )
    
    try:
        mf_imputer.fit(df_train_masked)
        print("Modified MissForest model (No Temporal Features) trained.")
        return mf_imputer
    except Exception as e:
        print(f"ERROR: No temporal model training failed. Error: {e}")
        return None

def train_no_spatial_temporal_model(df_train_masked, distance_matrix, connectivity_matrix, all_feature_cols):
    """
    Trains a baseline ModifiedMissForest model without spatial or temporal features.
    Effectively, a basic MissForest.
    """
    print("--- Training Modified MissForest Model (No Spatial/Temporal Features) ---")
    
    # For no spatial/temporal, pass None or empty matrices/lists
    mf_imputer = ModifiedMissForest(
        distance_matrix=None, # Exclude spatial distance
        connectivity=None,    # Exclude connectivity
        max_iter=10,
        n_estimators=100,
        random_state=42,
        distance_weighting_type='inverse',
        temporal_feature_columns=[] # Exclude temporal features
    )
    
    try:
        mf_imputer.fit(df_train_masked)
        print("Modified MissForest model (No Spatial/Temporal Features) trained.")
        return mf_imputer
    except Exception as e:
        print(f"ERROR: No spatial/temporal model training failed. Error: {e}")
        return None

def train_no_contributor_model(df_train_masked, distance_matrix, connectivity_matrix, all_feature_cols):
    """
    Trains ModifiedMissForest without contributor info (hydrological connectivity).
    This is kept as an alternative configuration.
    """
    print("--- Training Modified MissForest Model (No Contributor Info) ---")
    # Create a zero matrix for connectivity to nullify its effect
    connectivity_zero = pd.DataFrame(0.0, index=connectivity_matrix.index, columns=connectivity_matrix.columns)
    
    mf_imputer = ModifiedMissForest(
        distance_matrix=distance_matrix,
        connectivity=connectivity_zero,
        max_iter=10,
        n_estimators=100,
        random_state=42,
        distance_weighting_type='inverse',
        temporal_feature_columns=all_feature_cols
    )
    
    try:
        mf_imputer.fit(df_train_masked)
        print("Modified MissForest model (No Contributor Info) trained.")
        return mf_imputer
    except Exception as e:
        print(f"ERROR: Model training failed. Error: {e}")
        return None