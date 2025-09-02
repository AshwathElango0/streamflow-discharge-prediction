# gradio_app.py

import gradio as gr
import pandas as pd
import tempfile
import warnings

# Import functions from the user's provided files
from burst_pipeline import run_rolling_imputation_pipeline
# from utils import load_and_preprocess_data, add_temporal_features, build_distance_matrix, build_connectivity_matrix, find_best_training_period
# from missforest_imputer import ModifiedMissForest
# from model_configurations import train_full_model, train_no_contributor_model

# We only need to import the top-level function that runs the whole pipeline
# The run_rolling_imputation_pipeline function in burst_pipeline.py handles all the sub-calls.
# This assumes the burst_pipeline.py is self-contained in its execution flow.

warnings.filterwarnings('ignore')

def gradio_pipeline_wrapper(discharge_file, lat_long_file, contrib_file, initial_train_window_size, imputation_chunk_size_years, overall_min_year, overall_max_year, min_completeness_percent_train):
    """
    Wrapper function to handle file paths from Gradio and run the pipeline.
    
    Args:
        discharge_file (gr.File): The uploaded discharge data file.
        lat_long_file (gr.File): The uploaded latitude/longitude data file.
        contrib_file (gr.File): The uploaded contributor data file (can be None).
        initial_train_window_size (int): Size of the initial training window in years.
        imputation_chunk_size_years (int): Size of the imputation chunk in years.
        overall_min_year (int): The minimum year for the overall data range.
        overall_max_year (int): The maximum year for the overall data range.
        min_completeness_percent_train (float): Minimum completeness for the training period.

    Returns:
        tuple: A status message and the path to the output file for download.
    """
    # Gradio provides file objects; we need their paths.
    discharge_path = discharge_file.name if discharge_file else None
    lat_long_path = lat_long_file.name if lat_long_file else None
    contrib_path = contrib_file.name if contrib_file else None
    
    if not discharge_path or not lat_long_path:
        return "Error: Discharge and Latitude/Longitude files are mandatory.", None
        
    try:
        # Call the main pipeline function from burst_pipeline.py
        final_imputed_data = run_rolling_imputation_pipeline(
            discharge_path=discharge_path,
            lat_long_path=lat_long_path,
            contrib_path=contrib_path,
            initial_train_window_size=int(initial_train_window_size),
            imputation_chunk_size_years=int(imputation_chunk_size_years),
            overall_min_year=int(overall_min_year),
            overall_max_year=int(overall_max_year),
            min_completeness_percent_train=float(min_completeness_percent_train)
        )
        
        if final_imputed_data.empty:
            return "Error: The pipeline returned an empty dataset. Please check your input files and parameters.", None
            
        # Save the result to a temporary file
        output_file_path = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
        # Reset the index to save the 'Date' column properly
        final_imputed_data.to_csv(output_file_path, index=True, date_format='%Y-%m-%d')
        
        return "Imputation complete. You can download the file below.", output_file_path
        
    except Exception as e:
        # Handle any errors that occur during the pipeline execution
        return f"An error occurred: {e}", None

# Define the Gradio interface
with gr.Blocks(title="Streamflow Data Imputation") as demo:
    gr.Markdown(
        """
        # Streamflow Data Imputation
        Upload your data files and configure the imputation parameters to generate a complete streamflow dataset.
        
        **Note:** This application assumes that `burst_pipeline.py`, `missforest_imputer.py`, `model_configurations.py`, and `utils.py` are in the same directory.
        """
    )
    with gr.Row():
        with gr.Column():
            discharge_input = gr.File(label="1. Discharge Data (Required)", file_types=[".csv"])
            lat_long_input = gr.File(label="2. Latitude/Longitude Data (Required)", file_types=[".csv"])
            contrib_input = gr.File(label="3. Hydrological Contributor Data (Optional)", file_types=[".csv"])
            
            gr.Markdown("---")
            gr.Markdown("### Imputation Parameters")
            
            initial_train_window = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Initial Training Window Size (Years)")
            imputation_chunk = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Imputation Chunk Size (Years)")
            min_year_slider = gr.Slider(minimum=1950, maximum=2020, value=1970, step=1, label="Overall Minimum Year")
            max_year_slider = gr.Slider(minimum=1950, maximum=2020, value=2010, step=1, label="Overall Maximum Year")
            min_completeness = gr.Slider(minimum=0, maximum=100, value=70, step=5, label="Min Completeness for Training (%)")

            run_button = gr.Button("Run Imputation", variant="primary")

        with gr.Column():
            status_text = gr.Textbox(label="Status", interactive=False)
            output_file = gr.File(label="Imputed Data Output", file_types=[".csv"])
            
    run_button.click(
        fn=gradio_pipeline_wrapper,
        inputs=[
            discharge_input,
            lat_long_input,
            contrib_input,
            initial_train_window,
            imputation_chunk,
            min_year_slider,
            max_year_slider,
            min_completeness
        ],
        outputs=[status_text, output_file]
    )

if __name__ == "__main__":
    demo.launch()
