# Streamflow Data Imputation Application

This application provides a user interface (UI) for imputing missing streamflow data using a spatio-temporal model. It supports uploading discharge data, latitude/longitude data, and optional hydrological contributor information. The core imputation logic is based on the `burst_pipeline.py` script, which now includes model caching to avoid redundant retraining.

## Features

*   **File Uploads**: Upload required discharge and latitude/longitude data, and optional contributor data.
*   **Dynamic Model Selection**: Automatically uses a full spatio-temporal model if contributor data is provided, otherwise falls back to a model without contributor information.
*   **Model Caching**: Saves and loads trained models to disk, significantly speeding up repeated imputations with the same input parameters.
*   **Configurable Parameters**: Adjust parameters such as initial training window, imputation chunk size, overall year range, and minimum completeness for training.

## Setup and Run Instructions

Follow these steps to set up and run the application.

### 1. Clone the Repository (if not already done)

```bash
git clone <repository_url>
cd discharge
```

### 2. Create and Activate a Python Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install the required Python packages. You might need to create a `requirements.txt` file first if one doesn't exist. Assuming `pandas`, `numpy`, `gradio`, `geopy`, `scikit-learn` (for `MissForest`), `scipy` (if used by `ModifiedMissForest`), `matplotlib` are needed:

```bash
pip install pandas numpy gradio geopy scikit-learn matplotlib
```

*Note*: If `MissForest` or `ModifiedMissForest` is a custom implementation, ensure all its dependencies are met.

### 4. Prepare Your Data Files

Ensure you have your `.csv` data files ready:

*   **`discharge_data_cleaned.csv`**: Your main streamflow discharge data.
*   **`lat_long_discharge.csv`**: Latitude and longitude information for your monitoring stations.
*   **`mahanadi_contribs.csv` (Optional)**: Hydrological contributor information. Provide this if you want to use the full spatio-temporal model with connectivity.

Place these files in the same directory as the application files or specify their paths when prompted by the UI.

### 5. Run the Gradio Application

Execute the `frontend.py` script to start the Gradio web interface:

```bash
python3 frontend.py
```

Once the application starts, it will provide a local URL (e.g., `http://127.0.0.1:7860`). Open this URL in your web browser.

### 6. Using the UI

1.  **Upload Files**: Use the file upload components to provide your Discharge Data, Latitude/Longitude Data, and optionally, Hydrological Contributor Data.
2.  **Configure Parameters**: Adjust the imputation parameters using the sliders and input fields.
3.  **Run Imputation**: Click the "Run Imputation" button. The application will process your data. This might take some time for large datasets or the first run (due to model training).
4.  **Download Results**: Once complete, a download link for the imputed data (as a CSV file) will appear. You can also see status messages in the "Status" textbox.

### 7. Model Caching

*   Trained models are automatically saved to a directory named `trained_models_cache/`.
*   If you run the imputation again with the same input files and parameters, the application will load the pre-trained model from the cache instead of retraining, significantly reducing processing time.
*   To force retraining, you can delete the relevant `.pkl` files from the `trained_models_cache/` directory.

## Project Structure

*   `frontend.py`: The Gradio web application interface.
*   `burst_pipeline.py`: Contains the core spatio-temporal imputation logic, including model caching.
*   `utils.py`: Utility functions for data preprocessing, feature engineering, and metric evaluation.
*   `model_configurations.py`: Defines different model training configurations (e.g., full model, no contributor model).
*   `missforest_imputer.py`: Implementation of the Modified MissForest imputer.
*   `eval.py`: Script for evaluating model performance.
*   `README.md`: This file.
*   `trained_models_cache/`: Directory where trained models are saved/loaded.
*   `bursting_imputed_results/`: Directory for output data from the pipeline (if `output_dir` is set in `run_rolling_imputation_pipeline`).

---

For any issues or questions, please refer to the project maintainers.
