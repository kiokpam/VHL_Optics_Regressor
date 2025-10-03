# VHL Optics – PPM Prediction (Global Model)

**VHL Optics** is an AI project aimed at predicting the concentration (ppm) of chemical samples based on images captured from various mobile devices. In the initial version, each device model was trained with a separate model; however, the current version has been **refactored** to use **a single shared model** for all devices. Phone information is no longer a training feature, simplifying the process and making it easy to scale to new devices.

### Key Features

- **Automated data processing**: scans raw data, creates metadata files, and crops regions of interest (ROI) to standardize images.
- **Regression feature extraction**: calculates color channel statistics, GLCM contrast, entropy, edge density, etc. for each image.
- **Single regression model**: uses RandomForest or XGBoost to learn from all data, without requiring phone column.
- **Intuitive interface**: supports Streamlit for users to upload images and receive ppm prediction results instantly.
- **Batch prediction**: predicts in batches from aggregated feature files and saves results to CSV.

### Directory Structure

```
├── app.py           # Simple Streamlit application for single and batch predictions
├── main.py          # Entry point: e2e pipeline, feature extraction, training, Streamlit
├── config.py        # Defines data paths and constants
├── loading.py       # Creates and loads metadata
├── processing.py    # Image processing, ROI extraction and square images
├── normalize.py     # Regression feature extraction (does not contain phone)
├── model.py         # Trains regression model (per-phone & global)
├── predict.py       # Predicts ppm (single & batch)
├── roi.py           # ROI cropping support functions
├── squares.py       # Square contour detection
├── data/            # (auto-created) contains raw data and processed images
```

### Installation

1. **Clone the project** and install required libraries:

```bash
git clone <repository-url>
cd VHL_Optics_Regressor
pip install -r requirements.txt
```

If `requirements.txt` is not available, install minimum required libraries:

```
numpy
pandas
opencv-python
scikit-learn
xgboost
streamlit
scikit-image
tqdm
```

2. **Prepare data**: place original image data in `data/full/HP5_data` following the structure: *chemical type / phone / capture session / images*. The application will automatically scan and create metadata.

### Usage

#### Run the complete pipeline (e2e)

The following command will create metadata, process images, extract features, and train the shared model:

```bash
python main.py e2e
```

After running, feature files will be saved in `data/csv/features_all.csv` and trained models will be saved in `data/models/`.

#### Extract features and train separately

You can run individual steps separately:

```bash
# Extract features (when square images are available)
python main.py feature

# Train shared model (when features_all.csv is available)
python main.py train
```

#### Predict ppm for a single image

Use the simple Streamlit interface:

```bash
streamlit run app.py
```

Or run directly `python main.py` (default will open Streamlit interface). Users simply upload an image, select a model (**RF** or **XGB**), and receive ppm results. No need to input device model.

You can also call the prediction function from Python code:

```python
from predict import predict_regression_general

ppm = predict_regression_general('path/to/image.jpg', model_choice='RF')
print(f"Predicted ppm: {ppm:.2f}")
```

#### Batch prediction

To predict in batches, run from Streamlit (`app.py`) or use the function:

```python
from predict import predict_test_set_general

results = predict_test_set_general(output_dir='batch_predictions')
print(results.head())
```

Results are saved to `predictions_general.csv` with columns: `id_img`, `true_ppm`, `pred_rf_ppm`, `pred_xgb_ppm`, `diff_rf_pct`, `diff_xgb_pct`.

### Notes on Phone Information

The current version **does not use** any phone model information as a training feature. Previous versions trained separately for each device, but this caused difficulties when encountering new devices. The shared model is simpler and applicable to all images.

### System Requirements

* Python 3.8 or higher.
* Sufficient memory to process images and train models (GPU environment recommended for XGBoost if dataset is large).

### License

This project is distributed under the MIT license. You are free to use and modify the source code.

```
This README has been rewritten to reflect the shared model that does not require phone data and focuses solely on ppm concentration regression.
```
