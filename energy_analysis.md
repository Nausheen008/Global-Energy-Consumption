# World Energy Consumption Analysis - Jupyter Notebook

## Overview
This notebook performs a comprehensive analysis of world energy consumption data, implementing advanced time series forecasting techniques, outlier detection, missing value imputation, and feature engineering.

## Requirements
Install the required packages:
```bash
pip install pandas numpy scikit-learn lightgbm statsmodels prophet ydata-profiling shap matplotlib seaborn pyod plotly kaleido
```

## Files Generated
- `data_raw.csv` - Original raw dataset backup
- `data_dictionary.md` - Data dictionary with imputation strategies
- `test_predictions.csv` - Final predictions on test set
- `artifact_preprocessor.pkl` - Saved preprocessing pipeline
- `model_lightgbm.pkl` - Saved LightGBM model
- `profile_report.html` - Data profiling report
- Various plots and analysis outputs

## How to Run
1. Ensure you have `World Energy Consumption.csv` in the same directory
2. Install required packages
3. Run all cells in order
4. Check the executive summary at the end for key insights

## Key Features
- Robust outlier detection using multiple methods (Z-score, IQR, LOF)
- Advanced imputation strategies (STL decomposition, MICE, model-based)
- Comprehensive feature engineering for time series
- Multiple baseline and advanced models
- SHAP explainability analysis
- Walk-forward cross-validation
- Prediction intervals with calibration analysis

## What to Inspect First
1. **Executive Summary** (final cell) - Key findings and recommendations
2. **Data Quality Report** - Missing data patterns and outliers
3. **Model Performance Comparison** - Best performing models
4. **SHAP Analysis** - Feature importance and interactions
5. **Calibration Plots** - Prediction interval reliability