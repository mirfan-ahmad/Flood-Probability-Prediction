# Flood Probability Prediction

This project aims to predict the probability of floods using machine learning techniques. The workflow involves data preprocessing, feature engineering, and model training using LightGBM and Linear Regression models. The notebook uses a structured approach to analyze and train models on the given dataset.

## Overview

This project is organized into the following key steps:

1. **Data Preprocessing**:
   - Data loading and initial exploration.
   - Handling missing values and outliers.
   - Data transformation, including scaling and applying Yeo-Johnson transformation.

2. **Feature Engineering**:
   - Correlation matrix to visualize relationships between features.
   - PCA (Principal Component Analysis) to reduce dimensionality and handle feature redundancy.

3. **Model Building**:
   - Implementing LightGBM for regression and comparing results with Linear Regression.
   - Evaluating model performance using metrics such as Mean Squared Error (MSE) and R² score.

4. **Model Evaluation**:
   - Visualization of MSE over different numbers of features.
   - Performance comparison between models to select the best one for flood probability prediction.

## Dependencies

- Python 3.7+
- LightGBM
- Pandas
- Scikit-learn
- Matplotlib
- NumPy

## Instructions for Running

1. Clone this repository.
2. Run the notebook `flood-probability-prediction.ipynb` step by step to preprocess the data, build the models, and evaluate the performance.

## Key Insights

- **Preprocessing**: The Yeo-Johnson transformation was applied to stabilize variance and make the data more Gaussian-like, improving model performance.
- **Feature Engineering**: PCA reduced dimensionality while maintaining the most important features.
- **Model Performance**: 
   - LightGBM achieved better results in terms of MSE and R² compared to Linear Regression.
   - Visualizations showed how the number of features impacts MSE, with an optimal set minimizing error.

## Conclusion

The notebook effectively predicts flood probability using different models, with LightGBM outperforming Linear Regression. Feature engineering and careful preprocessing helped improve the model's predictive power.
