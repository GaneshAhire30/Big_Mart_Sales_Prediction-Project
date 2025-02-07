# Big Mart Sales Prediction Project

![image](https://github.com/user-attachments/assets/27a33cc1-2a2d-49ec-8bf8-30a2bfa14c21)

This repository contains the code and resources for the **Big Mart Sales Prediction** project, which aims to predict the sales of items in various stores based on historical data and item/store features. This project demonstrates the application of machine learning techniques to solve regression problems.

## Introduction

Big Mart operates multiple stores across different regions and offers various products. The goal of this project is to build a machine learning model to predict the sales of products in each store. By leveraging features such as product type, weight, visibility, and store characteristics, we aim to improve sales forecasting and inventory management.

## Business Case

This section provides the business justification for the project and its value to Big Mart, highlighting how accurate sales forecasting can optimize inventory management, reduce costs, and enhance decision-making.

## Dataset Overview

The dataset used for this project consists of the following columns:

### Features

- **Item_Identifier**: Unique product ID (dropped in preprocessing).
- **Item_Weight**: Weight of the product.
- **Item_Fat_Content**: Whether the product is low-fat or regular.
- **Item_Visibility**: Percentage of total display area allocated to the product.
- **Item_Type**: Category of the product.
- **Item_MRP**: Maximum Retail Price (MRP) of the product.
- **Outlet_Identifier**: Unique store ID.
- **Outlet_Establishment_Year**: Year the store was established.
- **Outlet_Size**: Size of the store (Small, Medium, Large).
- **Outlet_Location_Type**: Location type of the store (e.g., Tier 1, Tier 2).
- **Outlet_Type**: Type of the outlet (e.g., Grocery Store, Supermarket).

### Target Variable

- **Item_Outlet_Sales**: Sales of the product in the particular store (to be predicted).

## Project Workflow

### Data Preprocessing

* Missing Value Treatment:
    * Item_Weight: Imputed missing values using the mean weight per product category.
    * Outlet_Size: Filled missing entries based on the mode of Outlet_Type.
* Outlier Detection and Handling:
    * Identified and capped outliers in Item_Visibility using the Interquartile Range (IQR) method.
* Categorical Encoding:
    * Standardized categories in Item_Fat_Content to ensure consistency.
    * Applied label encoding and one-hot encoding to transform categorical variables into numerical formats.
* Feature Scaling:
    * Standardized continuous variables to ensure uniformity across features.

### Feature Engineering

* New Feature Creation:
    * Outlet_Age: Calculated as the difference between the current year and Outlet_Establishment_Year.
* Transformation:
    * Applied log transformation to Item_Outlet_Sales to normalize its distribution.

### Model Training and Evaluation

Implemented and evaluated multiple regression models:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor
* LightGBM Regressor
* CatBoost Regressor
* Support Vector Regressor (SVR)
* Artificial Neural Networks (ANNs)
### Ensemble Techniques:

* Stacking Regressor: Combined multiple models to leverage their individual strengths.
* Voting Regressor: Aggregated predictions from various models to enhance accuracy.
### Hyperparameter Tuning:

* Employed RandomizedSearchCV for optimizing model parameters.
### Evaluation Metrics:

* R² Score: Indicates the proportion of variance explained by the model.
* Root Mean Squared Error (RMSE): Measures the average magnitude of the prediction errors.
### Prediction and Submission

- Applied the trained model to predict sales on the test dataset.
- Saved the predictions to a CSV file for submission.

## Key Files and Folders

- **train.csv**: Training dataset.
- **test.csv**: Test dataset (used for final predictions).
- **best_xgb_rf_model.pkl**: Saved trained model using XGBRF.
- **submission_1.csv**: File containing final predictions for the test dataset.

## Evaluation Metrics

The model was evaluated using the following metrics:

- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors in predictions.
- **Root Mean Squared Error (RMSE)**: Penalizes larger errors more than MAE.
- **R² Score**: Indicates the proportion of variance explained by the model (closer to 1 is better).

### Final Results

- **MAE**: 0.4
- **RMSE**: 0.52
- **R² Score**: 0.739

## Technologies Used

* Programming Language: Python
* Libraries:
    * Data Manipulation: pandas, numpy
    * Visualization: matplotlib, seaborn
    * Machine Learning: scikit-learn, xgboost, lightgbm, catboost
    * Deep Learning: tensorflow, keras
    * Model Persistence: pickle
* Development Environment: Jupyter Notebook
## Results
The ensemble models, particularly the Stacking Regressor, demonstrated superior performance:

* Stacking Regressor:

    * Train R²: 77.51%
    * Test R²: 74.20%
    * Test RMSE: 0.521
* Voting Regressor:

    * Test R²: 74.00%
    * Test RMSE: 0.523
These results underscore the efficacy of ensemble techniques in capturing complex patterns within the data.

# Conclusion
The Big Mart Sales Prediction project successfully developed predictive models to forecast product sales across various outlets. Through meticulous data preprocessing, feature engineering, and the application of advanced modeling techniques, the project achieved robust predictive performance. The insights derived can assist Big Mart in optimizing inventory management and formulating data-driven business strategies.

# Acknowledgements
We extend our gratitude to the data science community for their invaluable resources and support, which significantly contributed to the success of this project.


Sources
