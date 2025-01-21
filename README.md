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

- Handle missing values.
- Encode categorical variables.
- Scale and normalize features.
- Ensure consistency between training and test datasets.

### Feature Engineering

- Add derived features like **Outlet_Age**.
- Drop less significant features (e.g., **Item_Identifier**).

### Model Training and Evaluation

- Experimented with multiple regression models: **Gradient Boosting**, **LightGBM**, **XGBRF**.
- Used **RandomizedSearchCV** for hyperparameter optimization.
- Evaluated performance using metrics such as **R² Score**, **MAE**, and **RMSE**.

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

- **Python** (pandas, numpy, scikit-learn, xgboost, lightgbm)
- **Jupyter Notebook**
- **Matplotlib** and **Seaborn** (for visualizations)
- **Pickle** (for model persistence)

## Results
- The trained model achieved strong performance with an R² score of 0.739 on unseen test data. Predictions have been saved in submission_1.csv.
