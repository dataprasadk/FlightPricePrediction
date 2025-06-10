# Flight Fare Prediction Project

This repository contains the implementation of a Machine Learning project aimed at predicting flight ticket prices. The project leverages structured data transformation, cyclic encoding, and various machine learning algorithms to achieve accurate fare predictions.

**Project done by**: Prasad Kamble

## Table of Contents
- [Project Objective](#project-objective)
- [Domain Background](#domain-background)
- [Dataset Description](#dataset-description)
- [Project Flow](#project-flow)
- [Exploratory Data Analysis (EDA) Insights](#exploratory-data-analysis-eda-insights)
- [Model Evaluation Framework](#model-evaluation-framework)
- [Model Comparison Report](#model-comparison-report)
- [Conclusion](#conclusion)
- [Getting Started](#getting-started)
- [Files](#files)

## Project Objective
The primary objective of this project is to predict the price of airline tickets based on various input features such as airline, date/time of journey, source/destination, duration, and number of stops. This is approached as a regression problem, with `Price` being the continuous target variable.

## Domain Background
The airline industry is highly competitive and price-sensitive. Airfare prices are influenced by factors like booking time, seasonal demand, source/destination popularity, airline service levels, and the number of stops. Accurate fare prediction benefits consumers by helping them find better deals and aids platforms (like travel agencies) in optimizing pricing strategies.

## Dataset Description
The dataset consists of anonymized records of flight bookings with the following features:
* **Airline**: Name of the airline.
* **Date_of_Journey**: Journey date.
* **Source**: Departure city.
* **Destination**: Arrival city.
* **Route**: Route followed by the flight.
* **Dep_Time**: Flight departure time.
* **Arrival_Time**: Flight arrival time.
* **Duration**: Total time of the flight.
* **Total_Stops**: Number of stops between source and destination.
* **Additional_Info**: Miscellaneous information.
* **Price**: Fare of the flight (target variable).

The dataset contains 10683 entries and 11 columns.

## Project Flow
The project followed a structured approach including:

1.  **Prechecks**: Initial data inspection for shape, data types, missing values, duplicates, unique values, and zero variance.
2.  **Exploratory Data Analysis (EDA)**: Univariate and bivariate analysis of key features to extract meaningful insights.
3.  **Feature Engineering**: Transformation of date/time columns into numerical components (hour, minute, month, day), normalization of 'Duration' into total minutes, and conversion of 'Total_Stops' to numerical format. Dropping irrelevant features like 'Route'.
4.  **Data Preprocessing**: Handling categorical variables through encoding (e.g., One-Hot Encoding or Label Encoding), and splitting data into training and test sets.
5.  **Model Building**: Implementation of various regression algorithms, including Linear Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, Extra Trees Regressor, Ridge Regression, K-Neighbors Regressor, SVR, Linear SVM, and XGBoost Regressor.
6.  **Hyperparameter Tuning**: Applied RandomizedSearchCV for tuning top models to optimize performance.
7.  **Model Evaluation**: Performance was assessed using R² score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
8.  **Final Model Saving and Inference**: Retraining the best model on the full dataset and saving it for future predictions.

## Exploratory Data Analysis (EDA) Insights
* **Airline Distribution**: Jet Airways is the most frequent airline, while others like Air Asia and Trujet appear less often. This imbalance might influence model predictions.
* **Source City Distribution**: Most flights originate from Delhi and Kolkata, with fewer from Chennai and Mumbai.
* **Destination City Distribution**: New Delhi and Bangalore are the most common destinations, showing symmetric travel trends between major cities.
* **Total Stops Distribution**: Non-stop and 1-stop flights dominate the dataset. Flights with more stops are rare, and fare generally increases with the number of stops.
* **Additional Info Category**: Over 75% of records have "No info", indicating sparse categorical richness in this feature. However, niche entries like "Business class" might be significant price drivers.

## Model Evaluation Framework
The project prioritized MAE and RMSE for understanding real-value error impact and used R² to assess explained variance. Both initial and tuned models were evaluated for all major algorithms, with RandomizedSearchCV employed for tuning top models.

## Model Comparison Report

The following table summarizes the performance of various models, including initial and tuned versions, sorted by R2 score:

| Model Name                      | R2 Score | MAE      | RMSE     | Model Type          |
| :------------------------------ | :------- | :------- | :------- | :------------------ |
| Extra Trees Regressor (Tuned)   | 0.9634   | 1134.42  | 1708.20  | Tree-Based (Tuned)  |
| Random Forest Regressor (Tuned) | 0.9602   | 1184.67  | 1780.08  | Tree-Based (Tuned)  |
| XGBoost Regressor (Tuned)       | 0.9578   | 1238.16  | 1832.74  | Tree-Based (Tuned)  |
| Extra Trees Regressor           | 0.9555   | 1183.07  | 1878.69  | Tree-Based (Initial)|
| Random Forest Regressor         | 0.9515   | 1259.04  | 1960.91  | Tree-Based (Initial)|
| Gradient Boosting Regressor     | 0.9080   | 1673.81  | 2665.41  | Tree-Based (Initial)|
| Decision Tree Regressor         | 0.8841   | 1689.70  | 2984.77  | Tree-Based (Initial)|
| XGBoost Regressor               | 0.9515   | 1259.04  | 1960.91  | Tree-Based (Initial)|
| K-Neighbors Regressor (Tuned)   | 0.6970   | 2496.06  | 4381.18  | Linear (Tuned)      |
| Linear SVM (LinearSVR)          | 0.6025   | 2646.01  | 5057.49  | Linear (Initial)    |
| Linear Regression               | 0.6010   | 2639.18  | 5078.61  | Linear (Initial)    |
| Ridge Regression                | 0.6010   | 2639.18  | 5078.61  | Linear (Initial)    |
| K-Neighbors Regressor           | 0.5898   | 2588.67  | 5146.59  | Linear (Initial)    |
| Elastic Net Regression          | 0.5367   | 2915.25  | 5472.93  | Linear (Initial)    |
| Lasso Regression                | 0.5367   | 2915.25  | 5472.93  | Linear (Initial)    |

**Best Performing Model**: Based on the R² score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE), the **Extra Trees Regressor (Tuned)** is chosen as the best model. It achieved an R² score of **0.9634**, with an MAE of **1134.42** and RMSE of **1708.20**, indicating high accuracy and low prediction error.

## Conclusion
This Machine Learning project successfully predicted flight ticket prices. Among all models evaluated, the **Extra Trees Regressor (Tuned)** consistently delivered the best balance of accuracy, interpretability, and low prediction error. Its performance is production-ready, with high R² and very low MAE/RMSE values. The developed pipeline and model are suitable for real-world airline pricing engines, fare predictors, and demand estimation systems. This project demonstrates the effectiveness of a well-rounded, feature-rich ML approach to regression problems in pricing domains.

## Getting Started
To run this project locally, you will need to set up a Python environment and install the required dependencies.

### Prerequisites
* Python 3.x
* Jupyter Notebook