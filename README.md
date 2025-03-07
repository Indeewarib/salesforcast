# Route Sales Forecasting Algorithm
This project implements time series forecasting algorithms to predict the next week's expected orders per route for Arcese. The two models used for forecasting are ARIMA (AutoRegressive Integrated Moving Average) and ETS (Exponential Smoothing), both of which account for seasonality, trends, and noise. The performance of these models is evaluated using metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

### Project Overview
The dataset contains historical sales data, and the task is to create forecasting models that predict future orders based on past trends. This project utilizes ARIMA and ETS models to handle the time series data and forecasts future order volumes per route.

### Dataset
The input dataset, ``` Input_data_timeseries.xlsx ```, contains the following columns:

- ORDER DATE: The date of the order.
- ROUTE: The route for which the order was placed.
- NUMBER OF ACTUAL ORDERS: The number of orders for that date and route.
The data is pre-processed and aggregated by week to form a weekly sales dataset.

### How it Works
- The algorithm splits the data by route and aggregates it weekly.
- The training set is used to train the model, while the validation and test sets are used to evaluate its performance.
- A forecast for the next week is generated, and the accuracy of the model is measured.
- The forecast and actual values are plotted for visual analysis.

### Usage
- Place the ``` Input_data_timeseries.xlsx ``` file in the project directory.
- Run the Streamlit app:
```streamlit run .\arcese_arima.py``` or ```streamlit run .\arcese_EST.py```
