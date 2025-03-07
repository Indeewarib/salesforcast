import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import os

# Load the data
full_path = os.path.realpath(__file__)
project_folder = os.path.dirname(full_path)

route_sale_df = pd.read_excel(f'{project_folder}\\Input_data_timeseries.xlsx', parse_dates=['ORDER DATE'])
route_sale_df.head()

# Process the data to create weekly sales data
df_weekly = route_sale_df.groupby([pd.Grouper(key='ORDER DATE', freq='W'), 'ROUTE'])['NUMBER OF ACTUAL ORDERS'].sum().reset_index()
df_weekly.rename(columns={'ORDER DATE': 'WEEK', 'NUMBER OF ACTUAL ORDERS': 'WEEKLY ORDERS'}, inplace=True)
df_weekly.set_index('WEEK', inplace=True)
df_weekly.head()

# Standardize the orders
scaler = StandardScaler()
df_weekly['SCALED_ORDERS'] = scaler.fit_transform(df_weekly[['WEEKLY ORDERS']])

# Function to split data into train, validation, and test sets
def split_data_by_route(df, route_col='ROUTE', target_col='SCALED_ORDERS', train_size=0.6, val_size=0.2):
    train_set = []
    val_set = []
    test_set = []

    routes = df[route_col].unique()

    for route in routes:
        route_data = df[df[route_col] == route]

        # Split data into train (60%), validation (20%), and test (20%)
        train, temp = train_test_split(route_data, test_size=1 - train_size, shuffle=False)
        val, test = train_test_split(temp, test_size=0.5, shuffle=False)

        train_set.append(train)
        val_set.append(val)
        test_set.append(test)

    # Combine back into single dataframes for easier handling
    train_df = pd.concat(train_set)
    val_df = pd.concat(val_set)
    test_df = pd.concat(test_set)

    return train_df, val_df, test_df

# Function to apply Triple Exponential Smoothing (Additive)
def fit_triple_exp_smoothing(train_data, target_col='SCALED_ORDERS', seasonal_periods=7):
    model = ETSModel(train_data[target_col], trend='add', seasonal='add', seasonal_periods=seasonal_periods)
    model_fit = model.fit()
    return model_fit

# Function to forecast and evaluate the model on validation and test sets
def evaluate_model(model_fit, val_data, test_data, target_col='SCALED_ORDERS'):
    # Forecasting for validation and test data
    val_forecast = model_fit.forecast(steps=len(val_data))
    test_forecast = model_fit.forecast(steps=len(test_data))

    # Evaluate performance on validation and test sets
    val_mae = mean_absolute_error(val_data[target_col], val_forecast)
    test_mae = mean_absolute_error(test_data[target_col], test_forecast)
    val_rmse = np.sqrt(mean_squared_error(val_data[target_col], val_forecast))
    test_rmse = np.sqrt(mean_squared_error(test_data[target_col], test_forecast))

    return val_forecast, test_forecast, val_mae, val_rmse, test_mae, test_rmse

# Function to forecast for a specific route and plot results
def forecast_for_route(route, df, target_col='SCALED_ORDERS'):
    # Filter the data for the selected route
    route_data = df[df['ROUTE'] == route]

    # Split data into train, validation, and test sets
    train_df, val_df, test_df = split_data_by_route(route_data)

    # Fit the Triple Exponential Smoothing model
    model_fit = fit_triple_exp_smoothing(train_df)

    # Evaluate model performance on validation and test sets
    val_forecast, test_forecast, val_mae, val_rmse, test_mae, test_rmse = evaluate_model(model_fit, val_df, test_df)

    # Forecast the next week's value (one step ahead)
    next_week_forecast = model_fit.forecast(steps=1)[0]  # Forecasting one period ahead (next week)
    
    # Convert the forecasted scaled value back to the original scale
    next_week_forecast_actual = scaler.inverse_transform([[next_week_forecast]])[0][0]

    # Get the date for the next week
    next_week_date = df.index[-1] + pd.Timedelta(weeks=1)
    
    # Plot the forecast vs actual for training, validation, and test sets
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_df.index, train_df[target_col], label='Training Data', color='blue')
    ax.plot(val_df.index, val_df[target_col], label='Actual (Validation)', color='green')
    ax.plot(val_df.index, val_forecast, label='Forecast (Validation)', color='red')
    ax.plot(test_df.index, test_df[target_col], label='Actual (Test)', color='orange')
    ax.plot(test_df.index, test_forecast, label='Forecast (Test)', color='purple')
    ax.set_xlabel('Date')
    ax.set_ylabel(target_col)
    ax.set_title(f"Forecast vs Actual for {route}")
    ax.legend()

    # Display plot in Streamlit
    st.pyplot(fig)

    # Display evaluation metrics and forecasted values
    st.write(f"Validation MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")
    st.write(f"Test MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")

    # Display forecasted values for validation and test sets
    #st.write("Forecasted values for Validation Set:")
    #val_forecast_df = pd.DataFrame({'Date': val_df.index, 'Forecasted Orders': val_forecast})
    #st.write(val_forecast_df)
    
    #st.write("Forecasted values for Test Set:")
    #test_forecast_df = pd.DataFrame({'Date': test_df.index, 'Forecasted Orders': test_forecast})
    #st.write(test_forecast_df)

    # Display the forecasted value for next week along with the corresponding week
    st.write(f"Forecasted Orders for Next Week (Week starting {next_week_date.strftime('%Y-%m-%d')}): {next_week_forecast_actual:.4f}")

# Streamlit UI for selecting route
st.title('Route Sales Forecasting')
route_options = df_weekly['ROUTE'].unique()
selected_route = st.selectbox('Select a route to forecast', route_options)

if selected_route:
    forecast_for_route(selected_route, df_weekly)

