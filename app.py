
import gradio as gr
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import numpy as np

# Load the saved Prophet model (trained on TCS stock data)
# Make sure the model file 'prophet_model.joblib' is in the same directory as app.py
try:
    model = joblib.load('prophet_model.joblib')
except FileNotFoundError:
    # This is a placeholder for handling the case where the model file is not found
    # In a real scenario, you would need to train the model or ensure the file is present
    model = None
    print("Error: 'prophet_model.joblib' not found. Please ensure the model file is in the same directory.")

# Load the test data for evaluation metrics
# Assuming 'test' DataFrame is available globally or can be loaded
# In a real app, you might load this from a file or skip test set evaluation in the deployed app
try:
    # This assumes you have saved your test data to a file
    # Replace 'test_data.csv' with the actual filename if you saved it
    # For this example, we'll assume 'test' is not strictly necessary for the forecast plot itself
    # but we'll keep the evaluation logic for demonstration if test data is available.
    # You might need to adjust this part based on how you handle test data.
    test_data_available = False # Set to True if you have saved test data
    if test_data_available:
         test = pd.read_csv('test_data.csv') # Load test data if available
         test['Date'] = pd.to_datetime(test['Date']) # Convert Date column to datetime
         test = test.set_index('Date') # Set Date as index
    else:
        # Create a dummy test DataFrame if not available, to avoid errors in evaluation logic
        test = pd.DataFrame(columns=['Open'])


except FileNotFoundError:
    print("Warning: 'test_data.csv' not found. Evaluation metrics on test set will not be available.")
    test = pd.DataFrame(columns=['Open']) # Create an empty DataFrame if test data file is not found


def forecast_and_plot(days_to_forecast):
    if model is None:
        return None, None, None, "⚠️ Model not loaded. Please check server logs.", "Model not loaded."

    try:
        # Create future dataframe
        future = model.make_future_dataframe(periods=days_to_forecast, freq='D')

        # Generate predictions
        forecast = model.predict(future)

        # --- Calculate Evaluation Metrics ---
        # This part depends on whether 'test' data was successfully loaded
        evaluation_metrics_str = "Evaluation metrics on test set are not available (test data not loaded)."
        if not test.empty:
            evaluation_forecast = forecast[forecast['ds'].isin(test.index)]
            evaluation_df = test.merge(
                evaluation_forecast[['ds', 'yhat']],
                left_index=True, right_on='ds', how='left'
            )

            if not evaluation_df.empty:
                mae = mean_absolute_error(evaluation_df['Open'], evaluation_df['yhat'])
                mse = mean_squared_error(evaluation_df['Open'], evaluation_df['yhat'])
                rmse = sqrt(mse)
                mape = (
                    np.mean(np.abs((evaluation_df['Open'] - evaluation_df['yhat']) / evaluation_df['Open'])) * 100
                    if np.any(evaluation_df['Open'] != 0) else float('inf')
                )

                evaluation_metrics_str = (
                    f"📊 **Evaluation Metrics on Test Set:**\n\n"
                    f"🔹 MAE: {mae:.4f}\n"
                    f"🔹 MSE: {mse:.4f}\n"
                    f"🔹 RMSE: {rmse:.4f}\n"
                    f"🔹 MAPE: {mape:.2f}%"
                )
            else:
                evaluation_metrics_str = "⚠️ No overlapping dates between forecast and test set for evaluation."


        # --- Forecast Summary ---
        last_actual = test['Open'].iloc[-1] if not test.empty else None
        first_pred = forecast['yhat'].iloc[-days_to_forecast] if not forecast.empty and days_to_forecast > 0 else None
        last_pred = forecast['yhat'].iloc[-1] if not forecast.empty else None


        summary_str = ""
        if last_actual is not None:
            summary_str += f"📌 Last Actual TCS Open Price: {last_actual:.2f}\n"
        if first_pred is not None:
             summary_str += f"📌 First Predicted TCS Open Price: {first_pred:.2f}\n"
        if last_pred is not None:
             summary_str += f"📌 Last Predicted TCS Open Price: {last_pred:.2f}"

        if not summary_str:
             summary_str = "No data available for summary."


        # --- Generate Plots ---
        # Ensure plots are generated without displaying them directly
        plt.ioff() # Turn off interactive plotting
        forecast_fig = model.plot(forecast)
        plt.title("📈 TCS Stock Price Forecast (Open Price)")
        plt.xlabel("Date")
        plt.ylabel("Open Price (INR)")
        plt.grid(True)

        components_fig = model.plot_components(forecast)

        # --- Rename columns for user-friendly display ---
        forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
            columns={
                'ds': 'Date',
                'yhat': 'Predicted TCS Open Price',
                'yhat_lower': 'Lower Confidence',
                'yhat_upper': 'Upper Confidence'
            }
        )

        # Convert the forecast DataFrame to a displayable format for Gradio
        # For large DataFrames, converting to HTML or a string might be better
        # For this example, we'll return the DataFrame directly as Gradio's DataFrame component can handle it.


        return (
            forecast_display,
            forecast_fig,
            components_fig,
            evaluation_metrics_str,
            summary_str
        )

    except Exception as e:
        return None, None, None, f"⚠️ Error during forecasting: {str(e)}", "Error in forecast."


# Gradio Interface
interface = gr.Interface(
    fn=forecast_and_plot,
    inputs=gr.Slider(minimum=1, maximum=365, step=1, value=30, label="Number of days to forecast"),
    outputs=[
        gr.DataFrame(label="📊 Forecasted TCS Prices"),
        gr.Plot(label="📈 Forecast Plot"),
        gr.Plot(label="🔍 Forecast Components Plot"),
        gr.Textbox(label="📑 Evaluation Metrics"),
        gr.Textbox(label="📌 Forecast Summary")
    ],
    title="📈 TCS Stock Price Forecasting with Prophet",
    description=(
        "This app forecasts the **TCS stock Opening Price** using Facebook Prophet.<br>"
        "Enter the number of days to forecast and view predictions, plots, and evaluation metrics."
        "⚠️ **Note:** The prices shown are based on *adjusted values* from Yahoo Finance "
        "(adjusted for splits and dividends). These values may appear lower than the "
        "actual traded market price (₹3000+), but trends and percentage changes remain accurate."
    ),
    live=False # Set to False for potentially long running tasks like forecasting
)

# To run the Gradio app, you would typically save this code as app.py and run it.
# For deployment on Hugging Face, you'll need this app.py file, the model file, and requirements.txt

# interface.launch() # Uncomment this line to test the Gradio app locally
