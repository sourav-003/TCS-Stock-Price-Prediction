# TCS Stock Price Prediction Using Time Series Models (Prophet, ARIMA, LSTM)

## Overview
This project aims to predict the future stock prices of **Tata Consultancy Services (TCS)** using historical stock data. The project applies multiple time series forecasting techniques, including **Prophet**, **ARIMA**, and **LSTM** (Long Short-Term Memory), to predict stock prices based on patterns observed in past data. The goal is to help investors and analysts make informed decisions based on future stock predictions.

## Objective
The main objective of this project is to develop accurate forecasting models to predict **TCS stock prices**. By leveraging historical data, this project explores different time series forecasting techniques and evaluates their performance in stock price prediction.

## Technologies Used
- **Python Libraries**:
  - `pandas`: For data manipulation and analysis.
  - `numpy`: For numerical operations.
  - `matplotlib` and `seaborn`: For visualizing data and results.
  - `datetime`: To work with date and time formats.
  - `Prophet`: For forecasting time series data with trends and seasonality.
  - `ARIMA`: For modeling time series data with autoregression, moving averages, and differencing.
  - `LSTM`: For building deep learning models to capture complex patterns in time series data.

## Time Series Models Applied
1. **Prophet**:
   - Developed by Facebook, Prophet is a forecasting tool designed to handle daily observations that display trends and seasonality. In this project, Prophet was used to capture the seasonal patterns in TCS's stock prices and predict future prices.

2. **ARIMA (AutoRegressive Integrated Moving Average)**:
   - ARIMA is a classical statistical model that is widely used for time series forecasting. ARIMA combines autoregression, differencing, and moving averages to model time series data. It was applied to capture the short-term trends in TCS stock prices.

3. **LSTM (Long Short-Term Memory)**:
   - LSTM is a type of recurrent neural network (RNN) designed for sequential data. It is particularly useful for time series forecasting as it captures long-term dependencies in the data. This model was used to predict future stock prices by learning patterns over time.

## Dataset
The dataset used in this project contains historical stock price data for TCS, including:
- `Date`: The date of the stock price.
- `Open`: The stock price at market open.
- `High`: The highest price during the day.
- `Low`: The lowest price during the day.
- `Close`: The stock price at market close.
- `Adj Close`: The adjusted closing price accounting for splits and dividends.
- `Volume`: The number of shares traded on the day.

The data is available in CSV format and is loaded into a **pandas DataFrame** for processing.

## Key Steps in the Project

### 1. Data Loading & Preprocessing
- The historical stock data was loaded into a **pandas DataFrame** from a CSV file.
- The `Date` column was converted to **datetime** format to allow for time-based operations.
- Missing values were checked and handled appropriately.
- Basic exploratory data analysis (EDA) was conducted to understand stock price patterns.

### 2. Exploratory Data Analysis (EDA)
- The dataset was visualized using **line plots** to examine the closing prices over time.
- Moving averages (30-day and 100-day) were calculated and plotted to highlight stock price trends.
- Descriptive statistics were computed to understand the basic characteristics of the stock data.

### 3. Feature Engineering
- **Moving Averages**: Calculated 30-day and 100-day moving averages to smooth out short-term fluctuations and emphasize longer-term trends.
- These moving averages were used as features to enhance the performance of the forecasting models.

### 4. Time Series Forecasting
- **Prophet**: Applied to capture trends, seasonality, and holidays in stock price data. Prophet is particularly useful for handling data with missing points or large fluctuations.
- **ARIMA**: Modeled the temporal dependencies of stock prices. The ARIMA model was tuned for optimal results, capturing the autoregressive, moving average, and differencing components.
- **LSTM**: A deep learning model used to capture complex, long-term dependencies in the data. The LSTM model was trained on past stock prices to predict future values.

### 5. Model Evaluation
- **Performance Metrics**: Each model's performance was evaluated using **RMSE** (Root Mean Squared Error) and **MAE** (Mean Absolute Error) to compare their prediction accuracy.
- Visualizations were created to compare the predicted stock prices with the actual values, and the results were analyzed to understand the effectiveness of each model.

## Results and Insights
- The **Prophet** model effectively captured long-term trends and seasonal patterns in the stock price data.
- The **ARIMA** model worked well for capturing short-term fluctuations and providing predictions based on past values.
- The **LSTM** model demonstrated the ability to learn complex temporal dependencies in the data, offering a powerful tool for stock price prediction.

## Conclusion
This project demonstrates the application of multiple time series forecasting techniques to predict TCS stock prices. Each model provided unique insights into the stock's behavior:
- **Prophet** excelled in capturing seasonal patterns and long-term trends.
- **ARIMA** was effective for modeling short-term stock price movements.
- **LSTM** proved valuable in learning complex, non-linear patterns in the data.

The models offer useful predictions that can assist traders and investors in making informed decisions. Further improvements can be made by fine-tuning the models, incorporating more features, and testing with other stock data.

## Explore the Project
- [GitHub Repository](https://github.com/sourav-003/TCS-Stock-Price-Prediction)
- [Web Application on Hugging Face](https://huggingface.co/spaces/Sourav-003/tcs-stock-forecast)

## Acknowledgements
Special thanks to **Kumar Sundram Sir** and **LearnBay** for their mentorship and guidance throughout this project. Their insights were crucial in successfully implementing and evaluating these forecasting models.

---

Feel free to copy and paste this `README.md` directly into your GitHub repository! Let me know if you need any adjustments.
