# NFL Stats Projector

## Overview
The **NFL Stats Projector** is a Python script designed to fetch NFL player statistics and train/evaluate an LSTM model for fantasy football projections. It supports multiple offensive positions—Quarterbacks, Running Backs, Wide Receivers, and Tight Ends—and visualizes model performance through validation loss, actual vs. predicted values, and key statistical measures such as Mean Squared Error (MSE) and Mean Absolute Error (MAE).

## Key Features & Requirements
- **Data Preparation and Scaling**: Cleans and scales player stats, preparing the data for LSTM model training and evaluation.
- **LSTM Model Training and Evaluation**: Trains an LSTM model to predict fantasy football points and evaluates its performance with metrics such as MSE, MAE, RMSE, and R².
- **Visualization Generation**: Creates interactive plots to visualize model performance, including loss curves and actual vs. predicted fantasy points.
- **Next Game Fantasy Projections**: Provides projected fantasy points for the next game based on the most recent player performance.
- **Error Logging**: Implements logging to track and report any issues encountered during data processing or model training.

### Requirements
- Refer to [requirements.txt](../requirements.txt) for the required packages and libraries.
