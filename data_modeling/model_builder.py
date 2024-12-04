import os
import logging
import warnings
import pendulum
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from data_processing.stat_processor import NFLStatsProcessor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    mean_absolute_error,
)

warnings.filterwarnings("ignore")

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
processor_logger = logging.getLogger("NFL-Processor")
processor_logger.setLevel(logging.INFO)


class NFLStatsPredictor:
    """
    A class to fetch player stats, train a machine learning model, and plot actual vs predicted fantasy points.

    Attributes:
    - offensive_positions (list): Positions to process (currently only "QB").
    - current_year (int): The current year.
    - backfill_year (int): The year to start backfilling the stats from.
    """

    offensive_positions = ["QB"]#, "RB", "WR", "TE"]
    current_year = pendulum.now().year
    backfill_year = pendulum.now().subtract(years=(current_year - 2021)).year

    def __init__(self):
        """
        Initialize NFLStatsProcessor to process stats for offensive positions.
        """
        self.processor = NFLStatsProcessor()

    def get_player_stats(
        self, player_name: str, year: int = None, position: str = None
    ) -> pd.DataFrame:
        """
        Use NFLStatsProcessor to extract individual player's statistics based on player name, year, and position.

        Parameters:
        - player_name (str): Name of the player to filter for.
        - year (int, optional): Year of the statistics to filter for. Defaults to None (all years).
        - position (str, optional): Position of the player to filter for. Defaults to None (all positions).

        Returns:
        pd.DataFrame: A DataFrame containing the statistics of the specified player.
        """
        return self.processor.get_player_stats(player_name, year, position)

    def evaluate_lstm(self, player_name: str) -> None:
        """
        Train LSTM model to predict fantasy points based on player stats.

        Parameters:
        - player_name (str): The name of the player to train the model for.
        """
        player_stats = self.get_player_stats(player_name)
        X = player_stats.drop(
            columns=["MISC_FPTS", "RANK", "PLAYER", "GAMES", "POSITION"]
        )
        y = player_stats["MISC_FPTS"]
        X_scaler = StandardScaler()
        X_scaled = X_scaler.fit_transform(X)
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=0
        )

        model = Sequential()
        model.add(
            LSTM(
                128,
                activation="relu",
                return_sequences=True,
                input_shape=(X_scaled.shape[1], X_scaled.shape[2]),
            )
        )
        model.add(Dropout(0.3))
        model.add(LSTM(64, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(1))
        optimizer = Adam(learning_rate=0.0005)
        model.compile(optimizer=optimizer, loss="mean_squared_error")

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        history = model.fit(
            X_train,
            y_train,
            epochs=150,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1,
        )
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(1, len(loss) + 1)
        loss_data = pd.DataFrame({
            "Epoch": epochs,
            "Loss": loss,
            "Validation Loss": val_loss
        })
        loss_data_melted = loss_data.melt(
            id_vars="Epoch", 
            value_vars=["Loss", "Validation Loss"], 
            var_name="Type", 
            value_name="Value")
        fig_loss = px.line(
            loss_data_melted, 
            x="Epoch", 
            y="Value", 
            color="Type", 
            title=f"{player_name} - Loss Over Epochs",
            labels={
                "Value": "Loss", 
                "Epoch": "Epoch", 
                "Type": "Loss Type"
            }
        )
        self.processor.save_plot(
            fig_loss,
            player_name,
            "lstm_model_loss.html",
        )
        y_pred_scaled = model.predict(X_test)
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        y_test_actual = y_scaler.inverse_transform(y_test)
        prediction_df = pd.DataFrame({
            "Sample": range(len(y_test_actual.flatten())),
            "Actual": y_test_actual.flatten(),
            "Predicted": y_pred.flatten()
        })
        prediction_data_melted = prediction_df.melt(
            id_vars="Sample", 
            value_vars=["Actual", "Predicted"], 
            var_name="Type", 
            value_name="Value"
        )
        fig_pred = px.line( 
            prediction_data_melted, 
            x="Sample",
            y="Value", 
            color="Type", 
            title=f"{player_name} - Actual vs Predicted",
            labels={
                "Value": "Fantasy Points",  
                "Sample": "Sample"
            }
        )
        latest_input = X_scaled[-1].reshape(1, 1, X_scaled.shape[2])  
        next_scaled_prediction = model.predict(latest_input)
        next_prediction = y_scaler.inverse_transform(next_scaled_prediction)
        mse = mean_squared_error(prediction_df["Actual"], prediction_df["Predicted"])
        rmse = root_mean_squared_error(prediction_df["Actual"], prediction_df["Predicted"])
        mae = mean_absolute_error(prediction_df["Actual"], prediction_df["Predicted"])
        r2 = r2_score(prediction_df["Actual"], prediction_df["Predicted"])
        metrics_text = (
            f"Proj. FPTS: {next_prediction[0][0]:.2f}<br><br><sup>MSE: {mse:.4f}<br>RMSE: {rmse:.4f}<br>MAE: {mae:.4f}<br>R²: {r2:.4f}</sup>"
        )
        fig_pred.update_layout(
            title={
                "text": metrics_text,
                "yanchor": "top",
            }
        )
        self.processor.save_plot(
            fig_pred,
            player_name,
            "actual_vs_predicted.html",
        )


def trigger_process() -> None:
    """
    Trigger the data loading, model training, and visualization process.
    """
    predictor = NFLStatsPredictor()
    player_name = "Lamar Jackson"
    predictor.evaluate_lstm(player_name)


if __name__ == "__main__":
    trigger_process()
