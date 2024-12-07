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
projector_logger = logging.getLogger("NFL-Projector")
projector_logger.setLevel(logging.INFO)


class NFLStatsProjector:
    """
    A class to fetch player stats, train a machine learning model, and plot actual vs predicted fantasy points.

    Attributes:
    - current_year (int): The current year.
    - backfill_year (int): The year to start backfilling the stats from.
    """

    current_year = pendulum.now().year
    backfill_year = pendulum.now().subtract(years=(current_year - 2021)).year

    def __init__(self, player_name: str, offensive_position: str):
        """
        Initialize NFLStatsProcessor to process stats for offensive positions.

        Parameters:
        - player_name (str): Name of player being researched.
        - offensive_position (str): Position target player plays.
        """
        self.processor = NFLStatsProcessor(player_name, offensive_position)

    @property
    def player_stats(self) -> pd.DataFrame:
        """
        Use NFLStatsProcessor to extract individual player's statistics based on player name, year, and position.

        Returns:
        pd.DataFrame: A DataFrame containing the statistics of the specified player.
        """
        return self.processor.get_player_stats()

    def _prepare_data(self) -> tuple:
        """
        Prepare the data by scaling features and labels, and splitting into train and test sets.

        Returns:
        tuple: Scaled training and test data (X_train, X_test, y_train, y_test).
        """
        stats = self.player_stats
        X = stats.drop(columns=["MISC_FPTS", "RANK", "PLAYER", "GAMES", "POSITION"])
        y = stats["MISC_FPTS"]
        projector_logger.info("Split and scale dataset for model fitting")
        X_scaler = StandardScaler()
        X_scaled = X_scaler.fit_transform(X)
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=0
        )
        return X_train, X_test, y_train, y_test, X_scaler, y_scaler

    def _build_model(self, input_shape: tuple) -> tf.keras.models.Sequential:
        """
        Build the LSTM model for fantasy points prediction.

        Parameters:
        - input_shape (tuple): Shape of the input data.

        Returns:
        - model (tf.keras.models.Sequential): Compiled LSTM model.
        """
        projector_logger.info("Build LSTM model")
        model = Sequential(
            [
                LSTM(
                    128,
                    activation="relu",
                    return_sequences=True,
                    input_shape=input_shape,
                ),
                Dropout(0.3),
                LSTM(64, activation="relu"),
                Dropout(0.3),
                Dense(1),
            ]
        )
        model.compile(optimizer=Adam(learning_rate=0.0005), loss="mean_squared_error")
        return model

    def _train_model(
        self,
        model: tf.keras.models.Sequential,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tf.keras.callbacks.History:
        """
        Train the LSTM model with early stopping to prevent overfit.

        Parameters:
        - model (tf.keras.models.Sequential): The LSTM model.
        - X_train (np.ndarray): The training features.
        - y_train (np.ndarray): The training labels.
        - X_test (np.ndarray): The test features.
        - y_test (np.ndarray): The test labels.

        Returns:
        - history (tf.keras.callbacks.History): History object containing training metrics.
        """
        projector_logger.info(f"Train LSTM model")
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        return model.fit(
            X_train,
            y_train,
            epochs=150,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1,
        )

    def plot_loss(self, history: tf.keras.callbacks.History, title: str) -> None:
        """
        Plot training and validation loss over epochs.

        Parameters:
        - history (tf.keras.callbacks.History): History object containing training metrics.
        - title (str): Title for the plot.
        """
        projector_logger.info("Visualize validation loss over epochs")
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(1, len(loss) + 1)
        loss_data = pd.DataFrame(
            {"Epoch": epochs, "Loss": loss, "Validation Loss": val_loss}
        )
        loss_data_melted = loss_data.melt(
            id_vars="Epoch",
            value_vars=["Loss", "Validation Loss"],
            var_name="Type",
            value_name="Value",
        )
        fig_loss = px.line(
            loss_data_melted,
            x="Epoch",
            y="Value",
            color="Type",
            title=title,
            labels={"Value": "Loss", "Epoch": "Epoch", "Type": "Loss Type"},
        )
        self.processor.save_plot(
            fig_loss, self.processor.player_name, "lstm_model_loss.html"
        )

    def plot_predictions(
        self,
        y_test_actual: np.ndarray,
        y_pred: np.ndarray,
        metrics_text: str,
        title: str,
    ) -> None:
        """
        Plot actual vs predicted fantasy points.

        Parameters:
        - y_test_actual (np.ndarray): Actual fantasy points.
        - y_pred (np.ndarray): Predicted fantasy points.
        - metrics_text (str): Evaluation metrics and projected fantasy points for next game.
        - title (str): Title for the plot.
        """
        projector_logger.info("Evaluate LSTM performance")
        prediction_df = pd.DataFrame(
            {
                "Sample": range(len(y_test_actual.flatten())),
                "Actual": y_test_actual.flatten(),
                "Predicted": y_pred.flatten(),
            }
        )
        prediction_data_melted = prediction_df.melt(
            id_vars="Sample",
            value_vars=["Actual", "Predicted"],
            var_name="Type",
            value_name="Value",
        )
        fig_pred = px.line(
            prediction_data_melted,
            x="Sample",
            y="Value",
            color="Type",
            title=title,
            labels={"Value": "Fantasy Points", "Sample": "Sample"},
        )
        fig_pred.update_layout(
            title={
                "text": metrics_text,
                "yanchor": "top",
            }
        )
        self.processor.save_plot(
            fig_pred,
            self.processor.player_name,
            "actual_vs_predicted.html",
        )

    def evaluate_lstm(self) -> None:
        """
        Train the LSTM model, evaluate performance, and generate plots.
        """
        X_train, X_test, y_train, y_test, X_scaler, y_scaler = self._prepare_data()
        model = self._build_model((X_train.shape[1], X_train.shape[2]))
        history = self._train_model(model, X_train, y_train, X_test, y_test)
        self.plot_loss(history, f"{self.processor.player_name} - Loss Over Epochs")
        y_pred_scaled = model.predict(X_test)
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        y_test_actual = y_scaler.inverse_transform(y_test)
        latest_input = X_train[-1].reshape(1, 1, X_train.shape[2])
        next_scaled_prediction = model.predict(latest_input)
        next_prediction = y_scaler.inverse_transform(next_scaled_prediction)
        mse = mean_squared_error(y_test_actual, y_pred)
        rmse = root_mean_squared_error(y_test_actual, y_pred)
        mae = mean_absolute_error(y_test_actual, y_pred)
        r2 = r2_score(y_test_actual, y_pred)
        metrics_text = f"Proj. FPTS: {next_prediction[0][0]:.2f}<br><br><sup>MSE: {mse:.4f}<br>RMSE: {rmse:.4f}<br>MAE: {mae:.4f}<br>R²: {r2:.4f}</sup>"
        self.plot_predictions(
            y_test_actual,
            y_pred,
            metrics_text,
            f"{self.processor.player_name} - Actual vs Predicted",
        )


def trigger_process(player_name: str, offensive_position: str) -> None:
    """
    Trigger the data loading, model training, and visualization process.

    Parameters:
    - player_name (str): Name of player being researched.
    - offensive_position (str): Position target player plays.
    """
    predictor = NFLStatsProjector(player_name, offensive_position)
    predictor.evaluate_lstm()


if __name__ == "__main__":
    fantasy_team = {
        "QB": "Lamar Jackson",
        "WR": ["Justin Jefferson", "CeeDee Lamb"],
        "RB": ["Joe Mixon", "Derrick Henry"],
        "TE": "George Kittle",
    }
    for position, players in fantasy_team.items():
        if not isinstance(players, list):
            players = [players]
        for player_name in players:
            projector_logger.info(f"Predicting stats for {player_name} ({position})")
            try:
                trigger_process(player_name, position)
            except Exception as e:
                projector_logger.error(f"Failed for {player_name} ({position}): {e}")
