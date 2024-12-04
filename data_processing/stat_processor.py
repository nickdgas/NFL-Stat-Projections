import os
import logging
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.tools import mpl_to_plotly 
import pendulum
import shap
from shap.plots import beeswarm
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
processor_logger = logging.getLogger("NFL-Processor")
processor_logger.setLevel(logging.INFO)


class NFLStatsProcessor:
    """
    A class to process NFL stats, generate visualizations, and save them for feature analysis.

    Attributes:
    - offensive_positions (list): Positions to process.
    - current_year (int): The current year.
    - backfill_year (int): The year to start backfilling the stats from.
    """

    offensive_positions = ["QB"]#, "RB", "WR", "TE"]
    current_year = pendulum.now().year
    backfill_year = pendulum.now().subtract(years=(current_year - 2021)).year

    def __init__(self):
        """
        Initialize list of files to process for offensive positions and years.
        """
        self.files = self.list_files(
            self.offensive_positions, range(self.backfill_year, self.current_year + 1)
        )

    @staticmethod
    def list_files(positions: list[str], year_range: range) -> dict:
        """
        List files for the specified positions and year range.

        Parameters:
        - positions (list[str]): List of offensive positions to process.
        - year_range (range): Range of years to fetch stats for.

        Returns:
        dict: A dictionary mapping each year to a list of file names to be processed.
        """
        files = {}
        for year in year_range:
            for pos in positions:
                directory = f"stats/{pos.lower()}_stats/{year}"
                files[year] = sorted(
                    os.listdir(directory),
                    key=lambda x: int(x.split("_")[-1].split(".")[0]),
                )
        return files

    @property
    def stats(self) -> pd.DataFrame:
        """
        Read and combine all CSV files for the specified years and positions into one large DataFrame.

        Returns:
        pd.DataFrame: A DataFrame containing the combined NFL stats.
        """
        df_list = []
        file_dict = self.files
        for pos in self.offensive_positions:
            for year in range(self.backfill_year, self.current_year + 1):
                for file in file_dict[year]:
                    directory = f"stats/{pos.lower()}_stats/{year}"
                    file_path = os.path.join(directory, file)
                    processor_logger.info(f"Reading file: {file_path}")
                    df = pd.read_csv(file_path)
                    df["POSITION"] = pos
                    df["YEAR"] = year
                    df["WEEK"] = int(file.split("_")[-1].split(".")[0])
                    df_list.append(df)
        stats_df = pd.concat(df_list, ignore_index=True)
        return stats_df

    def get_player_stats(
        self, player_name: str, year: int = None, position: str = None
    ) -> pd.DataFrame:
        """
        Extracts individual player's statistics based on player name, year, and position.

        Parameters:
        - player_name (str): Name of the player to filter for.
        - year (int, optional): Year of the statistics to filter for. Defaults to None (all years).
        - position (str, optional): Position of the player to filter for. Defaults to None (all positions).

        Returns:
        pd.DataFrame: A DataFrame containing the statistics of the specified player.
        """
        stats_df = self.stats
        player_stats = stats_df[
            stats_df["PLAYER"].str.contains(player_name, case=False, na=False)
        ]
        if year:
            player_stats = player_stats[player_stats["YEAR"] == year]
        if position:
            player_stats = player_stats[player_stats["POSITION"] == position]
        return player_stats

    @staticmethod
    def save_plot(fig, playername: str, filename: str) -> None:
        """
        Save a Plotly figure as an HTML file.

        Parameters:
        - fig (plotly.graph_objs.Figure): The Plotly figure to save.
        - filename (str): The filename for saving the HTML file.
        """
        if not os.path.exists(f"visualizations/{playername.replace(' ', '').lower()}"):
            os.makedirs(f"visualizations/{playername.replace(' ', '').lower()}")
        fig.write_html(f"visualizations/{playername.replace(' ', '').lower()}/{filename}")
        processor_logger.info(f"Saved visualization: {filename}")

    @staticmethod
    def create_plot(
        df: pd.DataFrame,
        plot_type: str,
        x_col: str = None,
        y_col: str = None,
        c_col: str = None,
        f_row: str = None,
        title: str = None,
    ) -> go.Figure:
        """
        Create a plot based on the given parameters.

        Parameters:
        - df (pd.DataFrame): Data for plotting.
        - plot_type (str): Type of plot ('line', 'scatter', 'bar', 'stack_bar', 'correlation', 'shap').
        - x_col (str): Column name to use for the x-axis.
        - y_col (str): Column name to use for the y-axis.
        - c_col (str): Column name to use for color differentiation.
        - f_row (str): Column name for facet rows.
        - title (str): Title of the plot.

        Returns:
        plotly.graph_objs.Figure: A Plotly figure object.
        """
        if plot_type == "line":
            fig = px.line(
                df,
                x=x_col,
                y=y_col,
                facet_row=f_row,
                color=c_col,
                line_shape="linear",
                title=title,
            )
            fig.update_layout(showlegend=True)
        elif plot_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, title=title)
            fig.update_layout(showlegend=True)
        elif plot_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, color=c_col, title=title)
            fig.update_layout(showlegend=True)
        elif plot_type == "stack_bar":
            fig = px.bar(
                df,
                x=x_col,
                y=y_col,
                facet_row=f_row,
                color=c_col,
                barmode="stack",
                title=title,
            )
            fig.update_layout(showlegend=True)
        elif plot_type == "correlation":
            fig = go.Figure(
                go.Heatmap(
                    z=df.values,
                    x=df.columns,
                    y=df.index,
                    colorscale=px.colors.diverging.RdBu,
                    text=df.values,
                    texttemplate="%{text:.2f}",
                    zmin=-1,
                    zmax=1,
                )
            )
            fig.update_layout(
                title=title,
                yaxis_autorange="reversed",
            )
        elif plot_type == "shap":
            df = df[df["GAMES"] != 0]
            X = df.drop(
                columns=[
                    "MISC_FPTS",
                    "RANK",
                    "PLAYER",
                    "GAMES",
                    "POSITION",
                    "YEAR",
                    "WEEK",
                ]
            )
            y = df["MISC_FPTS"]
            model = RandomForestRegressor(random_state=42)
            model.fit(X, y)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            explanation = shap.Explanation(
                values=shap_values, 
                base_values=explainer.expected_value, 
                data=X
            )
            beeswarm(explanation, max_display=len(X.columns), show=False)
            fig =  mpl_to_plotly(
                plt.gcf()
            )
            fig.update_layout(
                title=title,
                yaxis=dict(
                    tickmode="array", 
                    tickvals=list(range(len(X.columns))), 
                    ticktext=X.columns.tolist()  
                ),
            )
        return fig

    def generate_and_save_plots(self, player_name: str) -> None:
        """
        Generate and save multiple visualizations for a given player's performance over time.

        Parameters:
        - player_name (str): The name of the target player.
        """
        historical_player_stats = self.get_player_stats(player_name)
        overall_fantasy_points = (
            historical_player_stats[["MISC_FPTS", "YEAR", "WEEK"]]
            .groupby(by=["YEAR", "WEEK"])
            .agg("sum")
            .reset_index()
            .rename(columns={"MISC_FPTS": "FPTS"})
        )
        overall_yards_attempts = (
            historical_player_stats[
                [
                    "PASSING_YDS",
                    "RUSHING_YDS",
                    "PASSING_ATT",
                    "RUSHING_ATT",
                    "YEAR",
                    "WEEK",
                ]
            ]
            .groupby(by=["YEAR", "WEEK"])
            .agg("sum")
            .reset_index()
            .melt(
                id_vars=["YEAR", "WEEK"],
                value_vars=["PASSING_YDS", "RUSHING_YDS", "PASSING_ATT", "RUSHING_ATT"],
                var_name="STAT",
                value_name="VALUE",
            )
        )
        overall_attempts = overall_yards_attempts[
            overall_yards_attempts["STAT"].isin(["PASSING_ATT", "RUSHING_ATT"])
        ].rename(columns={"VALUE": "ATT"})
        overall_yards = overall_yards_attempts[
            overall_yards_attempts["STAT"].isin(["PASSING_YDS", "RUSHING_YDS"])
        ].rename(columns={"VALUE": "YDS"})
        overall_touchdowns = (
            historical_player_stats[["PASSING_TD", "RUSHING_TD", "YEAR", "WEEK"]]
            .groupby(by=["YEAR", "WEEK"])
            .sum()
            .reset_index()
            .melt(
                id_vars=["YEAR", "WEEK"],
                value_vars=["PASSING_TD", "RUSHING_TD"],
                var_name="STAT",
                value_name="TD",
            )
        )
        correlation_matrix = (
            historical_player_stats[historical_player_stats["GAMES"] != 0]
            .drop(columns=["RANK", "PLAYER", "GAMES", "POSITION", "YEAR", "WEEK"])
            .corr()[["MISC_FPTS"]]
            .sort_values(by="MISC_FPTS", ascending=False)
        )
        plot_definitions = [
            (
                overall_fantasy_points,
                "line",
                "WEEK",
                "FPTS",
                "YEAR",
                "YEAR",
                f"{player_name} - Fantasy Points",
                "pts_per_season.html",
            ),
            (
                overall_yards,
                "line",
                "WEEK",
                "YDS",
                "STAT",
                "YEAR",
                f"{player_name} - Passing/Rushing Yards",
                "yards_per_season.html",
            ),
            (
                overall_attempts,
                "stack_bar",
                "WEEK",
                "ATT",
                "STAT",
                "YEAR",
                f"{player_name} - Passing/Rushing Attempts",
                "attempts_per_season.html",
            ),
            (
                overall_touchdowns,
                "stack_bar",
                "WEEK",
                "TD",
                "STAT",
                "YEAR",
                f"{player_name} - Touchdowns",
                "touchdowns_per_season.html",
            ),
            (
                correlation_matrix,
                "correlation",
                None,
                None,
                None,
                None,
                f"{player_name} - Correlation to Fantasy Points",
                "stats_correlation_heatmap.html",
            ),
            (
                historical_player_stats,
                "shap",
                None,
                None,
                None,
                None,
                "SHAP Feature Importance",
                "shap_summary_plot.html",
            ),
        ]
        for (
            df,
            plot_type,
            x_col,
            y_col,
            c_col,
            f_row,
            title,
            file_name,
        ) in plot_definitions:
            fig = self.create_plot(df, plot_type, x_col, y_col, c_col, f_row, title)
            self.save_plot(fig, player_name, file_name)
        processor_logger.info("EDA completed and plots saved.")


def trigger_process() -> None:
    """
    Trigger the data loading and visualization process.
    """
    process = NFLStatsProcessor()
    historical_stats = process.stats
    processor_logger.info(f"Data loaded with {len(historical_stats)} rows.")
    process.generate_and_save_plots("Lamar Jackson")
    # process.generate_and_save_plots("Justin Jefferson")
    # process.generate_and_save_plots("Joe Mixon")
    # process.generate_and_save_plots("George Kittle")


if __name__ == "__main__":
    trigger_process()
