import os
import logging
import warnings
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pendulum

warnings.filterwarnings("ignore")

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
processor_logger = logging.getLogger("NFL-Processor")
processor_logger.setLevel(logging.INFO)


class NFLStatsProcessor:
    """
    A class to process NFL stats, generate visualizations, and save them for later viewing.

    Attributes:
        offensive_positions (list): Positions to process (currently only "QB").
        current_year (int): The current year.
        backfill_year (int): The year to start backfilling the stats from.
    """

    offensive_positions = ["QB"]  # , "RB", "WR", "TE"]
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
    def save_plot(fig, filename: str) -> None:
        """
        Save a Plotly figure as an HTML file.

        Parameters:
        - fig (plotly.graph_objs.Figure): The Plotly figure to save.
        - filename (str): The filename for saving the HTML file.
        """
        if not os.path.exists("visualizations"):
            os.makedirs("visualizations")
        fig.write_html(f"visualizations/{filename}")
        processor_logger.info(f"Saved visualization: {filename}")

    @staticmethod
    def create_plot(
        df: pd.DataFrame, plot_type: str, x_col: str, y_col: str, title: str
    ) -> go.Figure:
        """
        Create a plot based on the given parameters.

        Parameters:
        - df (pd.DataFrame): Data for plotting.
        - plot_type (str): Type of plot ('line', 'scatter', 'bar').
        - x_col (str): Column name to use for the x-axis.
        - y_col (str): Column name to use for the y-axis.
        - title (str): Title of the plot.

        Returns:
        plotly.graph_objs.Figure: A Plotly figure object.
        """
        if plot_type == "line":
            fig = px.line(df, x=x_col, y=y_col, title=title, color="YEAR", line_shape="linear")
        elif plot_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, title=title)
        elif plot_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, title=title, color="YEAR")
        elif plot_type == "stack_bar":
            fig = px.bar(df, x=x_col, y=y_col, title=title, color="YEAR", barmode="group")
        elif plot_type == "hist":
            fig = px.histogram(df, x=x_col, nbins=30, title=title)
        fig.update_layout(showlegend=True)
        return fig

    def generate_and_save_plots(self, player_name: str) -> None:
        """
        Generate and save multiple visualizations for a given player's performance over time.

        Parameters:
        - player_name (str): The name of the target player.

        Returns:
        None
        """
        historical_player_stats = self.get_player_stats(player_name)
        historical_player_stats["TOTAL_TD"] = (
            historical_player_stats["PASSING_TD"]
            + historical_player_stats["RUSHING_TD"]
        )
        overall_passing_yards = (
            historical_player_stats[["PASSING_YDS", "YEAR", "WEEK"]]
            .groupby(by=["YEAR", "WEEK"])
            .agg("sum")
            .reset_index()
        )
        overall_rushing_yards = (
            historical_player_stats[["RUSHING_YDS", "YEAR", "WEEK"]]
            .groupby(by=["YEAR", "WEEK"])
            .agg("sum")
            .reset_index()
        )
        overall_touchdowns = (
            historical_player_stats[["TOTAL_TD", "YEAR", "WEEK"]]
            .groupby(by=["YEAR", "WEEK"])
            .agg("sum")
            .reset_index()
        )
        games_missed = (
            historical_player_stats[historical_player_stats["GAMES"] == 0][["GAMES", "YEAR"]]
            .groupby(by=["YEAR"])
            .agg("count")
            .reset_index()
        )

        plot_definitions = [
            (
                historical_player_stats,
                "hist",
                "PASSING_YDS",
                None,
                f"{player_name} - Distribution of Passing Yards",
                "passing_yards_distribution.html",
            ),
            (
                historical_player_stats,
                "hist",
                "RUSHING_YDS",
                None,
                f"{player_name} - Distribution of Rushing Yards",
                "rushing_yards_distribution.html",
            ),
            (
                historical_player_stats,
                "hist",
                "TOTAL_TD",
                None,
                f"{player_name} - Distribution of Touchdowns",
                "touchdowns_distribution.html",
            ),
            (
                overall_passing_yards,
                "line",
                "WEEK",
                "PASSING_YDS",
                f"{player_name} - Passing Yards per Game",
                "passing_yards_season.html",
            ),
            (
                overall_rushing_yards,
                "line",
                "WEEK",
                "RUSHING_YDS",
                f"{player_name} - Rushing Yards Per Game",
                "rushing_yards_game.html",
            ),
            (
                overall_touchdowns,
                "stack_bar",
                "WEEK",
                "TOTAL_TD",
                f"{player_name} - Total TDs Per Game",
                "touchdowns_game.html",
            ),
            (
                games_missed,
                "bar",
                "YEAR",
                "GAMES",
                f"{player_name} - Games Missed Per Season",
                "games_missed_season.html",
            ),
        ]
        for df, plot_type, x_col, y_col, title, file_name in plot_definitions:
            fig = self.create_plot(df, plot_type, x_col, y_col, title)
            self.save_plot(fig, file_name)
        processor_logger.info("EDA completed and plots saved.")


def trigger_process() -> None:
    """
    Trigger the data loading and visualization process.
    """
    process = NFLStatsProcessor()
    historical_stats = process.stats
    processor_logger.info(f"Data loaded with {len(historical_stats)} rows.")
    process.generate_and_save_plots("Lamar Jackson")


if __name__ == "__main__":
    trigger_process()
