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
    - current_year (int): The current year.
    - backfill_year (int): The year to start backfilling the stats from.
    """

    current_year = pendulum.now().year
    backfill_year = pendulum.now().subtract(years=(current_year - 2021)).year

    def __init__(self, player_name: str, offensive_position: str):
        """
        Initialize list of files to process for offensive positions and years.

        Parameters:
        - player_name (str): Name of player being researched.
        - offensive_position (str): Position target player plays.
        """
        self.player_name = player_name
        self.offensive_position = offensive_position
        self.files = self.list_files(
            self.offensive_position, 
            range(self.backfill_year, self.current_year + 1)
        )

    @staticmethod
    def list_files(position: str, year_range: range) -> dict:
        """
        List files for the specified positions and year range.

        Parameters:
        - position ((str): Position target player plays
        - year_range (range): Range of years to fetch stats for.

        Returns:
        dict: A dictionary mapping each year to a list of file names to be processed.
        """
        files = {}
        for year in year_range:
            directory = f"stats/{position.lower()}_stats/{year}"
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
        for year in range(self.backfill_year, self.current_year + 1):
            for file in file_dict[year]:
                directory = f"stats/{self.offensive_position.lower()}_stats/{year}"
                file_path = os.path.join(directory, file)
                processor_logger.info(f"Reading file: {file_path}")
                df = pd.read_csv(file_path)
                df["POSITION"] = self.offensive_position
                df["YEAR"] = year
                df["WEEK"] = int(file.split("_")[-1].split(".")[0])
                df_list.append(df)
        stats_df = pd.concat(df_list, ignore_index=True)
        return stats_df

    def get_player_stats(
        self, year: int = None
    ) -> pd.DataFrame:
        """
        Extracts individual player's statistics based on player name, year, and position.

        Parameters:
        - year (int, optional): Year of the statistics to filter for. Defaults to None (all years).

        Returns:
        pd.DataFrame: A DataFrame containing the statistics of the specified player.
        """
        stats_df = self.stats
        player_stats = stats_df[
            stats_df["PLAYER"].str.contains(self.player_name, case=False, na=False)
        ]
        if year:
            player_stats = player_stats[player_stats["YEAR"] == year]
            player_stats = player_stats[player_stats["POSITION"] == self.offensive_position]
        return player_stats

    @staticmethod
    def save_plot(fig, playername: str, filename: str) -> None:
        """
        Save a Plotly figure as an HTML file.

        Parameters:
        - fig (plotly.graph_objs.Figure): The Plotly figure to save.
        - playername (str): Name of player being researched.
        - filename (str): The filename for saving the HTML file.
        """
        graph_path = f"visualizations/{playername.replace(' ', '').lower()}/"
        if not os.path.exists(graph_path):
            os.makedirs(graph_path)
        fig.write_html(graph_path+filename)
        processor_logger.info(f"Saved visualization for {playername}: {filename}")

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

    def generate_and_save_plots(self) -> None:
        """
        Generate and save multiple visualizations for a given player's performance over time.
        """
        historical_player_stats = self.get_player_stats()
        overall_fantasy_points = (
            historical_player_stats[["MISC_FPTS", "YEAR", "WEEK"]]
            .groupby(by=["YEAR", "WEEK"])
            .agg("sum")
            .reset_index()
            .rename(columns={"MISC_FPTS": "FPTS"})
        )
        if self.offensive_position == "QB":
            yards_col = "PASSING_YDS"
            attempts_col = "PASSING_ATT"
            td_col = "PASSING_TD"
        else:
            yards_col = "RECEIVING_YDS"
            attempts_col = "RECEIVING_REC"
            td_col = "RECEIVING_TD"

        overall_yards_attempts = (
            historical_player_stats[
                [
                    yards_col,
                    attempts_col,
                    "RUSHING_YDS",
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
                value_vars=[yards_col, attempts_col, "RUSHING_YDS", "RUSHING_ATT"],
                var_name="STAT",
                value_name="VALUE",
            )
        )
        overall_attempts = overall_yards_attempts[
            overall_yards_attempts["STAT"].isin([attempts_col, "RUSHING_ATT"])
        ].rename(columns={"VALUE": "ATT"})
        overall_yards = overall_yards_attempts[
            overall_yards_attempts["STAT"].isin([yards_col, "RUSHING_YDS"])
        ].rename(columns={"VALUE": "YDS"})
        overall_touchdowns = (
            historical_player_stats[[td_col, "RUSHING_TD", "YEAR", "WEEK"]]
            .groupby(by=["YEAR", "WEEK"])
            .sum()
            .reset_index()
            .melt(
                id_vars=["YEAR", "WEEK"],
                value_vars=[td_col, "RUSHING_TD"],
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
                f"{self.player_name} - Fantasy Points",
                "pts_per_season.html",
            ),
            (
                overall_yards,
                "line",
                "WEEK",
                "YDS",
                "STAT",
                "YEAR",
                f"{self.player_name} - Overall Yards",
                "yards_per_season.html",
            ),
            (
                overall_attempts,
                "stack_bar",
                "WEEK",
                "ATT",
                "STAT",
                "YEAR",
                f"{self.player_name} - Overall Attempts",
                "attempts_per_season.html",
            ),
            (
                overall_touchdowns,
                "stack_bar",
                "WEEK",
                "TD",
                "STAT",
                "YEAR",
                f"{self.player_name} - Oveerall Touchdowns",
                "touchdowns_per_season.html",
            ),
            (
                correlation_matrix,
                "correlation",
                None,
                None,
                None,
                None,
                f"{self.player_name} - Correlation to Fantasy Points",
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
            self.save_plot(fig, self.player_name, file_name)
        processor_logger.info("EDA completed and plots saved.")


def trigger_process(player_name: str, offensive_position: str) -> None:
    """
    Trigger the data loading and visualization process.
    """
    process = NFLStatsProcessor(player_name, offensive_position)
    historical_stats = process.stats
    processor_logger.info(f"Data loaded with {len(historical_stats)} rows.")
    process.generate_and_save_plots()


if __name__ == "__main__":
    fantasy_team = {
        "QB": "Lamar Jackson",
        "WR": ["Justin Jefferson", "CeeDee Lamb"],
        "RB": ["Joe Mixon", "Derrick Henry"],
        "TE": "George Kittle"
    }
    for position, players in fantasy_team.items():
        if not isinstance(players, list):
            players = [players]
        for player_name in players:
            processor_logger.info(f"Processing stats for {player_name} ({position})")
            try:
                trigger_process(player_name, position)
            except Exception as e:
                processor_logger.error(f"Failed for {player_name} ({position}): {e}")