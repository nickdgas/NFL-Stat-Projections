import os
import logging
import warnings
import pandas as pd
import plotly.express as px
import pendulum

warnings.filterwarnings("ignore")

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
processor_logger = logging.getLogger("NFL-Processor")
processor_logger.setLevel(logging.INFO)

class NFLStatsProcessor():
    """
    A class to process NFL stats, generate visualizations, and save them for later viewing.
    
    Attributes:
        offensive_positions (list): Positions to process (currently only "QB").
        current_year (int): The current year.
        backfill_year (int): The year to start backfilling the stats from.
    """
    offensive_positions = ["QB"] #, "RB", "WR", "TE"]
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
                files[year] = sorted(os.listdir(directory), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return files

    def read_data(self) -> pd.DataFrame:
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
                    df["WEEK"] = int(file.split("_")[-1].split('.')[0])
                    df_list.append(df)
        stats_df = pd.concat(df_list, ignore_index=True)
        return stats_df

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

    def create_visualizations(self, df: pd.DataFrame) -> None:
        """
        Generate and save visualizations for various player statistics.

        Parameters:
        - df (pd.DataFrame): The cleaned DataFrame containing NFL stats.
        """
        self.create_fantasy_points_over_time(df)
        self.create_passing_yards_vs_fantasy_points(df)
        self.create_rushing_yards_vs_fantasy_points(df)
        self.create_deep_balls_vs_fantasy_points(df)
        self.create_passing_pct_vs_fantasy_points(df)

    def create_fantasy_points_over_time(self, df: pd.DataFrame) -> None:
        """
        Create a visualization showing Fantasy Points over time (by Week).
        
        Parameters:
        - df (pd.DataFrame): The DataFrame containing NFL stats.
        """
        fig = px.line(df, x="WEEK", y="MISC_FPTS", color="PLAYER", 
                      title="Fantasy Points Over Time (by Week)",
                      labels={"MISC_FPTS": "Fantasy Points", "WEEK": "Week", "PLAYER": "Player"})
        fig.update_traces(mode='markers+lines')
        self.save_plot(fig, "fantasy_points_over_time.html")

    def create_passing_yards_vs_fantasy_points(self, df: pd.DataFrame) -> None:
        """
        Create a scatter plot of Passing Yards vs Fantasy Points.
        
        Parameters:
        - df (pd.DataFrame): The DataFrame containing NFL stats.
        """
        fig = px.scatter(df, x="PASSING_YDS", y="MISC_FPTS", color="PLAYER", 
                         title="Passing Yards vs Fantasy Points",
                         labels={"PASSING_YDS": "Passing Yards", "MISC_FPTS": "Fantasy Points", "PLAYER": "Player"})
        fig.update_traces(marker=dict(size=8))
        self.save_plot(fig, "passing_yards_vs_fantasy_points.html")

    def create_rushing_yards_vs_fantasy_points(self, df: pd.DataFrame) -> None:
        """
        Create a scatter plot of Rushing Yards vs Fantasy Points.
        
        Parameters:
        - df (pd.DataFrame): The DataFrame containing NFL stats.
        """
        fig = px.scatter(df, x="RUSHING_YDS", y="MISC_FPTS", color="PLAYER",
                         title="Rushing Yards vs Fantasy Points",
                         labels={"RUSHING_YDS": "Rushing Yards", "MISC_FPTS": "Fantasy Points", "PLAYER": "Player"})
        fig.update_traces(marker=dict(size=8))
        self.save_plot(fig, "rushing_yards_vs_fantasy_points.html")

    def create_deep_balls_vs_fantasy_points(self, df: pd.DataFrame) -> None:
        """
        Create a scatter plot of Deep Balls (20+ Yards) vs Fantasy Points.
        
        Parameters:
        - df (pd.DataFrame): The DataFrame containing NFL stats.
        """
        fig = px.scatter(df, x="DEEP_BALL_20+_YDS", y="MISC_FPTS", color="PLAYER", 
                         title="Deep Balls (20+ Yards) vs Fantasy Points",
                         labels={"DEEP_BALL_20+_YDS": "Deep Balls 20+ Yards", "MISC_FPTS": "Fantasy Points", "PLAYER": "Player"})
        fig.update_traces(marker=dict(size=8))
        self.save_plot(fig, "deep_balls_vs_fantasy_points.html")

    def create_passing_pct_vs_fantasy_points(self, df: pd.DataFrame) -> None:
        """
        Create a scatter plot of Passing Completion Percentage vs Fantasy Points.
        
        Parameters:
        - df (pd.DataFrame): The DataFrame containing NFL stats.
        """
        fig = px.scatter(df, x="PASSING_PCT", y="MISC_FPTS", color="PLAYER", 
                         title="Passing Completion Percentage vs Fantasy Points",
                         labels={"PASSING_PCT": "Passing Completion %", "MISC_FPTS": "Fantasy Points", "PLAYER": "Player"})
        fig.update_traces(marker=dict(size=8))
        self.save_plot(fig, "passing_pct_vs_fantasy_points.html")


def trigger_process() -> None:
    """
    Trigger web crawler
    """
    process = NFLStatsProcessor()
    historical_data = process.read_data()
    processor_logger.info(f"Data loaded with {len(historical_data)} rows.")
    process.create_visualizations(historical_data)


if __name__ == "__main__":
    trigger_process()
