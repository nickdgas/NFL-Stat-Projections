import os
import logging
import warnings
import pendulum
import pandas as pd
from typing import Generator, Any
from faker import Faker
from scrapy.http import Request
from scrapy import Spider
from scrapy.crawler import CrawlerProcess
from scrapy.spidermiddlewares.httperror import HttpError

warnings.filterwarnings("ignore")

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
crawler_logger = logging.getLogger("NFL-Crawler")
crawler_logger.setLevel(logging.INFO)


class NFLStatsCrawler(Spider):
    name = "nfl_stats_crawler"
    allowed_domains = ["fantasypros.com"]

    offensive_positions = ["QB"]  # , "RB", "WR", "TE"]
    year = pendulum.now().year
    backfill_year = pendulum.now().subtract(years=(year - 2021))

    def __init__(self, backfill: bool = False, *args, **kwargs):
        """
        Initialize the spider with the backfill flag.

        Parameters:
        - self (Self@NFLStatsCrawler): Class instance reference.
        - backfill (bool): Whether to backfill historical data (default is False).
        """
        super().__init__(*args, **kwargs)
        if backfill:
            self.start_urls = self.list_urls(
                self.offensive_positions, range(self.backfill_year.year, self.year + 1)
            )
        else:
            self.start_urls = self.list_urls(
                self.offensive_positions, range(self.year, self.year + 1)
            )

    @staticmethod
    def list_urls(positions: list[str], year_range: range) -> list[tuple[str]]:
        """
        Generate URLs for the specified positions and year.

        Parameters:
        - positions (list[str]): List of offensive positions.
        - year_range (range): Range of years to fetch stats for.

        Returns:
        list[tuple[str]]: List of nested tuples containing the basic and advanced URLs to scrape.
        """
        urls = []
        for year in year_range:
            for pos in positions:
                directory = f"stats/{pos.lower()}_stats/{year}"
                os.makedirs(directory, exist_ok=True)
                for week in range(1, 19):
                    urls.append(
                        (
                            f"https://www.fantasypros.com/nfl/stats/{pos.lower()}.php?year={year}&week={week}&range=week",
                            f"https://www.fantasypros.com/nfl/advanced-stats-{pos.lower()}.php?year={year}&week={week}&range=week",
                        )
                    )
        return sorted(urls)

    def start_requests(self) -> Generator[Request, Any, None]:
        """
        Make requests to URLs and set callbacks for parsing and error logging.

        Parameters:
        - self (Self@NFLStatsCrawler): Class instance reference.

        Returns:
        Generator[Request, Any, None]: Yield request(s) to URL(s) and collect response/log errors.
        """
        for urls in self.start_urls:
            basic_url, advanced_url = urls
            yield Request(
                basic_url,
                headers={"User-Agent": Faker().chrome()},
                callback=self.parse_basic_nfl_stats,
                errback=self.log_errors,
                meta={"dont_retry": False, "advanced_url": advanced_url},
            )

    @staticmethod
    def clean_data(value: Any) -> int | float | Any | None:
        """
        Clean and convert extracted data to appropriate types.

        Parameters:
        - value (Any): The extracted value to clean.

        Returns:
        (int | float | Any | None): The cleaned value, or None if it was None.
        """
        if value is None or value == "" or value == " " or value is pd.NA:
            return 0.0
        value = value.strip()
        if "%" in value:
            try:
                return float(value.replace("%", "")) / 100.0
            except ValueError:
                return 0.0 
        if value.isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            return value

    @staticmethod
    def create_dataframe(position: str, rows: Any, type: str) -> pd.DataFrame:
        """
        Create a DataFrame from the extracted rows based on the player's position.

        Parameters:
        - position (str): The position of the players (QB, RB, WR, TE).
        - rows (Any): The rows extracted from the response.
        - type (str): Advanced or Basic statistics

        Returns:
        pd.DataFrame: A DataFrame containing the statistics.
        """
        if type == "advanced":
            position_mapping = {
                "QB": {
                    "RANK": "./td[@class='player-rank']/text()",
                    "PLAYER": "./td[@class='player-label player-label-report-page']/a/text()",
                    "GAMES": "./td[3]/text()",
                    "PASSING_COMP": "./td[4]/text()",
                    "PASSING_ATT": "./td[5]/text()",
                    "PASSING_PCT": "./td[6]/text()",
                    "PASSING_YDS": "./td[7]/text()",
                    "PASSING_Y/A": "./td[8]/text()",
                    "PASSING_AIR": "./td[9]/text()",
                    "PASSING_AIR/A": "./td[10]/text()",
                    "DEEP_BALL_10+_YDS": "./td[11]/text()",
                    "DEEP_BALL_20+_YDS": "./td[12]/text()",
                    "DEEP_BALL_30+_YDS": "./td[13]/text()",
                    "DEEP_BALL_40+_YDS": "./td[14]/text()",
                    "DEEP_BALL_50+_YDS": "./td[15]/text()",
                    "PRESSURE_PKT_TIME": "./td[16]/text()",
                    "PRESSURE_SACK": "./td[17]/text()",
                    "PRESSURE_KNCK": "./td[18]/text()",
                    "PRESSURE_HRRY": "./td[19]/text()",
                    "PRESSURE_BLITZ": "./td[20]/text()",
                    "MISC_POOR": "./td[21]/text()",
                    "MISC_DROP": "./td[22]/text()",
                    "MISC_RZ_ATT": "./td[23]/text()",
                    "MISC_RTG": "./td[24]/text()",
                },
                "RB": {
                    "RANK": "./td[@class='player-rank']/text()",
                    "PLAYER": "./td[@class='player-label player-label-report-page']/a/text()",
                    "GAMES": "./td[3]/text()",
                    "RUSHING_ATT": "./td[4]/text()",
                    "RUSHING_YDS": "./td[5]/text()",
                    "RUSHING_Y/ATT": "./td[6]/text()",
                    "RUSHING_YBCON": "./td[7]/text()",
                    "RUSHING_YBCON/ATT": "./td[8]/text()",
                    "RUSHING_YACON": "./td[9]/text()",
                    "RUSHING_YACON/ATT": "./td[10]/text()",
                    "RUSHING_BRKTKL": "./td[11]/text()",
                    "RUSHING_TK_LOSS": "./td[12]/text()",
                    "RUSHING_TK_LOSS_YDS": "./td[13]/text()",
                    "RUSHING_LNG_TD": "./td[14]/text()",
                    "BIG_RUSH_PLAYS_10+_YDS": "./td[15]/text()",
                    "BIG_RUSH_PLAYS_20+_YDS": "./td[16]/text()",
                    "BIG_RUSH_PLAYS_30+_YDS": "./td[17]/text()",
                    "BIG_RUSH_PLAYS_40+_YDS": "./td[18]/text()",
                    "BIG_RUSH_PLAYS_50+_YDS": "./td[19]/text()",
                    "BIG_RUSH_PLAYS_LNG": "./td[20]/text()",
                    "RECEIVING_REC": "./td[21]/text()",
                    "RECEIVING_TGT": "./td[22]/text()",
                    "RECEIVING_RZ_TGT": "./td[23]/text()",
                    "RECEIVING_YACON": "./td[24]/text()",
                },
                ("WR", "TE"): {
                    "RANK": "./td[@class='player-rank']/text()",
                    "PLAYER": "./td[@class='player-label player-label-report-page']/a/text()",
                    "GAMES": "./td[3]/text()",
                    "RECEIVING_REC": "./td[4]/text()",
                    "RECEIVING_YDS": "./td[5]/text()",
                    "RECEIVING_Y/R": "./td[6]/text()",
                    "RECEIVING_YBC": "./td[7]/text()",
                    "RECEIVING_YBC/R": "./td[8]/text()",
                    "RECEIVING_AIR": "./td[9]/text()",
                    "RECEIVING_AIR/R": "./td[10]/text()",
                    "RECEIVING_YAC": "./td[11]/text()",
                    "RECEIVING_YACON": "./td[12]/text()",
                    "RECEIVING_YACON/R": "./td[13]/text()",
                    "RECEIVING_BRKTKL": "./td[14]/text()",
                    "TARGETS_TGT": "./td[15]/text()",
                    "TARGETS_%_TM": "./td[16]/text()",
                    "TARGETS_CATCHABLE": "./td[17]/text()",
                    "TARGETS_DROP": "./td[18]/text()",
                    "TARGETS_RZ_TGT": "./td[19]/text()",
                    "BIG_PLAYS_10+_YDS": "./td[20]/text()",
                    "BIG_PLAYS_20+_YDS": "./td[21]/text()",
                    "BIG_PLAYS_30+_YDS": "./td[22]/text()",
                    "BIG_PLAYS_40+_YDS": "./td[23]/text()",
                    "BIG_PLAYS_50+_YDS": "./td[24]/text()",
                    "BIG_PLAYS_LNG": "./td[24]/text()",
                },
            }
        elif type == "basic":
            position_mapping = {
                "QB": {
                    "PLAYER": "./td[@class='player-label player-label-report-page']/a/text()",
                    "PASSING_TD": "./td[8]/text()",
                    "PASSING_INT": "./td[9]/text()",
                    "RUSHING_ATT": "./td[11]/text()",
                    "RUSHING_YDS": "./td[12]/text()",
                    "RUSHING_TD": "./td[13]/text()",
                    "MISC_FL": "./td[14]/text()",
                    "MISC_FPTS": "./td[16]/text()",
                },
            }
        for pos_key, pos_mapping in position_mapping.items():
            if position in pos_key:
                data = []
                for row in rows:
                    data.append(
                        {
                            stat_name: NFLStatsCrawler.clean_data(
                                row.xpath(xpath).extract_first()
                            )
                            for stat_name, xpath in pos_mapping.items()
                        }
                    )
                df = pd.DataFrame(data).dropna(how="all").fillna(0.0).reset_index(drop=True)
                return df

    def parse_basic_nfl_stats(self, response: Any) -> Generator[Request, Any, None]:
        """
        Parse the response and yield a new request for advanced statistics.
        Save basic statistics as metadata for new request.

        Parameters:
        - self (Self@NFLStatsCrawler): Class instance reference.
        - response (Any): The response object from Scrapy.

        Returns:
        Generator[Request, Any, None]: Yield request(s) to advanced URL(s) and collects response/log errors.
        """
        crawler_logger.info(f"Processing {response.url}")

        position = response.url.split("stats/")[1].split(".php")[0].upper()

        rows = response.xpath("//tbody/tr")
        basic_stats_df = self.create_dataframe(position, rows, type="basic")

        advanced_url = response.meta["advanced_url"]
        yield Request(
            advanced_url,
            headers={"User-Agent": Faker().chrome()},
            callback=self.parse_advanced_nfl_stats,
            errback=self.log_errors,
            meta={"basic_stats_df": basic_stats_df},
        )

    def parse_advanced_nfl_stats(self, response: Any) -> None:
        """
        Parse the response and return a DataFrame with advanced statistics.
        Merge the basic and advanced stats DataFrames on the player name.
        Save the statistics to a CSV file.

        Parameters:
        - self (Self@NFLStatsCrawler): Class instance reference.
        - response (Any): The response object from Scrapy.

        Returns:
        None: Saves combined stat line as CSV.
        """
        crawler_logger.info(f"Processing {response.url}")

        year = response.url.split("year=")[1].split("&")[0]
        week = response.url.split("week=")[1].split("&")[0]
        position = response.url.split("advanced-stats-")[1].split(".php")[0].upper()

        rows = response.xpath("//tbody/tr")
        advanced_stats_df = self.create_dataframe(position, rows, type="advanced")

        basic_stats_df = response.meta["basic_stats_df"]

        merged_df = pd.merge(
            advanced_stats_df, basic_stats_df, on=["PLAYER"], how="left"
        ).fillna(0.0)
        filepath = f"stats/{position.lower()}_stats/{year}/nfl_stats_{position.lower()}_week_{week}.csv"
        merged_df.to_csv(filepath, index=False)
        crawler_logger.info(f"Exported data to {filepath}")

    def log_errors(self, failure: Any) -> None:
        """
        Log errors encountered during requests and data retrieval.

        Parameters:
        - self (Self@NFLStatsCrawler): Class instance reference.
        - failure (Any): The failure object containing error details.

        Returns:
        None: Display error logs within console.
        """
        crawler_logger.error(repr(failure))
        if failure.check(HttpError):
            response = failure.value.response
            crawler_logger.error(f"HttpError on {response.url}: {response.status}")
            if hasattr(response, "data"):
                player_names = [
                    row.xpath(
                        "./td[@class='player-label player-label-report-page']/a/text()"
                    ).extract_first()
                    for row in response.xpath("//tbody/tr")
                ]
                crawler_logger.error("Player names in this response: %s", player_names)
        else:
            crawler_logger.error("Request failed with %s", failure)


def trigger_process(backfill: bool = False) -> None:
    """
    Trigger web crawler

    Parameters:
    - backfill (bool): Whether to perform backfilling.

    Returns:
    None: Crawl fantasypros.com.
    """
    logging.getLogger("scrapy").propagate = False
    process = CrawlerProcess(install_root_handler=False)
    process.crawl(NFLStatsCrawler, backfill=backfill)
    process.start()


if __name__ == "__main__":
    trigger_process()
    # trigger_process(backfill=True)
