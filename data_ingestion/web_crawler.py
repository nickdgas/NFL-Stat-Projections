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

    offensive_positions = ["QB", "RB", "WR", "TE"]
    year = pendulum.now().year

    @staticmethod
    def list_urls(positions, year) -> list[str]:
        """
        Generate URLs for the specified positions and year.
        
        Parameters:
        - positions (list): List of offensive positions.
        - year (int): The year for which to fetch the stats.

        Returns:
        list[str]: List of URLs to scrape.
        """
        urls = []
        for pos in positions:
            directory = f"advanced_stats/{pos.lower()}_stats/{year}"
            os.makedirs(directory, exist_ok=True)
            existing_weeks = [
                int(file.split("_week_")[1].split(".csv")[0])
                for file in os.listdir(directory)
                if file.startswith("nfl_stats_") and "_week_" in file
            ]
            next_week = 1
            while next_week in existing_weeks:
                next_week += 1
            if next_week <= 18:
                urls.append(
                    f"https://www.fantasypros.com/nfl/advanced-stats-{pos.lower()}.php?year={year}&week={next_week}&range=week"
                )
        return urls

    start_urls = list_urls(offensive_positions, year)

    def start_requests(self) -> Generator[Request, Any, None]:
        """
        Make requests to URLs and set callbacks for parsing and error logging.

        Parameters:
        - self (Self@NFLStatsCrawler): Class instance reference.

        Returns:
        Generator[Request, Any, None]: Yield request(s) to URL(s) and collect response/log errors.
        """
        for url in self.start_urls:
            yield Request(
                url,
                headers={"User-Agent": Faker().chrome()},
                callback=self.parse_nfl_stats,
                errback=self.log_errors,
                meta={"dont_retry": False},
            )

    @staticmethod
    def clean_data(value: Any) -> (int | float | Any | None):
        """
        Clean and convert extracted data to appropriate types.
        
        Parameters:
        - value (Any): The extracted value to clean.

        Returns:
        (int | float | Any | None): The cleaned value, or None if it was None.
        """
        if value is None:
            return None
        value = value.strip()
        if value.isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            return value

    @staticmethod
    def create_dataframe(position: str, rows: Any) -> pd.DataFrame:
        """
        Create a DataFrame from the extracted rows based on the player's position.
        
        Parameters:
        - position (str): The position of the players (QB, RB, WR, TE).
        - rows (Any): The rows extracted from the response.

        Returns:
        pd.DataFrame: A DataFrame containing the statistics.
        """
        position_mapping = {
            "QB": [
                ("", "RANK", "./td[@class='player-rank']/text()"),
                ("", "PLAYER", "./td[@class='player-label player-label-report-page']/a/text()"),
                ("GAMES", "G", "./td[3]/text()"),
                ("PASSING", "COMP", "./td[4]/text()"),
                ("PASSING", "ATT", "./td[5]/text()"),
                ("PASSING", "PCT", "./td[6]/text()"),
                ("PASSING", "YDS", "./td[7]/text()"),
                ("PASSING", "Y/A", "./td[8]/text()"),
                ("PASSING", "AIR", "./td[9]/text()"),
                ("PASSING", "AIR/A", "./td[10]/text()"),
                ("DEEP BALL PASSING", "10+ YDS", "./td[11]/text()"),
                ("DEEP BALL PASSING", "20+ YDS", "./td[12]/text()"),
                ("DEEP BALL PASSING", "30+ YDS", "./td[13]/text()"),
                ("DEEP BALL PASSING", "40+ YDS", "./td[14]/text()"),
                ("DEEP BALL PASSING", "50+ YDS", "./td[15]/text()"),
                ("PRESSURE", "PKT TIME", "./td[16]/text()"),
                ("PRESSURE", "SACK", "./td[17]/text()"),
                ("PRESSURE", "KNCK", "./td[18]/text()"),
                ("PRESSURE", "HRRY", "./td[19]/text()"),
                ("PRESSURE", "BLITZ", "./td[20]/text()"),
                ("MISC", "POOR", "./td[21]/text()"),
                ("MISC", "DROP", "./td[22]/text()"),
                ("MISC", "RZ ATT", "./td[23]/text()"),
                ("MISC", "RTG", "./td[24]/text()"),
            ],
            "RB": [
                ("", "RANK", "./td[@class='player-rank']/text()"),
                ("", "PLAYER", "./td[@class='player-label player-label-report-page']/a/text()"),
                ("GAMES", "G", "./td[3]/text()"),
                ("RUSHING", "ATT", "./td[4]/text()"),
                ("RUSHING", "YDS", "./td[5]/text()"),
                ("RUSHING", "Y/ATT", "./td[6]/text()"),
                ("RUSHING", "YBCON", "./td[7]/text()"),
                ("RUSHING", "YBCON/ATT", "./td[8]/text()"),
                ("RUSHING", "YACON", "./td[9]/text()"),
                ("RUSHING", "YACON/ATT", "./td[10]/text()"),
                ("RUSHING", "BRKTKL", "./td[11]/text()"),
                ("RUSHING", "TK LOSS", "./td[12]/text()"),
                ("RUSHING", "TK LOSS YDS", "./td[13]/text()"),
                ("RUSHING", "LNG TD", "./td[14]/text()"),
                ("BIG RUSH PLAYS", "10+ YDS", "./td[15]/text()"),
                ("BIG RUSH PLAYS", "20+ YDS", "./td[16]/text()"),
                ("BIG RUSH PLAYS", "30+ YDS", "./td[17]/text()"),
                ("BIG RUSH PLAYS", "40+ YDS", "./td[18]/text()"),
                ("BIG RUSH PLAYS", "50+ YDS", "./td[19]/text()"),
                ("BIG RUSH PLAYS", "LNG", "./td[20]/text()"),
                ("RECEIVING", "REC", "./td[21]/text()"),
                ("RECEIVING", "TGT", "./td[22]/text()"),
                ("RECEIVING", "RZ TGT", "./td[23]/text()"),
                ("RECEIVING", "YACON", "./td[24]/text()"),
            ],
            "WR": [
                ("", "RANK", "./td[@class='player-rank']/text()"),
                ("", "PLAYER", "./td[@class='player-label player-label-report-page']/a/text()"),
                ("GAMES", "G", "./td[3]/text()"),
                ("RECEIVING", "REC", "./td[4]/text()"),
                ("RECEIVING", "YDS", "./td[5]/text()"),
                ("RECEIVING", "Y/R", "./td[6]/text()"),
                ("RECEIVING", "YBC", "./td[7]/text()"),
                ("RECEIVING", "YBC/R", "./td[8]/text()"),
                ("RECEIVING", "AIR", "./td[9]/text()"),
                ("RECEIVING", "AIR/R", "./td[10]/text()"),
                ("RECEIVING", "YAC", "./td[11]/text()"),
                ("RECEIVING", "YACON", "./td[12]/text()"),
                ("RECEIVING", "YACON/R", "./td[13]/text()"),
                ("RECEIVING", "BRKTKL", "./td[14]/text()"),
                ("TARGETS", "TGT", "./td[15]/text()"),
                ("TARGETS", "% TM", "./td[16]/text()"),
                ("TARGETS", "CATCHABLE", "./td[17]/text()"),
                ("TARGETS", "DROP", "./td[18]/text()"),
                ("TARGETS", "RZ TGT", "./td[19]/text()"),
                ("BIG PLAYS", "10+ YDS", "./td[20]/text()"),
                ("BIG PLAYS", "20+ YDS", "./td[21]/text()"),
                ("BIG PLAYS", "30+ YDS", "./td[22]/text()"),
                ("BIG PLAYS", "40+ YDS", "./td[23]/text()"),
                ("BIG PLAYS", "50+ YDS", "./td[24]/text()"),
                ("BIG PLAYS", "LNG", "./td[24]/text()"),
            ],
            "TE": [
                ("", "RANK", "./td[@class='player-rank']/text()"),
                ("", "PLAYER", "./td[@class='player-label player-label-report-page']/a/text()"),
                ("GAMES", "G", "./td[3]/text()"),
                ("RECEIVING", "REC", "./td[4]/text()"),
                ("RECEIVING", "YDS", "./td[5]/text()"),
                ("RECEIVING", "Y/R", "./td[6]/text()"),
                ("RECEIVING", "YBC", "./td[7]/text()"),
                ("RECEIVING", "YBC/R", "./td[8]/text()"),
                ("RECEIVING", "AIR", "./td[9]/text()"),
                ("RECEIVING", "AIR/R", "./td[10]/text()"),
                ("RECEIVING", "YAC", "./td[11]/text()"),
                ("RECEIVING", "YACON", "./td[12]/text()"),
                ("RECEIVING", "YACON/R", "./td[13]/text()"),
                ("RECEIVING", "BRKTKL", "./td[14]/text()"),
                ("TARGETS", "TGT", "./td[15]/text()"),
                ("TARGETS", "% TM", "./td[16]/text()"),
                ("TARGETS", "CATCHABLE", "./td[17]/text()"),
                ("TARGETS", "DROP", "./td[18]/text()"),
                ("TARGETS", "RZ TGT", "./td[19]/text()"),
                ("BIG PLAYS", "10+ YDS", "./td[20]/text()"),
                ("BIG PLAYS", "20+ YDS", "./td[21]/text()"),
                ("BIG PLAYS", "30+ YDS", "./td[22]/text()"),
                ("BIG PLAYS", "40+ YDS", "./td[23]/text()"),
                ("BIG PLAYS", "50+ YDS", "./td[24]/text()"),
                ("BIG PLAYS", "LNG", "./td[24]/text()"),
            ],
        }  
        data = []
        for row in rows:
            data.append(
                {
                    (pos[0], pos[1]): NFLStatsCrawler.clean_data(row.xpath(pos[2]).extract_first())
                    for pos in position_mapping[position]
                }
            )
        df = pd.DataFrame(data)
        df.columns = pd.MultiIndex.from_tuples(list(df.columns))
        df = df.dropna(axis="rows", how="all").reset_index(drop=True)
        return df

    def parse_nfl_stats(self, response: Any) -> None:
        """
        Parse the response and save the statistics to a CSV file.
        
        Parameters:
        - self (Self@NFLStatsCrawler): Class instance reference.
        - response (Any): The response object from Scrapy.

        Returns:
        None: Stores weekly statistics as CSV.  
        """
        crawler_logger.info(f"Processing {response.url}")
        position = response.url.split("advanced-stats-")[1].split(".php")[0].upper()
        year = response.url.split("year=")[1].split('&')[0]
        week = response.url.split('week=')[1].split('&')[0]

        rows = response.xpath("//tbody/tr")
        stats_df = self.create_dataframe(position, rows)

        stats_df.to_csv(f"advanced_stats/{position.lower()}_stats/{year}/nfl_stats_{position.lower()}_week_{week}.csv", index=False)
        crawler_logger.info(f"{year} data exported to nfl_stats_{position.lower()}_week_{week}.csv")

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


def trigger_process() -> None:
    """
    Trigger web crawler

    Parameters:
    - None

    Returns:
    None: Crawl fantasypros.com 
    """
    stats_crawler = NFLStatsCrawler
    logging.getLogger("scrapy").propagate = False
    process = CrawlerProcess(install_root_handler=False)
    process.crawl(stats_crawler)
    process.start()


if __name__ == "__main__":
    trigger_process()
