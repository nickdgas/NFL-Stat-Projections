import re
import logging
import pendulum 
from scrapy.http import Request
from scrapy import Spider 
import pandas as pd
from scrapy.crawler import CrawlerProcess
from scrapy.spidermiddlewares.httperror import HttpError
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(message)s')
crawler_logger = logging.getLogger("NFL-Crawler")
crawler_logger.setLevel(logging.INFO)

class NFLStatsCrawler(Spider):
    name = 'nfl_stats_crawler'
    allowed_domains = ['fantasypros.com']

    @staticmethod
    def list_urls():
        """
        Retrieve list of URLs to be scraped
        """
        offensive_positions = ["QB", "RB", "WR", "TE"]
        return [f"https://www.fantasypros.com/nfl/advanced-stats-{pos.lower()}.php" for pos in offensive_positions]

    start_urls = list_urls()

    def start_requests(self):
        """
        Make requests to URLs and set callbacks for parsing and error logging
        """
        for url in self.start_urls:
            yield Request(
                url,
                callback=self.parse_nfl_stats,
                errback=self.log_errors,
            )

    @staticmethod
    def create_dataframe() -> pd.DataFrame:
        """
        Perform xpath queries for extracting data from URLs
        Store data into dataframe
        """
        data = []
        for row in rows:
            data.append({
                'RANK': row.xpath("").extract_first(),
                'PLAYER': row.xpath("").extract_first(),
                'G': row.xpath("").extract_first(),
                'COMP': row.xpath("").extract_first(),
                'ATT': row.xpath("").extract_first(),
                'PCT': row.xpath("").extract_first(),
                'YDS': row.xpath("").extract_first(),
                'Y/A': row.xpath("").extract_first(),
                'AIR': row.xpath("").extract_first(),
                'AIR/A': row.xpath("").extract_first(),
                '10+ YDS': row.xpath("").extract_first(),
                '20+ YDS': row.xpath("").extract_first(),
                '30+ YDS': row.xpath("").extract_first(),
                '40+ YDS': row.xpath("").extract_first(),
                '50+ YDS': row.xpath("").extract_first(),
                'PKT TIME': row.xpath("").extract_first(),
                'SACK': row.xpath("").extract_first(),
                'KNCK': row.xpath("").extract_first(),
                'HRRY': row.xpath("").extract_first(),
                'BLITZ': row.xpath("").extract_first(),
                'POOR': row.xpath("").extract_first(),
                'DROP': row.xpath("").extract_first(),
                'RZ ATT': row.xpath("").extract_first(),
                'RTG': row.xpath("").extract_first(),
            })
        df = pd.DataFrame(data)
        df = df.dropna(axis='rows', how='all').reset_index(drop=True)
        return df


    def parse_nfl_stats(self, response):
        """
        Parse response from request made to Domestic Daily URLs
        Store in DataFrame and output to excel
        """
        crawler_logger.info(f'Processing {response.url}')

    def log_errors(self, failure):
        """
        Log errors if http request fails
        """
        crawler_logger.error(repr(failure))
        if failure.check(HttpError):
            response = failure.value.response
            crawler_logger.error("HttpError on %s", response.url)

def trigger_process():
    stats_crawler = NFLStatsCrawler
    logging.getLogger('scrapy').propagate = False
    process = CrawlerProcess(install_root_handler=False)
    process.crawl(stats_crawler)
    process.start()

if __name__ == "__main__":
    trigger_process()