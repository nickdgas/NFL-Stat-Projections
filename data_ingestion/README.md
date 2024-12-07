# NFL Stats Crawler

## Overview
The **NFL Stats Crawler** is a web scraper built using the Scrapy framework, designed to collect advanced statistics for NFL players from the FantasyPros website. The tool provides automated scraping of key offensive positions—Quarterbacks, Running Backs, Wide Receivers, and Tight Ends—and exports the gathered data into clean, structured CSV files for analysis.

## Key Features
- **Dynamic URL Generation**: Automatically constructs URLs for scraping based on the selected position, week, and year.
- **Data Cleaning and Formatting**: The scraper processes raw data by converting it into appropriate formats (e.g., integers, floats).
- **Multi-Position Scraping**: Supports stats collection for multiple offensive positions.
- **Error Logging**: Implements logging to track and report any issues encountered during web crawling.
- **CSV Export**: Efficiently exports scraped player statistics to CSV files.

## Requirements
- Refer to [requirements.txt](../requirements.txt) for the required packages and libraries
