# NFL Fantasy Football 

A suite of tools for gathering, processing, and predicting NFL player statistics. The tools are designed to automate the collection of player stats, process and analyze them, and then train an LSTM (Long Short-Term Memory) model for fantasy football projections.

## Table of Contents

- [NFL Stats Crawler](#nfl-stats-crawler)
- [NFL Stats Processor](#nfl-stats-processor)
- [NFL Stats Projector](#nfl-stats-projector)
- [Requirements](#requirements)
- [Installation](#installation)

## NFL Stats Crawler
The **NFL Stats Crawler** is a web scraper built using the **Scrapy** framework. It is designed to collect advanced statistics for NFL players from the FantasyPros website. This tool automates the process of gathering key offensive statistics for players in the following positions:
- **Quarterbacks (QB)**: Passing yards, touchdowns, interceptions, rushing yards, rushing touchdowns, etc.
- **Running Backs (RB)**: Rushing yards, rushing touchdowns, receiving yards, receiving touchdowns, etc.
- **Wide Receivers (WR)**: Receptions, receiving yards, receiving touchdowns, rushing yards, etc.
- **Tight Ends (TE)**: Receptions, receiving yards, receiving touchdowns, etc.
- **Additional Info**: Found [here](data_ingestion/README.md).

## NFL Stats Processor
The **NFL Stats Processor** is a Python script designed to load, process, and analyze NFL player statistics over multiple seasons. The tool supports multiple offensive positions—Quarterbacks, Running Backs, Wide Receivers, and Tight Ends—aggregating data from various seasons to help identify trends and evaluate player performance. 
- **Additional Info**: Found [here](data_processing/README.md).

## NFL Stats Projector
The **NFL Stats Projector** is a Python script designed to fetch NFL player statistics and train/evaluate an LSTM (Long Short-Term Memory) model for fantasy football projections. The model is trained on historical data and can predict future fantasy points based on a player's previous performances.
- **Additional Info**: Found [here](data_modeling/README.md).

## Requirements
Ensure the packages and libraries listed in the `requirements.txt` are installed to run the tools. Several key dependencies include:
- **Scrapy**: Web scraping framework for collecting player stats from websites like FantasyPros.
- **Scikit-learn**: Machine learning utilities for model evaluation.
- **TensorFlow**: For building and training the LSTM model.
- **Keras**: High-level neural networks API for model construction and training.

## Installation
To install the necessary dependencies, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/NFL-stat-projections.git
    ```

2. Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # For Linux/macOS
    venv\Scripts\activate  # For Windows
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the tools:
    - For scraping player stats:
      ```bash
      python .\data_ingestion\web_crawler.py
      ```
    - For processing player stats:
      ```bash
      python .\data_processing\stat_processor.py
      ```
    - For training and projecting fantasy points:
      ```bash
      python .\data_modeling\model_builder.py
      ```

Once installed, you can start gathering, processing, and predicting NFL player statistics to optimize your fantasy football strategy.
