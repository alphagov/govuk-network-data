# govuk-network-data
Data pipeline for extraction and preprocessing of BigQuery user journey data.

## Python version
Python 3.6.0

## Virtual environment

pip install -r requirements.txt

## Running a big query extract

python data/bq_extract_data.py start_date='2018-10-15' end_date='2018-10-15' dest_dir='data' filename='test' query='simple_test.sql'

## Creating a csv with each row a user journey with sessions rolled into it

For example is 3 users went A -> B -> C on different devices, then this would be a single row with each device listed in that column

