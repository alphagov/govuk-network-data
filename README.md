# govuk-network-data
Data pipeline for extraction and preprocessing of BigQuery user journey data.

## Python version
Python 3.6.0

## Virtual environment

```pip install -r requirements.txt```

## Where to put your BigQuery key

```mkdir key```

then put the json file in there

## Running a big query extract

```python src/data/bq_extract_data.py '2018-10-15' '2018-10-15' 'test_dir' 'test' 'simple_test.sql'```

## Creating a csv with each row a user journey with sessions rolled into it

For example is 3 users went A -> B -> C on different devices, then this would be a single row with each device listed in that column

