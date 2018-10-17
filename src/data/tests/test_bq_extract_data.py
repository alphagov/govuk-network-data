# to get correct relative path, run the following command from ./src/data/
# python -m pytest tests/
import bq_extract_data

def test_find_query():
    assert bq_extract_data.find_query("test_bq_extract_data.py", "./tests") == "./tests/test_bq_extract_data.py"


# test removing linebreaks from sql query file
# add space for line breaks
def test_read_query():
    assert bq_extract_data.read_query("./tests/test.sql") == "SELECT * FROM tables WHERE thing < 5"
    # handles indent as represented by two-spaces
    assert bq_extract_data.read_query("./tests/query.sql") == "SELECT * FROM TABLE_DATE_RANGE([govuk-bigquery-analytics:1337.ga_sessions_],     TIME_STAMP))     WHERE PageSeq_Length > 1"

def test_change_timestamp():
    """
    Unit test for change_timestamp. Tests for both "standard" and "legacy" SQL timestamp differences.
    """
    # standard
    assert bq_extract_data.change_timestamp(x =  "SELECT * FROM TABLE_DATE_RANGE([govuk-bigquery-analytics:1337.ga_sessions_], TIME_STAMP)) WHERE PageSeq_Length > 1", date = "2018-12-31", dialect = "standard") == 'SELECT * FROM TABLE_DATE_RANGE([govuk-bigquery-analytics:1337.ga_sessions_], 20181231)) WHERE PageSeq_Length > 1'
    # legacy
    assert bq_extract_data.change_timestamp(x =  "SELECT * FROM TABLE_DATE_RANGE([govuk-bigquery-analytics:1337.ga_sessions_], TIME_STAMP)) WHERE PageSeq_Length > 1", date = "2018-12-31", dialect = "legacy") == 'SELECT * FROM TABLE_DATE_RANGE([govuk-bigquery-analytics:1337.ga_sessions_], TIMESTAMP("2018-12-31"), TIMESTAMP("2018-12-31")))) WHERE PageSeq_Length > 1'
    # standard, input x with read_query output
    assert bq_extract_data.change_timestamp(x = bq_extract_data.read_query("./tests/query.sql"), date = "2018-12-31", dialect =  "standard") == 'SELECT * FROM TABLE_DATE_RANGE([govuk-bigquery-analytics:1337.ga_sessions_],     20181231))     WHERE PageSeq_Length > 1'
