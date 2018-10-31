# to get correct relative path, run the following command from ./src/data/
# python -m pytest tests/
import bq_extract_data


def test_find_query():
    # only returns .sql files, addresses issue 10 somewhat
    assert bq_extract_data.find_query("test_bq_extract_data.py", "./tests") is None
    assert bq_extract_data.find_query("quer", "./tests") == "./tests/query.sql"
    # returns first file to match query_arg, bug or feature?
    assert bq_extract_data.find_query("", "./tests") == "./tests/test.sql"
    # potential bug spotted
    # assert bq_extract_data.find_query("query.sql", "./tests") == "./tests/query.sql"


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

# functional test
def test_find_read_change_timestamp_combined():
    """
    Combines the three functions above. A user provides an
    approximate name of the file in a given dir that holds their
    SQL query of interest. This is read in and converted to a string,
    replacing line breaks with spaces. This "SQL query" str
    then has its timestamps adjusted to the correct dialect
    and so that the correct table is read in BigQuery.
    One table per day.
    """
    assert bq_extract_data.change_timestamp(bq_extract_data.read_query(bq_extract_data.find_query("query", "./tests")),
                                            date = "2018-12-31", dialect =  "standard") == 'SELECT * FROM TABLE_DATE_RANGE([govuk-bigquery-analytics:1337.ga_sessions_],     20181231))     WHERE PageSeq_Length > 1'


