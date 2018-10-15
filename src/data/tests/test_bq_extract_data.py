# to get correct relative path, run the following command from ./src/data/
# python -m pytest tests/
import bq_extract_data

def test_find_query():
    assert bq_extract_data.find_query("test_bq_extract_data.py", "./tests") == "./tests/test_bq_extract_data.py"
