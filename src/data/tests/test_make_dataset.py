# to get correct relative path, run the following command from ./src/data/
# python3 -m pytest tests/
import make_dataset
import pandas as pd

def test_list_to_dict():
    assert make_dataset.list_to_dict(['Desktop', 'Tablet', 'Mobile', 'Desktop', 'Mobile', 'Desktop']) == [('Desktop', 3), ('Tablet', 1), ('Mobile', 2)]

def test_str_to_dict():
    assert make_dataset.str_to_dict("Mobile,Desktop,Mobile") == [("Mobile", 2),("Desktop", 1)]

def test_aggregate_dict():
    assert make_dataset.aggregate_dict([[("Desktop", 3), ("Tablet", 1), ("Mobile", 2)] + [("Desktop", 3), ("Tablet", 1), ("Mobile", 2)]]) == [('Desktop', 6), ('Tablet', 2), ('Mobile', 4)]


# DATA PIPELINE
# read some test data in
def test_data_exists():
    user_journey_df = None
    user_journey_df = pd.read_pickle("./tests/user_journey_df.pkl")
    assert user_journey_df is not None
    assert user_journey_df.shape == (100, 7)

