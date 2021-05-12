# to get correct relative path, run the following command from ./src/data/
# python3 -m pytest tests/
import make_dataset
import pandas as pd
import pandas as pd

def test_list_to_dict():
    assert make_dataset.list_to_dict(['Desktop', 'Tablet', 'Mobile', 'Desktop', 'Mobile', 'Desktop']) ==\
           [('Desktop', 3), ('Tablet', 1), ('Mobile', 2)]


def test_str_to_dict():
    assert make_dataset.str_to_dict("Mobile,Desktop,Mobile") ==\
           [("Mobile", 2),("Desktop", 1)]


def test_aggregate_dict():
    assert make_dataset.aggregate_dict([[("Desktop", 3), ("Tablet", 1), ("Mobile", 2)] +
                                        [("Desktop", 3), ("Tablet", 1), ("Mobile", 2)]]) ==\
           [('Desktop', 6), ('Tablet', 2), ('Mobile', 4)]


# DATA PIPELINE
# generate some test data in
user_journey_dict = {
     'Occurrences': [1, 12, 35],
     'Sequence': ["/page1<<PAGE<:<NULL<:<NULL", "/page2<<PAGE<:<NULL<:<NULL", "/page1<<PAGE<:<NULL<:<NULL<<other>>/page2<<EVENT<:<yesNoFeedbackForm<:<ffNoClick<<other>>/page2<<EVENT<:<yesNoFeedbackForm<:<Send Form<<other"],
     'PageSequence': ["/page1", "/page2", "/page1>>/page2>>/page2"]
}

user_journey_df = pd.DataFrame(user_journey_dict)

def test_data_exists():
    assert user_journey_df is not None
    assert user_journey_df.shape == (3, 3)
