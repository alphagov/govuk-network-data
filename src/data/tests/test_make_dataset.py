# to get correct relative path, run the following command from ./src/data/
# python3 -m pytest tests/
import make_dataset

def test_list_to_dict():
    assert make_dataset.list_to_dict(['Desktop', 'Tablet', 'Mobile', 'Desktop', 'Mobile', 'Desktop']) == [('Desktop', 3), ('Tablet', 1), ('Mobile', 2)]

def test_str_to_dict():
    assert make_dataset.str_to_dict("Mobile,Desktop,Mobile") == [("Mobile", 2),("Desktop", 1)]
