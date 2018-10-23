# to get correct relative path, run the following command from ./src/data/
# python3 -m pytest tests/
import preprocess

def test_bq_journey_to_pe_list():
    assert preprocess.bq_journey_to_pe_list("page1<<eventCategory1<:<eventAction1>>page2<<eventCategory2<:<eventAction2>>") == [('page1', 'eventCategory1<:<eventAction1'),
 ('page2', 'eventCategory2<:<eventAction2')]


