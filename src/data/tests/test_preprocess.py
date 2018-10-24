# to get correct relative path, run the following command from ./src/data/
# python3 -m pytest tests/
import preprocess

def test_bq_journey_to_pe_list():
    assert preprocess.bq_journey_to_pe_list("page1<<eventCategory1<:<eventAction1>>page2<<eventCategory2<:<eventAction2>>") == [('page1', 'eventCategory1<:<eventAction1'),
 ('page2', 'eventCategory2<:<eventAction2')]


def test_split_event():
    assert preprocess.split_event("eventCategory<:<eventAction") == ('eventCategory', 'eventAction')
    assert preprocess.split_event("yesNoFeedbackForm<:<ffYesClick") == ("yesNoFeedbackForm", "ffYesClick")


def test_extract_pe_components():
    # get event category and the type of action taken when parameter i = 1
    assert preprocess.extract_pe_components([('page1', 'eventCategory1<:<eventAction1'), ('page2', 'eventCategory2<:<eventAction2'), ('page3', 'eventCategory2<:<eventAction1')], 1) == [('eventCategory1', 'eventAction1'),
 ('eventCategory2', 'eventAction2'),
 ('eventCategory2', 'eventAction1')]
    # should this return an empty list? Potential bug?
    assert preprocess.extract_pe_components([('page1', 'eventCategory1<:<eventAction1'), ('page2', 'eventCategory2<:<eventAction2'), ('page3', 'eventCategory2<:<eventAction1')], i = 0) == []

def test_collapse_loop():
    assert preprocess.collapse_loop(["page1","page1", "page2", "page3", "page1"]) == ["page1", "page2", "page3", "page1"]
    assert preprocess.collapse_loop(["page1","page1", "page2", "page1", "page1"]) == ["page1", "page2", "page1"]

def test_start_end_page():
    assert preprocess.start_end_page(["page1","page1", "page2", "page3", "page1"]) == ("page1", "page1")
    # is this intended? we would of dropped one page journies?
    assert preprocess.start_end_page(["page1"]) == ("page1", "page1")

