# to get correct relative path, run the following command from ./src/features/
# python -m pytest tests/
import build_features


def test_has_loop():
    assert build_features.has_loop(["page1", "page2", "page1"]) is False
    assert build_features.has_loop(["page1", "page2", "page2"]) is True


def test_has_repetition():
    assert build_features.has_repetition(["page1", "page2", "page3"]) is False
    # Yields true due to self-loop, should be run on collapsed-loop page lists
    assert build_features.has_repetition(["page1", "page1", "page1"]) is True
    assert build_features.has_repetition(["page1", "page2", "page3", "page1"]) is True
    assert build_features.has_repetition(["page2", "page3", "page2"]) is True


def test_count_event_cat():
    assert build_features.count_event_cat([('eventCategory1', 'eventAction1'),
                                           ('eventCategory2', 'eventAction2'),
                                           ('eventCategory2', 'eventAction1')]) == 2


def test_count_event_act():
    assert build_features.count_event_act([('eventCategory1', 'eventAction1'),
                                           ('eventCategory2', 'eventAction2'),
                                           ('eventCategory2', 'eventAction1')],
                                          category='eventCategory1', action='eventAction1') == 1


def test_aggregate_event_count():
    assert build_features.aggregate_event_cat([('eventCategory1', 'eventAction1'),
                                               ('eventCategory2', 'eventAction2'),
                                               ('eventCategory2', 'eventAction1')]) == \
           [('eventCategory1', 1), ('eventCategory2', 2)]


def test_aggregate_event_cat_act():
    assert build_features.aggregate_event_cat_act([('eventCategory1', 'eventAction1'),
                                                   ('eventCategory2', 'eventAction2'),
                                                   ('eventCategory2', 'eventAction1')]) == \
           [(('eventCategory1', 'eventAction1'), 1),
            (('eventCategory2', 'eventAction2'), 1),
            (('eventCategory2', 'eventAction1'), 1)]
