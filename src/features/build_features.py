from collections import Counter


def has_loop(page_list):
    """
    Check if a list of page hits contains an adjacent page loop (A >> A >> B) == True.
    :param page_list: list of page hits derived from BQ user journey
    :return: True if there is a loop
    """
    return any(i == j for i, j in zip(page_list, page_list[1:]))


def has_repetition(page_list):
    """
    Check if a list of page hits contains a page repetition (A >> B >> A) == True.
    Run on journeys with collapsed loops so stuff like A >> A >> B are not captured as a repetition.
    Similar to cycles/triangles, but from a flat perspective.
    :param page_list: list of page hits derived from BQ user journey
    :return: True if there is a repetition
    """
    return len(set(page_list)) != len(page_list)


# Counters for events
def count_event_cat(event_list):
    """
    TODO: possibly remove
    Count different event categories present in an event_list. Includes "NULL" events coming from page
    hits for the sake of completeness. Does not include frequency.
    :param event_list: list of event tuples (eventCategory,eventAction)
    :return: number of different eventCategories present
    """
    return len(set([cat for cat, _ in event_list]))


def count_event_act(event_list, category, action):
    """
    TODO: possibly remove
    Count number of specific eventActions given a specific eventCategory
    :param event_list: list of event tuples (eventCategory,eventAction)
    :param category: target eventCategory
    :param action: target eventAction
    :return: count
    """
    return [action for cat, action in event_list if cat == category].count(action)


def aggregate_event_cat(event_list):
    """
    Return a dictionary-like list of eventCategory frequency counts.
    :param event_list: list of event tuples (eventCategory,eventAction)
    :return: dict-like list of frequencies [(eventCat1, freq_1),(eventCat2, freq_2),...]
    """
    return list(Counter([cat for cat, _ in event_list]).items())


def aggregate_event_cat_act(event_list):
    """
    Return a dictionary-like list of (eventCategory,eventAction) frequency counts.
    :param event_list: list of event tuples (eventCategory,eventAction)
    :return: dict-like list of frequencies [((eventCat1,eventAction1) freq_1),((eventCat1,eventAction2) freq_2),...]
    """
    return list(Counter([(cat, act) for cat, act in event_list]).items())
