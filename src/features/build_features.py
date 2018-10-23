from collections import Counter


# Loop-related functions
def has_loop(page_list):
    """

    :param page_list:
    :return:
    """
    return any(i == j for i, j in zip(page_list, page_list[1:]))


# repetitions
def has_repetition(page_list):
    """
    Check if a list of page hits contains a page repetition (A >> B >> A) == True
    Run on journeys with collapsed loops so stuff like A >> A >> B are not captured as a repetition
    :param page_list: list of page hits derived from BQ user journey
    :return: True if there is a repetition
    """
    return len(set(page_list)) != len(page_list)


# Counts for events
def count_event_cat(event_list):
    """

    :param event_list:
    :return:
    """
    return len(set([cat for cat, _ in event_list]))


def count_event_act(event_list, category, action):
    """

    :param event_list:
    :param category:
    :param action:
    :return:
    """
    return [action for cat, action in event_list if cat == category].count(action)


def aggregate_event_cat(event_list):
    """

    :param event_list:
    :return:
    """
    return list(Counter([cat for cat, _ in event_list]).items())


def aggregate_event_cat_act(event_list):
    """

    :param event_list:
    :return:
    """
    return list(Counter([(cat, act) for cat, act in event_list]).items())
