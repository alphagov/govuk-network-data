# # -*- coding: utf-8 -*-

import logging.config
import os
import re
from collections import Counter
import numpy as np


# Transform raw SQL BigQuery string to list of page/event tuples
def clean_tuple(x):
    return [re.sub(r":|\"|\'", "", xs) for xs in x]


def bq_journey_to_pe_list(bq_journey_string):
    """

    :param bq_journey_string:
    :return:
    """
    journey_list = []
    for hit in bq_journey_string.split(">>"):
        page_event_tup = clean_tuple(hit.split("::"))
        if len(page_event_tup) == 2:
            journey_list.append(tuple(page_event_tup))
        else:
            journey_list.append(("::".join(page_event_tup[:-1]), page_event_tup[-1]))
    return journey_list


def reindex_pe_list(page_event_list):
    """
    Reindex and de-loop page_event_list if necessary. Used when absolute hit position within journey
    needs to be evaluated.
    If that's the case, page_list and event_list generators should be run based on this list, not
    page_event_list itself.
    :param page_event_list:
    :return:
    """
    if len(page_event_list) > 0:
        position_dict = [(0, page_event_list[0])]
        for i, (page, event) in enumerate(page_event_list[1:]):
            # print(i)
            if page != page_event_list[i][0]:
                index = position_dict[-1][0]
                position_dict.append((index + 1, (page, event)))
            elif page == page_event_list[i][0] and (event != position_dict[-1][1][1]):
                position_dict.append((position_dict[-1][0], (page, event)))
        return position_dict
    return np.NaN


def split_event_cat_act(event_list):
    """

    :param event_list:
    :return:
    """
    return [tuple(event.split("//")) for event in event_list]


def extract_pe_components(page_event_list, i):
    """

    :param page_event_list:
    :param i:
    :return:
    """
    return [page_event[i] for page_event in page_event_list]


# Counts for events
def count_event_categories(event_list):
    """

    :param event_list:
    :return:
    """
    return len(set([cat for cat, _ in event_list]))


def count_event_actions(event_list, category, action):
    """

    :param event_list:
    :param action:
    :param category:
    :return:
    """
    return [action for cat, action in event_list if cat == category].count(action)


def aggregate_event_categories(event_list):
    """

    :param event_list:
    :return:
    """
    return [(key, value) for key, value in Counter([cat for cat, _ in event_list]).items()]


def aggregate_event_actions(event_list):
    """

    :param event_list:
    :return:
    """
    return list(Counter([(cat, act) for cat, act in event_list]).items())


# Transform metadata lists to dictionary aggregates
def list_to_dict(metadata_list):
    # local_dict = Counter()
    # for xs in x:
    #     local_dict[xs] += 1
    # return list(local_dict.items())
    return list(Counter([xs for xs in metadata_list]).items())


def str_to_dict(metadata_str):
    return list_to_dict(metadata_str.split(','))


# TODO: needs more work, right now it is dependant on hardcoded df column specification
def zip_aggregate_metadata(user_journey_df):
    """

    :param user_journey_df:
    :return:
    """
    col = []
    for tup in user_journey_df.itertuples():
        locs = tup.Locations.split(',')
        langs = tup.Languages.split(',')
        devs = tup.DeviceCategories.split(',')
        zipped_meta_counter = Counter()
        for loc, lang, dev in zip(locs, langs, devs):
            zipped_meta_counter[(loc, lang, dev)] += 1
        col.append(list(zipped_meta_counter.items()))

    user_journey_df['AggMeta'] = col


# Loop-related functions
def has_loop(page_list):
    """

    :param page_list:
    :return:
    """
    return any(i == j for i, j in zip(page_list, page_list[1:]))


def collapse_loop(page_list):
    """

    :param page_list:
    :return:
    """
    return [node for i, node in enumerate(page_list) if i == 0 or node != page_list[i - 1]]


def add_loop_columns(user_journey):
    """

    :param user_journey:
    :return:
    """
    logger.info("Collapsing loops...")
    user_journey['Sequence_List_No_Loops'] = user_journey['Sequence_List'].map(collapse_loop)
    # logger.info("Has loop...")
    # user_journey['Has_Loop'] = user_journey['Sequence_List'].map(has_loop)
    # logger.info("To string...")
    # user_journey['Sequence_No_Loops'] = user_journey['Sequence_List_No_Loops'].map(list_to_path_string)
    logger.info("Aggregating de-looped journey occurrences...")
    user_journey['Occurrences_No_Loop'] = user_journey.groupby('Sequence_No_Loop') \
        ['Occurrences'].transform('sum')


# repetitions
def has_repetition(page_list):
    """
    Check if a list of page hits contains a page repetition (A >> B >> A) == True
    Run on journeys with collapsed loops so stuff like A >> A >> B are not captured as a repetition
    :param page_list: list of page hits derived from BQ user journey
    :return: True if there is a repetition
    """
    return len(set(page_list)) != len(page_list)


# Network things, should probably be moved somewhere else
def start_end_page(page_list):
    """

    :param page_list:
    :return:
    """
    return page_list[0], page_list[-1]


def subpaths_from_list(page_list):
    """

    :param page_list:
    :return:
    """
    return [[page, page_list[i + 1]] for i, page in enumerate(page_list) if i < len(page_list) - 1]


def start_page(page_list):
    return page_list[0]


def start_end_subpath_list(subpath_list):
    return subpath_list[0][0], subpath_list[-1][-1]


def test():
    print(split_event_cat_act(["ffyesno//yes", "NULL//NULL", "NULL//NULL"]))

    print("\n=======\n")
    eventlist = [("NULL", "NULL"), ("ffyesno", "yes"), ("ffyesno", "no"), ("ffman", "no"), ("ffyesno", "no")]

    print(aggregate_event_actions(eventlist))
    print(aggregate_event_categories(eventlist))

    # eventlist2 = [("NULL", "NULL")]
    eventlist2 = []
    print(aggregate_event_actions(eventlist2))
    print(aggregate_event_categories(eventlist2))

    pelist = [("p1", "ffyesno//yes"), ("p1", "NULL//NULL"), ("p1", "NULL//NULL")]
    pelist2 = [("p1", "ffyesno//yes"), ("p1", "NULL//NULL"), ("p1", "NULL//NULL"), ("p1", "NULL//jjj")]

    print("\n=======\n")
    print(reindex_pe_list(pelist))
    print(reindex_pe_list(pelist2))

    print("\n=======\n")
    print(split_event_cat_act(extract_pe_components(pelist, 1)))

    print(str_to_dict("1,2,3,4,3,4,2,23,2,32,3,23"))


if __name__ == "__main__":
    test()
    # parser = argparse.ArgumentParser(description='BigQuery extractor module')
    # parser.add_argument('source_dir', help='Source directory for input dataframe file(s).')
    # parser.add_argument('dest_dir', help='Specialized destination directory for output dataframe file.')
    # parser.add_argument('filename', help='Naming convention for resulting merged dataframe file.')
    # args = parser.parse_args()
    #
    # Logger setup
    LOGGING_CONFIG = os.getenv("LOGGING_CONFIG")
    logging.config.fileConfig(LOGGING_CONFIG)
    logger = logging.getLogger('make_dataset')
    #
    # DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data')
    # source_dir = ""
    # dest_dir = ""
    #
    # logger.info("Loading data")
    #
    # to_load = [os.path.join(source_dir, file) for file in os.listdir(source_dir)]
    #
    # pprint.pprint(to_load)
