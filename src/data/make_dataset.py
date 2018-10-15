# # -*- coding: utf-8 -*-

import logging.config
import os
import re
from collections import Counter
import numpy as np
import pandas as pd

from multiprocessing import Pool, cpu_count


AGGREGATE_COLUMNS = ['Languages', 'Locations', 'DeviceCategories',
        'TrafficSources', 'TrafficMediums', 'NetworkLocations', 'sessionID',
        'Times', 'Dates', 'Time_Spent', 'userID']

COUNTABLE_AGGREGATE_COLUMNS  = ['Languages', 'Locations', 'DeviceCategories',
                  'TrafficSources', 'TrafficMediums', 'NetworkLocations']

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
    user_journey['Occurrences_No_Loop'] = user_journey.groupby('Sequence_No_Loop')['Occurrences'].transform('sum')


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


def sequence_preprocess(df):
    df['Page_Event_List'] = df['Sequence'].map(bq_journey_to_pe_list)
    df['Page_List'] = df['Page_Event_List'].map(lambda x: extract_pe_components(x, 0))
    df['Event_List'] = df['Page_Event_List'].map(lambda x: extract_pe_components(x, 1))


def event_counters(df):
    df['num_event_cats'] = df['Event_List'].map(count_event_cat)
    df['Event_cats_agg'] = df['Event_List'].map(aggregate_event_cat)
    df['Event_cat_act_agg'] = df['Event_List'].map(aggregate_event_cat_act)


def multi_str_to_dict(countables, df):
    for agg in countables:
        if agg in df.columns:
            df[agg] = df[agg].map(str_to_dict())


def groupby_meta(df):
    for agg in COUNTABLE_AGGREGATE_COLUMNS:
        if agg in df.columns:
            logger.info("Aggregating {}...".format(agg))
            metadata_gpb = df.groupby('Sequence')[agg].apply(list_to_dict)
            df[agg] = df['Sequence'].map(metadata_gpb)


def merge(occurrence_limit, to_load, links):
    user_journeys = pd.DataFrame()
    logger.info("Limit: {}".format(occurrence_limit))
    logger.info("Starting...")

    aggs = ['Languages', 'Locations', 'DeviceCategories',
            'TrafficSources', 'TrafficMediums', 'NetworkLocations', 'sessionID',
            'Times', 'Dates', 'Time_Spent', 'userID']

    countable_aggs = ['Languages', 'Locations', 'DeviceCategories',
            'TrafficSources', 'TrafficMediums', 'NetworkLocations']

    for i, dataset in enumerate(to_load):
        date_queried = dataset.split("/")[-1].split("_")[-1].replace(".csv.gz", "")
        logger.info("RUN {} OF {} || DATE: {}".format(i + 1, len(to_load), date_queried))

        user_journey_i = pd.read_csv(dataset, compression='gzip')

        logger.info("Splitting BQ journeys to lists...")


        # Drop before you start doing transformations
        # before_drop_freq = user_journey_i.shape[0]
        # logger.info("Before Occurrence drop: {}".format(before_drop_freq))
        #
        # user_journey_i = user_journey_i[user_journey_i.Occurrences_No_Loops > occurrence_limit]
        #
        # after_drop_freq = user_journey_i.shape[0]
        # logger.info("After Occurrence drop: {}".format(after_drop_freq))
        # logger.info("Percentage dropped: {}".format(((before_drop_freq - after_drop_freq) * 100) / before_drop_freq))

        # Loop stuff
        add_loop_columns(user_journey_i)

        user_journey_i['Date_Queried'] = date_queried

        print("String metadata to dict...")
        multi_str_to_dict(countable_aggs, user_journey_i)

        print("Merge into main dataframe...")
        user_journeys = pd.concat([user_journeys, user_journey_i])

        logger.info("Aggregating Occurrences of paths...")
        user_journeys['Occurrences'] = user_journeys.groupby('Sequence')['Occurrences'].transform('sum')


        logger.info("Aggregating individual metadata frequencies...")
        for agg in aggs:
            print("Aggregating {}...".format(agg))
            metadata_gpb = user_journeys.groupby('Sequence')[agg].apply(list_to_dict)
            user_journeys[agg] = user_journeys['Sequence'].map(metadata_gpb)

        logger.info("Before dropping dupes: {}".format(user_journeys.shape))
        user_journeys.drop_duplicates(subset='Sequence', inplace=True)
        logger.info("After dropping dupes: {}".format(user_journeys.shape))

    logger.info("End")

    return user_journeys


def reader(filename):
    print(filename)
    temp = pd.read_csv(filename, compression="gzip")
    logger.info("Meta groupby time")
    groupby_meta(temp)
    return temp


def tryout_multi(files):
    cpus = int(cpu_count())
    logger.info("Using {} cores...".format(cpus))
    pool = Pool(cpus)
    file_list = files
    logger.info("Number of files: {}".format(len(file_list)))
    logger.info("Multi start")
    df_list = pool.map(reader, file_list)
    logger.info("Done reading.")
    df = pd.concat(df_list)
    print(df.columns)
    pool.close()
    logger.info("Multi done")
    logger.info("Shape: {}".format(df.shape))


def tryout(files):
    file_list = files
    df = pd.DataFrame()
    logger.info("simple start")
    for i, file in enumerate(file_list):
        logger.info("run {} file: {}".format(i, file))
        temp = reader(file)
        df = pd.concat([df, temp])
    logger.info("simple done")
    logger.info("Shape: {}".format(df.shape))


def test():
    print(split_event_cat_act(["ffyesno//yes", "NULL//NULL", "NULL//NULL"]))

    print("\n=======\n")
    eventlist = [("NULL", "NULL"), ("ffyesno", "yes"), ("ffyesno", "no"), ("ffman", "no"), ("ffyesno", "no")]

    print(aggregate_event_cat_act(eventlist))
    print(aggregate_event_cat(eventlist))

    # eventlist2 = [("NULL", "NULL")]
    eventlist2 = []
    print(aggregate_event_cat_act(eventlist2))
    print(aggregate_event_cat(eventlist2))

    pelist = [("p1", "ffyesno//yes"), ("p1", "NULL//NULL"), ("p1", "NULL//NULL")]
    pelist2 = [("p1", "ffyesno//yes"), ("p1", "NULL//NULL"), ("p1", "NULL//NULL"), ("p1", "NULL//jjj")]

    print("\n=======\n")
    print(reindex_pe_list(pelist))
    print(reindex_pe_list(pelist2))

    print("\n=======\n")
    print(split_event_cat_act(extract_pe_components(pelist, 1)))

    print(str_to_dict("1,2,3,4,3,4,2,23,2,32,3,23"))


if __name__ == "__main__":

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
    DATA_DIR = os.getenv("DATA_DIR")
    source_dir = os.path.join(DATA_DIR,"")
    dest_dir = os.path.join(DATA_DIR,"")
    #
    logger.info("Loading data")
    #
    # to_load = [os.path.join(source_dir, file) for file in os.listdir(source_dir)]
    #
    # pprint.pprint(to_load)
    source_dir = "/Users/felisialoukou/Documents/govuk-networks/data/metadata_user_paths"
    name_stub = "user_network_paths_meta_2018-04-0"
    to_load = [os.path.join(source_dir, file) for file in os.listdir(source_dir) if name_stub in file]
    test()
    tryout(to_load)
    # tryout_multi(to_load)



