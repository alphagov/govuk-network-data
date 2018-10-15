# # -*- coding: utf-8 -*-

import itertools
import logging.config
import os
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from src.data.preprocess import *

AGGREGATE_COLUMNS = ['Languages', 'Locations', 'DeviceCategories',
                     'TrafficSources', 'TrafficMediums', 'NetworkLocations', 'sessionID',
                     'Times', 'Dates', 'Time_Spent', 'userID']

# 'TrafficSources',
# 'Locations'
# , 'DeviceCategories'
COUNTABLE_AGGREGATE_COLUMNS = ['Languages', 'Locations', 'DeviceCategories', 'TrafficSources',
                               'TrafficMediums', 'NetworkLocations']

FEWER_THAN_CPU = False


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


def sequence_preprocess(df):
    df['Page_Event_List'] = df['Sequence'].map(bq_journey_to_pe_list)
    df['Page_List'] = df['Page_Event_List'].map(lambda x: extract_pe_components(x, 0))
    df['Event_List'] = df['Page_Event_List'].map(lambda x: extract_pe_components(x, 1))


def event_counters(df):
    df['num_event_cats'] = df['Event_List'].map(count_event_cat)
    df['Event_cats_agg'] = df['Event_List'].map(aggregate_event_cat)
    df['Event_cat_act_agg'] = df['Event_List'].map(aggregate_event_cat_act)


# def multi_str_to_dict(df):
#     for agg in AGGREGATE_COLUMNS:
#         if agg in df.columns:
#             df[agg] = df[agg].map(str_to_dict())


# def gpb(agg, df):
#     if agg in df.columns:
#         logger.info("Aggregating {}...".format(agg))
#         metadata_gpb = df.groupby('Sequence')[agg].apply(list_to_dict)
#         df[agg] = df['Sequence'].map(metadata_gpb)


def aggregate_dict(x):
    metadata_counter = {}
    for xs in x:
        for key, value in xs:
            if key not in metadata_counter:
                metadata_counter[key] = value
            else:
                metadata_counter[key] += value
    return list(metadata_counter.items())


def groupby_meta(df, depth):
    for agg in COUNTABLE_AGGREGATE_COLUMNS:
        if agg in df.columns:
            metadata_gpb = {}
            logger.info("Aggregating {}...".format(agg))
            if depth > 0:
                metadata_gpb = df.groupby('Sequence')[agg].apply(aggregate_dict)
            else:
                metadata_gpb = df.groupby('Sequence')[agg].apply(list_to_dict)
            logger.info("Mapping {}, items: {}...".format(agg, len(metadata_gpb)))
            df[agg] = df['Sequence'].map(metadata_gpb)
    if depth > 0:
        logger.info("Drop duplicates")
        print(df.shape)
        df.drop_duplicates(subset='Sequence', keep='first', inplace=True)
        print(df.shape)
    return df


# def mass_preprocess(df, depth, max_depth):
#     groupby_meta(df, depth)
#     if depth > 2:
#         df = df[df.Occurrences > 1]
#     elif depth == max_depth:
#         sequence_preprocess(df)
#         event_counters(df)
#
#     return df

#
# def merge(occurrence_limit, to_load, links):
#     user_journeys = pd.DataFrame()
#     logger.info("Limit: {}".format(occurrence_limit))
#     logger.info("Starting...")
#
#     for i, dataset in enumerate(to_load):
#         date_queried = dataset.split("/")[-1].split("_")[-1].replace(".csv.gz", "")
#         logger.info("RUN {} OF {} || DATE: {}".format(i + 1, len(to_load), date_queried))
#
#         user_journey_i = pd.read_csv(dataset, compression='gzip')
#
#         logger.info("Splitting BQ journeys to lists...")
#
#         # Drop before you start doing transformations
#         # before_drop_freq = user_journey_i.shape[0]
#         # logger.info("Before Occurrence drop: {}".format(before_drop_freq))
#         #
#         # user_journey_i = user_journey_i[user_journey_i.Occurrences_No_Loops > occurrence_limit]
#         #
#         # after_drop_freq = user_journey_i.shape[0]
#         # logger.info("After Occurrence drop: {}".format(after_drop_freq))
#         # logger.info("Percentage dropped: {}".format(((before_drop_freq - after_drop_freq) * 100) / before_drop_freq))
#
#         # Loop stuff
#         add_loop_columns(user_journey_i)
#
#         user_journey_i['Date_Queried'] = date_queried
#
#         print("String metadata to dict...")
#         multi_str_to_dict(COUNTABLE_AGGREGATE_COLUMNS, user_journey_i)
#
#         print("Merge into main dataframe...")
#         user_journeys = pd.concat([user_journeys, user_journey_i])
#
#         logger.info("Aggregating Occurrences of paths...")
#         user_journeys['Occurrences'] = user_journeys.groupby('Sequence')['Occurrences'].transform('sum')
#
#         logger.info("Aggregating individual metadata frequencies...")
#         for agg in AGGREGATE_COLUMNS:
#             print("Aggregating {}...".format(agg))
#             metadata_gpb = user_journeys.groupby('Sequence')[agg].apply(list_to_dict)
#             user_journeys[agg] = user_journeys['Sequence'].map(metadata_gpb)
#
#         logger.info("Before dropping dupes: {}".format(user_journeys.shape))
#         user_journeys.drop_duplicates(subset='Sequence', inplace=True)
#         logger.info("After dropping dupes: {}".format(user_journeys.shape))
#
#     logger.info("End")
#
#     return user_journeys


def read_file(filename):
    logging.info("Reading: {}".format(filename))
    # temp = pd.read_csv(filename, compression="gzip")
    # logger.info("Meta groupby time")
    # pooled_groupby(temp)
    return pd.read_csv(filename, compression="gzip")


def partition_list(x, chunks):
    # range(len(x))
    # print("=====")
    # print("original x",x)
    # print("x",x,"chunks",chunks)
    if chunks > 0:
        initial = [list(xs) for xs in np.array_split(list(range(len(x))), chunks)]
        # print("initial",initial)
        if len(initial) > 1 and not FEWER_THAN_CPU:
            # print("this",initial)
            # print(initial)
            to_merge = []
            for element in initial:
                if len(element) == 1:
                    to_merge.append(element[0])
                    initial.remove(element)
            # print(to_merge)
            # print(initial)

            # if len(to_merge) > 1:
            #     initial.append([m for m in to_merge])
            # ==1
            if len(to_merge) >= 1:
                # print(len(initial))
                initial[-1].extend(to_merge)
        return initial
    return [[0]]


def del_var(x):
    if isinstance(x, list):
        for xs in x:
            del xs
    del x


def distribute(pool, dflist, chunks, depth=0):
    # print(len(dflist))
    if len(dflist) > 1:
        new_list = []
        partitions = partition_list(dflist, chunks)
        for i, index_list in enumerate(partitions):
            lst = [dflist[ind] for ind in index_list]
            logger.info("Run: {} Num_of_pairs: {}".format(i, len(dflist)))
            pair_df = pd.concat(lst)
            del_var(lst)
            logger.info("Size of merged dataframe: {}".format(pair_df.shape))
            new_list.append(pair_df)
        new_list = pool.starmap(groupby_meta, zip(new_list, itertools.repeat(depth)))
        pool.join()
        # (lambda x: int(x / 2) if int(x / 2) > 0 else 1)(chunks)
        return distribute(pool, new_list, int(chunks / 2), depth + 1)
    else:
        print("list of things", [df.shape for df in dflist])
        print("0th", dflist[0].shape)
        return dflist[0]


def run_multi(files):
    global FEWER_THAN_CPU
    num_cpu = cpu_count()
    chunks = 2
    logger.info("Using {} workers...".format(num_cpu))
    pool = Pool(num_cpu)
    file_list = files
    logger.info("Number of files: {}".format(len(file_list)))
    logger.info("Multi start")
    df_list = pool.map(read_file, file_list)
    logger.info("Done reading.")
    logger.info("Let the fail begin")
    if len(df_list) < num_cpu:
        chunks = len(df_list)
        FEWER_THAN_CPU = True
    df = distribute(pool, df_list, chunks)
    # zip(l, itertools.repeat(o))
    # list(itertools.zip(COUNTABLE_AGGREGATE_COLUMNS, df))
    # pool.starmap(gpb, zip(COUNTABLE_AGGREGATE_COLUMNS, itertools.repeat(df)))

    print(df.iloc[0])

    pool.close()
    pool.join()
    #
    # print(len(df_list))
    # print(dir())
    logger.info("Multi done")
    logger.info("Shape: {}".format(df.shape))


def tryout(files):
    file_list = files
    df = pd.DataFrame()
    logger.info("simple start")
    for i, file in enumerate(file_list):
        logger.info("run {} file: {}".format(i, file))
        temp = read_file(file)
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


def compute_max_depth(test_list, chunks, depth):
    partitions = partition_list(test_list, chunks)
    # print(partitions)
    if len(test_list) > 1:
        # print("=========")
        # print("len",len(test_list))
        new_lst = []
        # print("testlist",test_list)
        for i, index_list in enumerate(partitions):
            # print(index_list)
            temp = [test_list[ind] for ind in index_list]
            new_lst.append("+".join([str(t) for t in (temp[0], temp[-1])]))
            # print("newlist",new_lst)
        # print("current depth", depth)
        return compute_max_depth(new_lst, (lambda x: int(x / 2) if int(x / 2) > 0 else 1)(chunks), depth + 1)
    else:
        return depth


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
    source_dir = os.path.join(DATA_DIR, "")
    dest_dir = os.path.join(DATA_DIR, "")
    #
    logger.info("Loading data")
    #
    # to_load = [os.path.join(source_dir, file) for file in os.listdir(source_dir)]
    #
    # pprint.pprint(to_load)
    source_dir = "/Users/felisialoukou/Documents/test1"
    name_stub = "user_network_paths_meta_2018-04-0"
    to_load = [os.path.join(source_dir, file) for file in os.listdir(source_dir) if name_stub in file]
    # test()
    # tryout(to_load)
    run_multi(to_load)
    # lst = list(range(60))
    # chunk = int(len(lst)/2)
    # print(chunk)
    # print(list(range(len([1, 2]))))
    # print("max depth",compute_max_depth(lst,chunk,0)-1)

    # list stuff
    # print(list(itertools.product(COUNTABLE_AGGREGATE_COLUMNS,[1,2,3,4])))
    # lst = [1,2]
    # chunk_size = int(len(lst)/2)
    # print(len(lst),chunk_size)
    # print(list(chunks(lst,chunk_size)))
    # print([list(x) for x in np.array_split(lst,3)])
    # print(np.array_split(range(len(lst)), 2))
    # print(list(split(lst,2)))
    # test = [1,2,3,4,6]
    # print("Final:",partition_list(test, 2))
