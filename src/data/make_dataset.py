# # -*- coding: utf-8 -*-

import itertools
import logging.config
import os
from multiprocessing import Pool, cpu_count

import pandas as pd

from src.data.preprocess import *

AGGREGATE_COLUMNS = ['Languages', 'Locations', 'DeviceCategories',
                     'TrafficSources', 'TrafficMediums', 'NetworkLocations', 'sessionID',
                     'Times', 'Dates', 'Time_Spent', 'userID']
# 'Locations', 'DeviceCategories', 'TrafficSources',
COUNTABLE_AGGREGATE_COLUMNS = ['Languages',
                               'TrafficMediums', 'NetworkLocations']

FEWER_THAN_CPU = False
MAX_DEPTH = -1


# Transform metadata lists to dictionary aggregates
def list_to_dict(metadata_list):
    return list(Counter([xs for xs in metadata_list]).items())


def str_to_dict(metadata_str):
    # print(metadata_str)
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


def add_loop_columns(user_journey_df):
    """

    :param user_journey_df:
    :return:
    """
    logger.info("Collapsing loops...")
    user_journey_df['Sequence_List_No_Loops'] = user_journey_df['Sequence_List'].map(collapse_loop)
    # logger.info("Has loop...")
    # user_journey['Has_Loop'] = user_journey['Sequence_List'].map(has_loop)
    # logger.info("To string...")
    # user_journey['Sequence_No_Loops'] = user_journey['Sequence_List_No_Loops'].map(list_to_path_string)
    logger.info("Aggregating de-looped journey occurrences...")
    user_journey_df['Occurrences_No_Loop'] = user_journey_df.groupby('Sequence_No_Loop')['Occurrences'].transform('sum')


def sequence_preprocess(user_journey_df):
    user_journey_df['Page_Event_List'] = user_journey_df['Sequence'].map(bq_journey_to_pe_list)
    user_journey_df['Page_List'] = user_journey_df['Page_Event_List'].map(lambda x: extract_pe_components(x, 1))
    # user_journey_df['Event_List'] = user_journey_df['Page_Event_List'].map(lambda x: extract_pe_components(x, 1))


def event_counters(user_journey_df):
    user_journey_df['num_event_cats'] = user_journey_df['Event_List'].map(count_event_cat)
    user_journey_df['Event_cats_agg'] = user_journey_df['Event_List'].map(aggregate_event_cat)
    user_journey_df['Event_cat_act_agg'] = user_journey_df['Event_List'].map(aggregate_event_cat_act)


# def multi_str_to_dict(df):
#     for agg in AGGREGATE_COLUMNS:
#         if agg in df.columns:
#             df[agg] = df[agg].map(str_to_dict())

# TODO one day
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


def groupby_meta(df, depth, multiple_dfs):
    for agg in COUNTABLE_AGGREGATE_COLUMNS:
        if agg in df.columns:
            logger.info("Aggregating {}...".format(agg))

            if depth == 0:
                df[agg] = df[agg].map(str_to_dict)
            if multiple_dfs:
                metadata_gpb = df.groupby('Sequence')[agg].apply(aggregate_dict)
                logger.info("Mapping {}, items: {}...".format(agg, len(metadata_gpb)))
                df[agg] = df['Sequence'].map(metadata_gpb)
    #
    # return df


# TODO
def mass_preprocess(user_journey_df, depth, multiple_dfs, num_dfs):
    logger.info("Mass preprocessing...")
    logger.info("Current depth: {} Number of dataframes: {}".format(depth, num_dfs))
    groupby_meta(user_journey_df, depth, multiple_dfs)
    # print(user_journey_df.Languages.iloc[0])
    logger.info("Occurrences...".format(depth, num_dfs))
    user_journey_df['Occurrences'] = user_journey_df.groupby('Sequence')['Occurrences'].transform('sum')
    if multiple_dfs:
        logger.info("Drop duplicates")
        # print(user_journey_df.shape)
        user_journey_df.drop_duplicates(subset='Sequence', keep='first', inplace=True)
        # print(user_journey_df.shape)
    if depth > 2 and MAX_DEPTH > 2:
        print("Dropping some rows...")
        user_journey_df = user_journey_df[user_journey_df.Occurrences > 1]
    if depth == MAX_DEPTH - 1:
        sequence_preprocess(user_journey_df)
        # event_counters(df)
    return user_journey_df


def read_file(filename):
    logging.info("Reading: {}".format(filename))
    return pd.read_csv(filename, compression="gzip")


def partition_list(x, chunks):
    if chunks > 0:
        initial = [list(xs) for xs in np.array_split(list(range(len(x))), chunks)]
        # print(initial)
        if len(initial) > 1 and not FEWER_THAN_CPU:
            to_merge = []
            for element in initial:
                if len(element) == 1:
                    to_merge.append(element[0])
                    initial.remove(element)
            if len(to_merge) >= 1:
                initial[-1].extend(to_merge)
        return initial
    else:
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
            multiple_df = len(lst) > 1
            logger.info("Run: {} Num_of_df_to_merge: {}".format(i, len(lst)))
            pair_df = pd.concat(lst)
            del_var(lst)
            logger.info("Size of merged dataframe: {}".format(pair_df.shape))
            new_list.append(pair_df)
        new_list = pool.starmap(mass_preprocess, zip(new_list, itertools.repeat(depth), itertools.repeat(multiple_df),
                                                     itertools.repeat(len(lst))))
        return distribute(pool, new_list, int(chunks / 2), depth + 1)
    else:
        print("list of things", [df.shape for df in dflist])
        print("0th", dflist[0].shape)
        return dflist[0]


def run_multi(files, destination, final_filename):
    global FEWER_THAN_CPU
    global MAX_DEPTH
    num_cpu = cpu_count()
    num_chunks = (lambda x: int(len(x) / 2) if len(x) > num_cpu else len(x))(files)
    FEWER_THAN_CPU = num_chunks == len(files)
    MAX_DEPTH = compute_max_depth(files, num_chunks, 0)

    logger.info("chunks {} fewer {} max_depth".format(num_chunks,FEWER_THAN_CPU, MAX_DEPTH))
    logger.info("Number of files: {}".format(len(files)))
    logger.info("Using {} workers...".format(num_cpu))
    pool = Pool(num_cpu)

    logger.info("Multi start...")
    df_list = pool.map(read_file, files)
    logger.info("Distributing tasks...")

    df = distribute(pool, df_list, num_chunks)

    print(df.iloc[0])

    logger.info("Shape: {}".format(df.shape))
    path_to_file = os.path.join(destination, final_filename)
    logger.info("Saving at: {}".format(path_to_file))
    df.to_csv(path_to_file, compression='gzip', index=False)

    pool.close()
    pool.join()
    #
    # print(len(df_list))
    # print(dir())
    logger.info("Multi done")


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
    if len(test_list) > 1:
        new_lst = [0 for _ in partitions]
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
    DOCUMENTS = os.getenv("DOCUMENTS")
    source_dir = os.path.join(DATA_DIR, "")
    dest_dir = os.path.join(DATA_DIR, "")

    logger.info("Loading data")

    source_dir = os.path.join(DOCUMENTS, "test1")
    name_stub = "user_network_paths_meta_2018-04-0"
    to_load = [os.path.join(source_dir, file) for file in os.listdir(source_dir) if name_stub in file]

    test()
    # tryout(to_load)

    lst = list(range(60))
    chunk = int(len(lst) / 2)
    print(chunk)
    print(list(range(len([1, 2]))))
    print("max depth", compute_max_depth(lst, chunk, 0) - 1)

    # list stuff
    print(list(itertools.product(COUNTABLE_AGGREGATE_COLUMNS, [1, 2, 3, 4])))
    lst = [1, 2]
    chunk_size = int(len(lst) / 2)
    print(len(lst), chunk_size)
    print([list(x) for x in np.array_split(lst, 3)])
    print(np.array_split(range(len(lst)), 2))
    test = [1, 2, 3, 4, 6]
    print("Final:", partition_list(test, 2))

    # num_cpu = cpu_count()
    # for i in range(0, 20, 1):
    #     test_list = list(range(i))
    #     chunks = (lambda x: int(len(x) / 2) if len(x) < num_cpu else len(x))(test_list)
    #     FEWER_THAN_CPU = chunks == len(test_list)
    #     print("List size", i, "Max depth:", compute_max_depth(test_list, chunks, 0))

    # Test multiprocessing
    # If dest_dir doesn't exist, create it.
    testout_dir = os.path.join(source_dir, "output")
    if not os.path.isdir(testout_dir):
        logging.info("Specified destination directory does not exist, creating...")
        os.mkdir(testout_dir)
    run_multi(to_load, testout_dir, "merge_test_1.csv.gz")
