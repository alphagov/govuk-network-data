# # -*- coding: utf-8 -*-

import argparse
import itertools
import logging.config
import os
import pprint
from multiprocessing import Pool, cpu_count

import pandas as pd

from src.data.preprocess import *

AGGREGATE_COLUMNS = ['Languages', 'Locations', 'DeviceCategories',
                     'TrafficSources', 'TrafficMediums', 'NetworkLocations', 'sessionID',
                     'Times', 'Dates', 'Time_Spent', 'userID']
#
COUNTABLE_AGGREGATE_COLUMNS = ['Languages', 'Locations', 'DeviceCategories', 'TrafficSources',
                               'TrafficMediums', 'NetworkLocations']
SLICEABLE_COLUMNS = ['Occurrences', 'Languages', 'Locations', 'DeviceCategories', 'TrafficSources',
                     'TrafficMediums', 'NetworkLocations']
FEWER_THAN_CPU = False
DROP_INFREQ = False
MAX_DEPTH = -1
DEPTH_LIM = 1
NUM_CPU = 0
BATCH_SIZE = 3


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


def sequence_preprocess(user_journey_df):
    logger.info("BQ Sequence string to Page_Event_List...")
    user_journey_df['Page_Event_List'] = user_journey_df['Sequence'].map(bq_journey_to_pe_list)
    logger.info("Page_Event_List to Page_List...")
    user_journey_df['Page_List'] = user_journey_df['Page_Event_List'].map(lambda x: extract_pe_components(x, 1))
    # logger.info("Page_Event_List to Event_List...")
    # user_journey_df['Event_List'] = user_journey_df['Page_Event_List'].map(lambda x: extract_pe_components(x, 1))


def add_loop_columns(user_journey_df):
    """

    :param user_journey_df:
    :return:
    """
    logger.info("Collapsing loops...")
    user_journey_df['Page_List_NL'] = user_journey_df['Page_List'].map(collapse_loop)
    logger.info("To string...")
    user_journey_df['Page_Seq_NL'] = user_journey_df['Page_List_NL'].map(lambda x: ">>".join(x))
    logger.info("Aggregating de-looped journey occurrences...")
    user_journey_df['Occurrences_NL'] = user_journey_df.groupby('Page_Seq_NL')['Occurrences'].transform('sum')


def event_counters(user_journey_df):
    user_journey_df['num_event_cats'] = user_journey_df['Event_List'].map(count_event_cat)
    user_journey_df['Event_cats_agg'] = user_journey_df['Event_List'].map(aggregate_event_cat)
    user_journey_df['Event_cat_act_agg'] = user_journey_df['Event_List'].map(aggregate_event_cat_act)


def aggregate_dict(x):
    metadata_counter = {}
    for xs in x:
        for key, value in xs:
            if key not in metadata_counter:
                metadata_counter[key] = value
            else:
                metadata_counter[key] += value
    return list(metadata_counter.items())


def sliced_groupby_meta(df_slice, depth, multiple_dfs):
    agg = df_slice.columns[1]
    # One-off
    if depth == 0:
        df_slice[agg] = df_slice[agg].map(str_to_dict)
    if multiple_dfs:
        metadata_gpb = df_slice.groupby('Sequence')[agg].apply(aggregate_dict)
        # logger.info("Mapping {}, items: {}...".format(agg, len(metadata_gpb)))
        df_slice[agg] = df_slice['Sequence'].map(metadata_gpb)
        drop_duplicate_rows(df_slice, multiple_dfs)


def drop_duplicate_rows(df_slice, multiple_dfs):
    if multiple_dfs:
        bef = df_slice.shape[0]
        # logger.info("Current # of rows: {}. Dropping duplicate rows..".format(bef))
        df_slice.drop_duplicates(subset='Sequence', keep='first', inplace=True)
        after = df_slice.shape[0]
        logger.info("Dropped {} duplicated rows.".format(bef - after))


def sliced_mass_preprocess(code_df_slice, depth, multiple_dfs):
    code = code_df_slice[0]
    df_slice = code_df_slice[1]
    # print(df_slice.shape)
    # logger.info("Mass preprocessing {}...".format(df_slice.columns))
    # aggregate-able columns
    if df_slice.columns[1] in COUNTABLE_AGGREGATE_COLUMNS:
        logger.info("Aggregating {}...".format(df_slice.columns[1]))
        sliced_groupby_meta(df_slice, depth, multiple_dfs)
    elif "Occurrences" in df_slice.columns[1]:
        logger.info("Occurrences...")
        df_slice['Occurrences'] = df_slice.groupby('Sequence')['Occurrences'].transform('sum')
        drop_duplicate_rows(df_slice, multiple_dfs)
        # print(DEPTH_LIM, MAX_DEPTH, DROP_INFREQ)
        if (depth >= DEPTH_LIM and MAX_DEPTH >= 2) and DROP_INFREQ:
            bef = df_slice.shape[0]
            # logger.info("Current # of rows: {}. Dropping journeys occurring only once..".format(bef))
            df_slice = df_slice[df_slice.Occurrences > 1]
            after = df_slice.shape[0]
            logger.info("Dropped {} one-off rows.".format(bef - after))
    return code, df_slice


def read_file(filename):
    logging.info("Reading: {}".format(filename))
    return pd.read_csv(filename, compression="gzip")


def partition_list(x, chunks):
    if chunks > 0:
        initial = [list(xs) for xs in np.array_split(list(range(len(x))), chunks)]
        # print(initial)
        if len(initial) > 1 and not FEWER_THAN_CPU:
            initial = merge_small_partition(initial)
        return initial
    else:
        return [[0]]


def merge_small_partition(initial):
    to_merge = []
    for element in initial:
        if len(element) == 1:
            to_merge.append(element[0])
            initial.remove(element)
    if len(to_merge) >= 1:
        initial[-1].extend(to_merge)
    return initial


def del_var(x):
    if isinstance(x, list):
        for xs in x:
            del xs
    del x


def slice_dataframe(df):
    return [[df.columns.get_loc("Sequence"), df.columns.get_loc(col)] for col in SLICEABLE_COLUMNS if
            col in df.columns]


def slice_many_df(df_list, ordered=False):
    if not ordered:
        return [(i, df.iloc[:, ind].copy(deep=True)) for i, df in enumerate(df_list) for ind in slice_dataframe(df)]
    else:
        return [(i, df.iloc[:, ind].copy(deep=True)) for i, df in enumerate(df_list) for ind in slice_dataframe(df) if
                "Occurrences" in df.columns[ind]], [(i, df.iloc[:, ind].copy(deep=True)) for i, df in enumerate(df_list)
                                                    for ind in
                                                    slice_dataframe(df) if
                                                    "Occurrences" not in df.columns[ind]]


def merge_sliced_df(sliced_df_list, expected_size):
    """

    :param sliced_df_list:
    :param expected_size:
    :return:
    """
    final_list = [pd.DataFrame()] * expected_size
    # print([df.shape for i, df in sliced_df_list if i == 0])
    for i, df in sliced_df_list:
        # print(df.columns)
        if len(final_list[i]) == 0:
            # print("new")
            final_list[i] = df.copy(deep=True)
        else:
            # print("merge")
            final_list[i] = pd.merge(final_list[i], df, how='left', on='Sequence')
    return final_list


def conditional_pre_gpb_drop(df_occ_slice, df_meta_slice):
    for df_code_i, df_slice_i in df_occ_slice:
        for df_code_j, df_slice_j in df_meta_slice:
            if df_code_i == df_code_j:
                seq_occ = df_slice_i.Sequence.values
                # print("original", seq_occ.shape)
                # print("bef", df_slice_j.shape)
                df_slice_j.query("Sequence.isin(@seq_occ)", inplace=True)
                # print("after", df_slice_j.shape)
    return df_meta_slice


def distribute_df_slices(pool, dflist, chunks, depth=0, additional=None):
    """

    :param additional:
    :param pool:
    :param dflist:
    :param chunks:
    :param depth:
    :return:
    """
    if len(dflist) > 1:
        new_list = []
        partitions = partition_list(dflist, chunks)
        multi_dfs = []
        for i, index_list in enumerate(partitions):
            lst = [dflist[ind] for ind in index_list]
            multi_dfs.append(len(lst) > 1)
            # logger.info("Run: {} Num_of_df_to_merge: {}".format(i, len(lst)))
            pair_df = pd.concat(lst)
            del_var(lst)
            # logger.info("Size of merged dataframe: {}".format(pair_df.shape))
            new_list.append(pair_df)

        if additional is not None and depth > 0 and any(multi_dfs):
            print("Adding to ", multi_dfs[-1])
            new_list[-1].append(additional)

        slices_occ, slices_meta = slice_many_df(new_list, True)

        multi_dfs = [multi_dfs[i] for i, _ in slices_occ]
        slices_occ = pool.starmap(sliced_mass_preprocess, zip(slices_occ, itertools.repeat(depth),
                                                              multi_dfs))
        # print(DEPTH_LIM, MAX_DEPTH, DROP_INFREQ)
        if (depth >= DEPTH_LIM and MAX_DEPTH >= 2) and DROP_INFREQ:
            logger.info("conditional_pre_gpb_drop")
            slices_meta = conditional_pre_gpb_drop(slices_occ, slices_meta)

        multi_dfs = [multi_dfs[i] for i, _ in slices_meta]
        slices_meta = pool.starmap(sliced_mass_preprocess, zip(slices_meta, itertools.repeat(depth),
                                                               multi_dfs))

        new_list = slices_occ + slices_meta

        # print("THEM COLUMNS", [(new.columns, new.shape) for _, new in new_list])
        new_list = merge_sliced_df(new_list, len(partitions))
         # print("THEM COLUMNS", [(new.columns, new.shape) for new in new_list])
        return distribute_df_slices(pool, new_list, int(chunks / 2), depth + 1, additional)
    else:
        return dflist[0]


def compute_batches(files, batchsize):
    """

    :param files:
    :param batchsize:
    :return:
    """
    if len(files) > 4:
        return True, merge_small_partition([files[i:i + batchsize] for i in range(0, len(files), batchsize)])
    else:
        return False, files


def compute_initial_chunksize(number_of_files):
    if number_of_files > NUM_CPU:
        return int(number_of_files / 2)
    else:
        return number_of_files


def multiprocess_make(files, destination, final_filename):
    global FEWER_THAN_CPU, MAX_DEPTH, NUM_CPU, BATCH_SIZE
    batch_size = BATCH_SIZE
    NUM_CPU = cpu_count()
    batching, batches = compute_batches(files, batch_size)
    num_chunks = compute_initial_chunksize(len(files)) if not batching else compute_initial_chunksize(batch_size)

    FEWER_THAN_CPU = num_chunks == len(files) if not batching else batch_size + 1 <= NUM_CPU
    MAX_DEPTH = compute_max_depth(files, num_chunks, 0) if not batching else compute_max_depth([0] * (batch_size + 1),
                                                                                               num_chunks, 0)
    logger.info("BATCH_SIZE: {} MAX_DEPTH: {} NUM_FILES: {}".format(batch_size, MAX_DEPTH,len(files)))
    logger.info("Using {} workers...".format(NUM_CPU))
    pool = Pool(NUM_CPU)

    if not batching:
        print("No batching")
        df = process(files, pool, num_chunks)
    else:
        print("Batching")
        df = None
        for batch_num, batch in enumerate(batches):
            logging.info("Working on batch: {} with {} file(s)...".format(batch_num + 1, len(batch)))
            df = process(batch, pool, num_chunks, df)

    print(df.iloc[0])

    sequence_preprocess(df)
    add_loop_columns(df)

    logger.info("Shape: {}".format(df.shape))
    path_to_file = os.path.join(destination, final_filename)
    logger.info("Saving at: {}".format(path_to_file))
    df.to_csv(path_to_file, compression='gzip', index=False)

    pool.close()
    pool.join()

    logger.info("Multi done")


def process(files, pool, num_chunks, df_prev=None):
    logger.info("chunks: {} fewer: {} max_depth: {}".format(num_chunks, FEWER_THAN_CPU, MAX_DEPTH))
    logger.info("Number of files: {}".format(len(files)))
    logger.info("Multi start...")
    df_list = pool.map(read_file, files)

    if df_prev is not None:
        logger.info("Adding previous dataframe")
        # df_list.append(df_prev)

    logger.info("Distributing tasks...")
    df = distribute_df_slices(pool, df_list, num_chunks, 0, df_prev)
    return df


# def test():
#     print(split_event_cat_act(["ffyesno//yes", "NULL//NULL", "NULL//NULL"]))
#
#     print("\n=======\n")
#     eventlist = [("NULL", "NULL"), ("ffyesno", "yes"), ("ffyesno", "no"), ("ffman", "no"), ("ffyesno", "no")]
#
#     print(aggregate_event_cat_act(eventlist))
#     print(aggregate_event_cat(eventlist))
#
#     # eventlist2 = [("NULL", "NULL")]
#     eventlist2 = []
#     print(aggregate_event_cat_act(eventlist2))
#     print(aggregate_event_cat(eventlist2))
#
#     pelist = [("p1", "ffyesno//yes"), ("p1", "NULL//NULL"), ("p1", "NULL//NULL")]
#     pelist2 = [("p1", "ffyesno//yes"), ("p1", "NULL//NULL"), ("p1", "NULL//NULL"), ("p1", "NULL//jjj")]
#
#     print("\n=======\n")
#     print(reindex_pe_list(pelist))
#     print(reindex_pe_list(pelist2))
#
#     print("\n=======\n")
#     print(split_event_cat_act(extract_pe_components(pelist, 1)))
#
#     print(str_to_dict("1,2,3,4,3,4,2,23,2,32,3,23"))


def compute_max_depth(test_list, chunks, depth):
    partitions = partition_list(test_list, chunks)
    if len(test_list) > 1:
        new_lst = [0 for _ in partitions]
        return compute_max_depth(new_lst, (lambda x: int(x / 2) if int(x / 2) > 0 else 1)(chunks), depth + 1)
    else:
        return depth


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make datasets module')
    # parser.add_argument('source_dir', help='Source directory for input dataframe file(s).')
    # parser.add_argument('dest_dir', help='Specialized destination directory for output dataframe file.')
    # parser.add_argument('filename', help='Naming convention for resulting merged dataframe file.')
    parser.add_argument('--drop_one_offs', action='store_true')
    args = parser.parse_args()
    DROP_INFREQ = args.drop_one_offs
    print(DROP_INFREQ)

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
    name_stub = "user_network_paths_meta_2018-04-"
    to_load = [os.path.join(source_dir, file) for file in os.listdir(source_dir) if name_stub in file]

    # test()
    # tryout(to_load)
    # lst = list(range(60))
    # chunk = int(len(lst) / 2)
    # print(chunk)
    # print(list(range(len([1, 2]))))
    # print("max depth", compute_max_depth(lst, chunk, 0) - 1)
    # # list stuff
    # print(list(itertools.product(COUNTABLE_AGGREGATE_COLUMNS, [1, 2, 3, 4])))
    # lst = [1, 2]
    # chunk_size = int(len(lst) / 2)
    # print(len(lst), chunk_size)
    # print([list(x) for x in np.array_split(lst, 3)])
    # print(np.array_split(range(len(lst)), 2))
    # test = [1, 2, 3, 4, 6]
    # print("Final:", partition_list(test, 2))

    testout_dir = os.path.join(source_dir, "output")

    print("batches")
    pprint.pprint(compute_batches(to_load, 3))

    if not os.path.isdir(testout_dir):
        logging.info("Specified destination directory does not exist, creating...")
        os.mkdir(testout_dir)
    multiprocess_make(to_load, testout_dir, "merge_test_sliced_13days.csv.gz")
