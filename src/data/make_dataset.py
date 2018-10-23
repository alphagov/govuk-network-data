# # -*- coding: utf-8 -*-

import argparse
import itertools
import logging.config
import os
import sys
from collections import Counter
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(src, "data"))
sys.path.append(os.path.join(src, "features"))
import preprocess as prep
import build_features as feat

# TODO: Integrate in the future
# AGGREGATE_COLUMNS = ['Languages', 'Locations', 'DeviceCategories',
#                      'TrafficSources', 'TrafficMediums', 'NetworkLocations', 'sessionID',
#                      'Times', 'Dates', 'Time_Spent', 'userID']
# TODO: Extend with more BigQuery fields. Pre-defined columns will be aggregated
COUNTABLE_AGGREGATE_COLUMNS = ['Languages', 'Locations', 'DeviceCategories', 'DeviceCategory', 'TrafficSources',
                               'TrafficMediums', 'NetworkLocations', 'Dates']
# TODO: Extend with more BigQuery fields. Pre-defined columns will be aggregated
SLICEABLE_COLUMNS = ['Occurrences', 'Languages', 'Locations', 'DeviceCategories', 'DeviceCategory', 'TrafficSources',
                     'TrafficMediums', 'NetworkLocations', 'Dates']
# Columns to drop post-internal processing if DROP_ONE_OFFS is true: these are initialized in order to set up
# "PageSequence" which is used for journey drops instead of "Sequence" which includes events, hence making journeys
# overall more infrequent.
DROPABLE_COLS = ['Page_Event_List', 'Page_List']
# Fewer files to process than available cpus.
FEWER_THAN_CPU = False
# Drop journeys occurring once (not in a day, multiple days, governed by DEPTH globals). If false, overrides depth
# globals and keeps journeys, resulting in massive dataframes (danger zone).
DROP_ONE_OFFS = False
# Drop journeys of length 1
DROP_ONES = False
# Keep only journeys of length 1
KEEP_ONES = False
# Maximum recursive depth for distribution function
MAX_DEPTH = -1
# Recursive depth limit for distribution function, so one-off journeys are drop in time.
DEPTH_LIM = 1
# Number of available CPUs, governs size of pool/number of daemon worker.
NUM_CPU = 0
# If there are many files to be merge, load in/preprocess in batches
BATCH_SIZE = 3
# A bit of a magic number, but limits dataframes that can be passed off to workers. If dataframe exceeds this size,
# switch to sequential execution.
ROW_LIMIT = 3000000


def list_to_dict(metadata_list):
    """
    Transform metadata lists to dictionary aggregates
    :param metadata_list:
    :return:
    """
    return list(Counter([xs for xs in metadata_list]).items())


def str_to_dict(metadata_str):
    """
    Transform metadata string eg mobile,desktop,mobile to [(mobile,2),(desktop,1)] dict-like
    list.
    :param metadata_str:
    :return: dict-like list of frequencies
    """
    # print(metadata_str)
    return list_to_dict(metadata_str.split(','))


def aggregate_dict(metadata_list):
    """
    Aggregate over multiple metadata frequency lists, sum up frequencies over course of multiple days.
    :param metadata_list:
    :return: dict-like list of frequencies
    """
    metadata_counter = {}
    for meta in metadata_list:
        for key, value in meta:
            if key not in metadata_counter:
                metadata_counter[key] = value
            else:
                metadata_counter[key] += value
    return list(metadata_counter.items())


def zip_aggregate_metadata(user_journey_df):
    """
    TODO: needs more work, right now it is dependant on hardcoded df column specification. Not used atm
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
    """
    Bulk-execute main input pre-processing functions: from BigQuery journey strings to Page_Event_List to Page_List.
    PageSequence required for dataframes groupbys/filtering.
    :param user_journey_df: dataframe
    :return: no return, columns added in place.
    """
    logger.info("BQ Sequence string to Page_Event_List...")
    user_journey_df['Page_Event_List'] = user_journey_df['Sequence'].map(prep.bq_journey_to_pe_list)
    logger.info("Page_Event_List to Page_List...")
    user_journey_df['Page_List'] = user_journey_df['Page_Event_List'].map(lambda x: prep.extract_pe_components(x, 0))
    logger.info("Page_List to PageSequence...")
    user_journey_df['PageSequence'] = user_journey_df['Page_List'].map(lambda x: ">>".join(x))


def event_preprocess(user_journey_df):
    """
    Bulk-execute event related functions... Run after sequence_preprocess(user_journey_df) so that
    Page_Event_List column exists
    :param user_journey_df: dataframe
    :return: no return, columns added in place.
    """
    logger.info("Page_Event_List to Event_List...")
    user_journey_df['Event_List'] = user_journey_df['Page_Event_List'].map(lambda x: prep.extract_pe_components(x, 1))
    logger.info("Computing event-related counts and frequencies...")
    event_counters(user_journey_df)


def event_counters(user_journey_df):
    """
    Bulk map functions for event frequency/counts.
    :param user_journey_df: dataframe
    :return: no return, columns added in place.
    """
    user_journey_df['num_event_cats'] = user_journey_df['Event_List'].map(feat.count_event_cat)
    user_journey_df['Event_cats_agg'] = user_journey_df['Event_List'].map(feat.aggregate_event_cat)
    user_journey_df['Event_cat_act_agg'] = user_journey_df['Event_List'].map(feat.aggregate_event_cat_act)


def add_loop_columns(user_journey_df):
    """
    Bulk map functions for event frequency/counts.
    :param user_journey_df: dataframe
    :return: no return, columns added in place.
    """
    logger.info("Collapsing loops...")
    user_journey_df['Page_List_NL'] = user_journey_df['Page_List'].map(prep.collapse_loop)
    # In order to groupby during analysis step
    logger.info("De-looped lists to string...")
    user_journey_df['Page_Seq_NL'] = user_journey_df['Page_List_NL'].map(lambda x: ">>".join(x))
    # Count occurrences of de-looped journeys, most generic journey frequency metric.
    logger.info("Aggregating de-looped journey occurrences...")
    user_journey_df['Occurrences_NL'] = user_journey_df.groupby('Page_Seq_NL')['Occurrences'].transform('sum')


def sliced_groupby_meta(df_slice, depth, multiple_dfs):
    """

    :param df_slice:
    :param depth:
    :param multiple_dfs:
    :return: no return, mapping and drops happen inplace.
    """
    agg = df_slice.columns[1]
    # One-off
    if depth == 0:
        df_slice[agg] = df_slice[agg].map(str_to_dict)
    if multiple_dfs:
        metadata_gpb = df_slice.groupby('Sequence')[agg].apply(aggregate_dict)
        # logger.info("Mapping {}, items: {}...".format(agg, len(metadata_gpb)))
        df_slice[agg] = df_slice['Sequence'].map(metadata_gpb)
        drop_duplicate_rows(df_slice)


def drop_duplicate_rows(df_slice):
    """
    Drop duplicate rows from a dataframe slice.
    :param df_slice:
    :return:
    """
    bef = df_slice.shape[0]
    # logger.info("Current # of rows: {}. Dropping duplicate rows..".format(bef))
    df_slice.drop_duplicates(subset='Sequence', keep='first', inplace=True)
    after = df_slice.shape[0]
    logger.info("Dropped {} duplicated rows.".format(bef - after))


def partition_list(dataframe_list, chunks):
    """

    :param dataframe_list:
    :param chunks:
    :return:
    """
    if chunks > 0:
        initial = [list(xs) for xs in np.array_split(list(range(len(dataframe_list))), chunks)]
        # print(initial)
        if len(initial) > 1 and not FEWER_THAN_CPU:
            initial = merge_small_partition(initial)
        return initial
    else:
        return [[0]]


def merge_small_partition(partitions):
    """
    Merge small partitions of length 1 into previous partition, reduce number of recursive runs.
    :param partitions:
    :return:
    """
    to_merge = []
    for partition in partitions:
        if len(partition) == 1:
            to_merge.append(partition[0])
            partitions.remove(partition)
    if len(to_merge) >= 1:
        partitions[-1].extend(to_merge)
    return partitions


def conditional_pre_gpb_drop(df_occ_slice, df_meta_slice):
    """
    Drop samples from metadata dataframe slice depending on already reduced Occurrences slice (occ slice set up as basis
    for drop because it's the fastest to compute. Only runs if contents df_occ_slice have already been reduced.
    :param df_occ_slice: list of (file_code, df_occurence_slice) tuples
    :param df_meta_slice: list of (file_code, df_meta_slice) tuples
    :return: reduced df_meta_slice
    """
    for df_code_i, df_slice_i in df_occ_slice:
        for df_code_j, df_slice_j in df_meta_slice:
            if df_code_i == df_code_j:
                seq_occ = df_slice_i.Sequence.values
                df_slice_j.query("Sequence.isin(@seq_occ)", inplace=True)
                # print("after", df_slice_j.shape)
    return df_meta_slice


def distribute_df_slices(pool, dflist, chunks, depth=0, additional=None):
    """

    :param pool:
    :param dflist:
    :param chunks:
    :param depth: Increases roughly every 4-5 days of data accumulation
    :param additional:
    :return:
    """
    # or (len(dflist) == 1 and depth == 0)
    if len(dflist) > 1 or (len(dflist) == 1 and depth == 0):
        new_list = []
        partitions = partition_list(dflist, chunks)
        multi_dfs = []
        for i, index_list in enumerate(partitions):
            lst = [dflist[ind] for ind in index_list]
            multi_dfs.append(len(lst) > 1)
            logger.info("Run: {} Num_of_df_to_merge: {}".format(i, len(lst)))
            pair_df = pd.concat(lst)
            delete_vars(lst)
            logger.info("Size of merged dataframe: {}".format(pair_df.shape))
            new_list.append(pair_df)

        if additional is not None and depth > 0 and any(multi_dfs):
            print("Adding to ", multi_dfs[-1])
            new_list[-1].append(additional)

        slices_occ, slices_meta = slice_many_df(new_list, True)

        multi_dfs = [multi_dfs[i] for i, _ in slices_occ]
        slices_occ = map_aggregate_function(depth, multi_dfs, pool, slices_occ)

        if (depth >= DEPTH_LIM and MAX_DEPTH >= 2) and DROP_ONE_OFFS:
            logger.info("conditional_pre_gpb_drop")
            slices_meta = conditional_pre_gpb_drop(slices_occ, slices_meta)

        multi_dfs = [multi_dfs[i] for i, _ in slices_meta]
        slices_meta = map_aggregate_function(depth, multi_dfs, pool, slices_meta)
        # slices_meta = pool.starmap(sliced_mass_preprocess, zip(slices_meta, itertools.repeat(depth),
        #                                                        multi_dfs)
        new_list = slices_occ + slices_meta

        # print("THEM COLUMNS", [(new.columns, new.shape) for _, new in new_list])
        new_list = merge_sliced_df(new_list, len(partitions))
        # print("THEM COLUMNS", [(new.columns, new.shape) for new in new_list])
        return distribute_df_slices(pool, new_list, int(chunks / 2), depth + 1, additional)
    else:
        return dflist[0]


def map_aggregate_function(depth, multi_dfs, pool, df_slices):
    """

    :param depth:
    :param multi_dfs:
    :param pool:
    :param df_slices:
    :return:
    """
    shape = max([slice_occ.shape[0] for _, slice_occ in df_slices])
    if shape < ROW_LIMIT:
        print("multiprocessing, input matrix shape ", shape)
        df_slices = pool.starmap(sliced_mass_preprocess, zip(df_slices, itertools.repeat(depth),
                                                             multi_dfs))
    else:
        print("no multiprocessing, input matrix too big ", shape)
        parameters = list(zip(df_slices, multi_dfs))
        df_slices = [sliced_mass_preprocess(code_df_slice_i, depth, multiple_dfs_i) for
                     code_df_slice_i, multiple_dfs_i in parameters]
    return df_slices


def sliced_mass_preprocess(code_df_slice, depth, multiple_dfs):
    """

    :param code_df_slice:
    :param depth:
    :param multiple_dfs:
    :return:
    """
    code = code_df_slice[0]
    df_slice = code_df_slice[1]
    # print(df_slice.columns)
    if df_slice.columns[1] in COUNTABLE_AGGREGATE_COLUMNS:
        logger.info("Aggregating {}...".format(df_slice.columns[1]))
        sliced_groupby_meta(df_slice, depth, multiple_dfs)
    elif "Occurrences" in df_slice.columns:
        logger.info("Occurrences...")
        df_slice['Occurrences'] = df_slice.groupby('Sequence')['Occurrences'].transform('sum')
        if DROP_ONE_OFFS:
            df_slice['Page_Seq_Occurrences'] = df_slice.groupby('PageSequence')['Occurrences'].transform('sum')
        if multiple_dfs:
            drop_duplicate_rows(df_slice)
        # print(DEPTH_LIM, MAX_DEPTH, DROP_INFREQ)
        if (depth >= DEPTH_LIM and MAX_DEPTH >= 2) and DROP_ONE_OFFS:
            bef = df_slice.shape[0]
            # logger.info("Current # of rows: {}. Dropping journeys occurring only once..".format(bef))
            df_slice = df_slice[df_slice.Page_Seq_Occurrences > 1]
            after = df_slice.shape[0]
            logger.info("Dropped {} one-off rows.".format(bef - after))
    return code, df_slice


def slice_many_df(df_list, ordered=False):
    """

    :param df_list:
    :param ordered:
    :return:
    """
    if not ordered:
        return [(i, df.iloc[:, ind].copy(deep=True)) for i, df in enumerate(df_list) for ind in slice_dataframe(df)]
    else:
        return [(i, df.iloc[:, ind].copy(deep=True)) for i, df in enumerate(df_list) for ind in slice_dataframe(df) if
                "Occurrences" in df.columns[ind]], [(i, df.iloc[:, ind].copy(deep=True)) for i, df in enumerate(df_list)
                                                    for ind in
                                                    slice_dataframe(df) if
                                                    "Occurrences" not in df.columns[ind]]


def slice_dataframe(df):
    """

    :param df:
    :return:
    """
    sliced_df = []
    for col in SLICEABLE_COLUMNS:
        if col in df.columns:
            if col == "Occurrences":
                if DROP_ONE_OFFS:
                    sliced_df.append(
                        [df.columns.get_loc("Sequence"), df.columns.get_loc("PageSequence"), df.columns.get_loc(col)])
                else:
                    sliced_df.append(
                        [df.columns.get_loc("Sequence"), df.columns.get_loc(col)])
            else:
                sliced_df.append([df.columns.get_loc("Sequence"), df.columns.get_loc(col)])
    return sliced_df


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


def multiprocess_make(files, destination, merged_filename):
    """

    :param files:
    :param destination:
    :param merged_filename:
    :return:
    """
    global FEWER_THAN_CPU, MAX_DEPTH, NUM_CPU, BATCH_SIZE
    batch_size = BATCH_SIZE
    NUM_CPU = cpu_count()
    batching, batches = compute_batches(files, batch_size)
    num_chunks = compute_initial_chunksize(len(files)) if not batching else compute_initial_chunksize(batch_size)

    FEWER_THAN_CPU = num_chunks == len(files) if not batching else batch_size + 1 <= NUM_CPU
    MAX_DEPTH = compute_max_depth(files, num_chunks, 0) if not batching else compute_max_depth([0] * (batch_size + 1),
                                                                                               num_chunks, 0)
    logger.info("BATCH_SIZE: {} MAX_DEPTH: {} NUM_FILES: {}".format(batch_size, MAX_DEPTH, len(files)))
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
    logger.info("")

    sequence_preprocess(df)
    event_preprocess(df)
    add_loop_columns(df)

    logger.info("Shape: {}".format(df.shape))
    path_to_file = os.path.join(destination, merged_filename)
    logger.info("Saving at: {}".format(path_to_file))
    df.to_csv(path_to_file, compression='gzip', index=False)

    pool.close()
    pool.join()

    logger.info("Multi done")


def process(files, pool, num_chunks, df_prev=None):
    """

    :param files:
    :param pool:
    :param num_chunks:
    :param df_prev:
    :return:
    """
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


def compute_initial_chunksize(number_of_files):
    """

    :param number_of_files:
    :return:
    """
    if number_of_files > NUM_CPU:
        return int(number_of_files / 2)
    else:
        return number_of_files


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


def read_file(filename):
    """

    :param filename:
    :return:
    """
    logging.info("Reading: {}".format(filename))
    df = pd.read_csv(filename, compression="gzip")
    # print(df.shape)
    if DROP_ONES:
        print("dropping ones")
        df.query("PageSeq_Length > 1", inplace=True)
    elif KEEP_ONES:
        print("keeping only ones")
        df.query("PageSeq_Length == 1", inplace=True)
    # print(df.shape)
    if DROP_ONE_OFFS:
        sequence_preprocess(df)
        df.drop(DROPABLE_COLS, axis=1, inplace=True)
    return df


def delete_vars(x):
    """

    :param x:
    :return:
    """
    if isinstance(x, list):
        for xs in x:
            del xs
    del x


def compute_max_depth(test_list, chunks, depth):
    """

    :param test_list:
    :param chunks:
    :param depth:
    :return:
    """
    partitions = partition_list(test_list, chunks)
    if len(test_list) > 1:
        new_lst = [0 for _ in partitions]
        return compute_max_depth(new_lst, (lambda x: int(x / 2) if int(x / 2) > 0 else 1)(chunks), depth + 1)
    else:
        return depth


def generate_file_list(source_dir, stub):
    """

    :param source_dir:
    :param stub:
    :return:
    """
    file_list = [os.path.join(source_dir, file) for file in os.listdir(source_dir)]
    if stub is not None:
        return [file for file in file_list if stub in file]
    else:
        return file_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make datasets module')
    parser.add_argument('source_directory', help='Source directory for input dataframe file(s).')
    parser.add_argument('dest_directory', help='Specialized destination directory for output dataframe file.')
    parser.add_argument('output_filename', help='Naming convention for resulting merged dataframe file.')
    parser.add_argument('--drop_one_offs', action='store_true')
    parser.add_argument('--keep_only_len_one', action='store_true')
    parser.add_argument('--drop_len_one', action='store_true')
    parser.add_argument('--filename_stub', nargs="?")
    args = parser.parse_args()

    # Set up variable values from parsed arguments
    DROP_ONE_OFFS = args.drop_one_offs
    DROP_ONES = args.drop_len_one
    KEEP_ONES = args.keep_only_len_one
    print(DROP_ONE_OFFS, DROP_ONES, KEEP_ONES)

    DATA_DIR = os.getenv("DATA_DIR")
    source_dir = os.path.join(DATA_DIR, args.source_directory)
    dest_dir = os.path.join(DATA_DIR, args.dest_directory)
    final_filename = args.output_filename
    filename_stub = args.filename_stub

    LOGGING_CONFIG = os.getenv("LOGGING_CONFIG")
    logging.config.fileConfig(LOGGING_CONFIG)
    logger = logging.getLogger('make_dataset')

    logger.info("Loading data")

    to_load = generate_file_list(source_dir, filename_stub)

    if not os.path.isdir(dest_dir):
        logging.info("Specified destination directory does not exist, creating...")

    multiprocess_make(to_load, dest_dir, final_filename + ".csv.gz")
