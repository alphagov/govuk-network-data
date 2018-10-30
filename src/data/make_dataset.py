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
from pandas import DataFrame

src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(src, "data"))
sys.path.append(os.path.join(src, "features"))
import multiprocess_utils as multi_utils
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
# Execute module for only one file
SINGLE: bool = False
# Fewer files to process than available cpus.
FEWER_THAN_CPU: bool = False
# Drop journeys occurring once (not in a day, multiple days, governed by DEPTH globals). If false, overrides depth
# globals and keeps journeys, resulting in massive dataframes (danger zone).
DROP_ONE_OFFS: bool = False
# Drop journeys of length 1
DROP_ONES: bool = False
# Keep only journeys of length 1
KEEP_ONES: bool = False
# Maximum recursive depth for distribution function
MAX_DEPTH: int = -1
# Recursive depth limit for distribution function, so one-off journeys are drop in time.
DEPTH_LIM: int = 1
# If there are many files to be merge, load in/preprocess in batches
BATCH_SIZE: int = 3
# A bit of a magic number, but limits dataframes that can be passed off to workers. If dataframe exceeds this size,
# switch to sequential execution.
ROW_LIMIT: int = 3000000


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
    # TODO: Remove condition + internal PageSequence post-testing/debugging.
    if 'PageSequence' not in user_journey_df.columns:
        user_journey_df['PageSequence'] = user_journey_df['Page_List'].map(lambda x: ">>".join(x))
    else:
        user_journey_df['PageSequence_internal'] = user_journey_df['Page_List'].map(lambda x: ">>".join(x))


def event_preprocess(user_journey_df):
    """
    Bulk-execute event related functions... Run after sequence_preprocess(user_journey_df) so that
    Page_Event_List column exists
    :param user_journey_df: dataframe
    :return: no return, columns added in place.
    """
    logger.info("Preprocess and aggregate events...")
    logger.debug("Page_Event_List to Event_List...")
    user_journey_df['Event_List'] = user_journey_df['Page_Event_List'].map(lambda x: prep.extract_pe_components(x, 1))
    logger.debug("Computing event-related counts and frequencies...")
    event_counters(user_journey_df)


def event_counters(user_journey_df):
    """
    Bulk map functions for event frequency/counts.
    :param user_journey_df: dataframe
    :return: no return, columns added in place.
    """
    logger.debug("Event_List to ...")
    user_journey_df['num_event_cats'] = user_journey_df['Event_List'].map(feat.count_event_cat)
    logger.debug("Event_List to ...")
    user_journey_df['Event_cats_agg'] = user_journey_df['Event_List'].map(feat.aggregate_event_cat)
    logger.debug("Event_List to ...")
    user_journey_df['Event_cat_act_agg'] = user_journey_df['Event_List'].map(feat.aggregate_event_cat_act)


def add_loop_columns(user_journey_df):
    """
    Bulk map functions for event frequency/counts.
    :param user_journey_df: dataframe
    :return: no return, columns added in place.
    """
    logger.info("Preprocess journey looping...")
    logger.debug("Collapsing loops...")
    user_journey_df['Page_List_NL'] = user_journey_df['Page_List'].map(prep.collapse_loop)
    # In order to groupby during analysis step
    logger.debug("De-looped lists to string...")
    user_journey_df['Page_Seq_NL'] = user_journey_df['Page_List_NL'].map(lambda x: ">>".join(x))
    # Count occurrences of de-looped journeys, most generic journey frequency metric.
    logger.debug("Aggregating de-looped journey occurrences...")
    user_journey_df['Occurrences_NL'] = user_journey_df.groupby('Page_Seq_NL')['Page_Seq_Occurrences'].transform('sum')
    logger.debug("De-looped page sequence to list...")
    user_journey_df['Page_List_NL'] = user_journey_df['Page_Seq_NL'].map(
        lambda x: x.split(">>") if isinstance(x, str) else np.NaN)


def groupby_meta(df_slice: DataFrame, depth: int, multiple_dfs: bool):
    """
    Aggregate specified metadata column. If it's the first recursive run, transform aggregate metadata string to a
    dict-like list.
    :param df_slice: specified metadata column (refer to AGGREGATE_COLUMNS values)
    :param depth: (int) recursive call tracker, depth = 0 indicates first recursive call
    :param multiple_dfs: (boolean) indicates whether many dfs have been merged and require grouping by
    :return: no return, mapping and drops happen inplace on df_slice.
    """
    agg = df_slice.columns[1]
    # One-off
    if depth == 0:
        df_slice[agg] = df_slice[agg].map(str_to_dict)
    if multiple_dfs:
        metadata_gpb = df_slice.groupby('Sequence')[agg].apply(aggregate_dict)
        logger.debug("Mapping {}, items: {}...".format(agg, len(metadata_gpb)))
        df_slice[agg] = df_slice['Sequence'].map(metadata_gpb)
        drop_duplicate_rows(df_slice)


def drop_duplicate_rows(df_slice: DataFrame):
    """
    Drop duplicate rows from a dataframe slice.
    :param df_slice:
    :return:
    """
    bef = df_slice.shape[0]
    logger.debug("Current # of rows: {}. Dropping duplicate rows..".format(bef))
    df_slice.drop_duplicates(subset='Sequence', keep='first', inplace=True)
    after = df_slice.shape[0]
    logger.debug("Dropped {} duplicated rows.".format(bef - after))


# noinspection PyUnusedLocal
def conditional_pre_gpb_drop(df_occ_slice: list, df_meta_slice: list):
    """
    Drop samples from metadata dataframe slice depending on already reduced Occurrences slice (occ slice set up as basis
    for drop because it's the fastest to compute. Only runs if contents df_occ_slice have already been reduced.
    :param df_occ_slice: list of (file_code, df_occurrence_slice) tuples
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


def process_dataframes(pool: Pool, dflist: list, chunks: int, depth: int = 0, additional: DataFrame = None):
    """
    Main func
    :param pool: pool of worker processes (daemons)
    :param dflist: list of dataframes to evaluate
    :param chunks: len(partitions)
    :param depth: Increases roughly every 4-5 days of data accumulation
    :param additional: from batching process, output from previous run that needs to be merged with dflist contents
    :return: contents of dflist merged into a single, metadata+occurrence-aggregated dataframe
    """
    # or (len(dflist) == 1 and depth == 0)
    if len(dflist) > 1 or (len(dflist) == 1 and depth == 0):
        new_list = []
        partitions = multi_utils.partition_list(dflist, chunks, FEWER_THAN_CPU)
        multi_dfs = []
        for i, index_list in enumerate(partitions):
            lst = [dflist[ind] for ind in index_list]
            multi_dfs.append(len(lst) > 1)
            logger.info("Run: {} Num_of_df_to_merge: {}".format(i, len(lst)))
            pair_df = pd.concat(lst)
            multi_utils.delete_vars(lst)
            logger.info("Size of merged dataframe: {}".format(pair_df.shape))
            new_list.append(pair_df)

        # There is a dataframe from a previous run to include, and recursive level is deep enough
        if additional is not None and depth > 0 and any(multi_dfs):
            print("Adding to ", multi_dfs[-1])
            new_list[-1].append(additional)

        # Slice dataframes contained in new_list into their base columns: one list for occurrences, another for
        # metadata
        slices_occ, slices_meta = multi_utils.slice_many_df(new_list, DROP_ONE_OFFS, SLICEABLE_COLUMNS, True)

        # Assign booleans indicating whether dataframes consist of multiple dataframes (from partitioning)
        multi_dfs = [multi_dfs[i] for i, _ in slices_occ]
        # Aggregate metadata and sum up occurrences (based on Sequence groupby). Drop duplicate rows based on
        # PageSequence values, to avoid over-dropping (since Sequence includes events fired within journey and
        # PageSequence doesn't. Occurrences computed first, since they're the list computationally intensive
        # and can be used as a basis for row drops, to reduce future computes.
        slices_occ = map_aggregate_function(depth, multi_dfs, pool, slices_occ)

        # If rows have been dropped due to one-offs, first reduce metadata slice size
        # Improve this condition
        if ((depth >= DEPTH_LIM and MAX_DEPTH >= 2) or SINGLE) and DROP_ONE_OFFS:
            logger.info("Conditional_pre_gpb_drop")
            slices_meta = conditional_pre_gpb_drop(slices_occ, slices_meta)

        # Boolean assignment for metadata slices
        multi_dfs = [multi_dfs[i] for i, _ in slices_meta]
        # Same with the occurrences run
        slices_meta = map_aggregate_function(depth, multi_dfs, pool, slices_meta)

        # Concatenate lists of aggregated df_slices of Occurrences and metadat
        new_list = slices_occ + slices_meta

        # TODO: debugging, remove post-testing
        # print("THEM COLUMNS", [(new.columns, new.shape) for _, new in new_list])
        #
        new_list = multi_utils.merge_sliced_df(new_list, len(partitions))
        # Recursive call, pass new_list for evaluation/merging, reduce number of chunks,
        # increase recursive level/depth. If available, propagate the additional df.
        return process_dataframes(pool, new_list, int(chunks / 2), depth + 1, additional)
    else:
        return dflist[0]


def map_aggregate_function(depth: int, multi_dfs: list, pool: Pool, df_slices: list):
    """
    Map aggregate to either a worker process or sequentially, depending on the slices'
    maximum size, compared against a global threshold. Accumulate output and return.
    :param depth: depth within recursive function
    :param multi_dfs: list of boolean values referring to
    :param pool:
    :param df_slices:
    :return:
    """
    shape = max([slice_occ.shape[0] for _, slice_occ in df_slices])
    if shape < ROW_LIMIT:
        logging.info("Multiprocessing, input rows: {}".format(shape))
        df_slices = pool.starmap(aggregate, zip(df_slices, itertools.repeat(depth),
                                                multi_dfs))
    else:
        logging.info("No multiprocessing, input rows: {}".format(shape))
        parameters = list(zip(df_slices, multi_dfs))
        df_slices = [aggregate(code_df_slice_i, depth, multiple_dfs_i) for
                     code_df_slice_i, multiple_dfs_i in parameters]
    return df_slices


def aggregate(code_df_slice: tuple, depth: int, multiple_dfs: bool):
    """

    :param code_df_slice:
    :param depth:
    :param multiple_dfs:
    :return:
    """
    code: int = code_df_slice[0]
    df_slice: DataFrame = code_df_slice[1]
    # print(df_slice.columns)
    if df_slice.columns[1] in COUNTABLE_AGGREGATE_COLUMNS:
        logging.debug("Aggregating {}...".format(df_slice.columns[1]))
        groupby_meta(df_slice, depth, multiple_dfs)
    elif "Occurrences" in df_slice.columns:
        logging.debug("Occurrences...")
        df_slice['Occurrences'] = df_slice.groupby('Sequence')['Occurrences'].transform('sum')
        if DROP_ONE_OFFS:
            df_slice['Page_Seq_Occurrences'] = df_slice.groupby('PageSequence')['Occurrences'].transform('sum')
        if multiple_dfs:
            drop_duplicate_rows(df_slice)
        if ((depth >= DEPTH_LIM and MAX_DEPTH >= 2) or SINGLE) and DROP_ONE_OFFS:
            bef = df_slice.shape[0]
            # logger.info("Current # of rows: {}. Dropping journeys occurring only once..".format(bef))
            df_slice = df_slice[df_slice.Page_Seq_Occurrences > 1]
            after = df_slice.shape[0]
            logger.info("Dropped {} one-off rows.".format(bef - after))

    return code, df_slice


def initialize_make(files: list, destination: str, merged_filename: str):
    """

    :param files:
    :param destination:
    :param merged_filename:
    :return:
    """
    global FEWER_THAN_CPU, MAX_DEPTH
    batch_size = BATCH_SIZE
    # Number of available CPUs, governs size of pool/number of daemon worker.
    num_cpu = cpu_count()
    batching, batches = multi_utils.compute_batches(files, batch_size)
    num_chunks, FEWER_THAN_CPU, MAX_DEPTH = setup_parameters(batch_size, batching, files, num_cpu)

    logger.debug("BATCH_SIZE: {} MAX_DEPTH: {} NUM_FILES: {}".format(batch_size, MAX_DEPTH, len(files)))
    logger.debug("Using {} workers...".format(num_cpu))
    if batching: logger.debug("With batching")
    pool = Pool(num_cpu)

    if not batching:
        df = distribute_tasks(files, pool, num_chunks)
    else:
        df = None
        for batch_num, batch in enumerate(batches):
            logging.info("Working on batch: {} with {} file(s)...".format(batch_num + 1, len(batch)))
            df = distribute_tasks(batch, pool, num_chunks, df)

    logging.debug(df.iloc[0])
    sequence_preprocess(df)
    event_preprocess(df)
    add_loop_columns(df)
    logger.info("Dataframe columns: {}".format(df.columns))
    logger.info("Shape: {}".format(df.shape))
    print("Example final row:\n", df.iloc[0])

    path_to_file = os.path.join(destination, merged_filename)
    logger.info("Saving at: {}".format(path_to_file))
    df.to_csv(path_to_file, compression='gzip', index=False)

    pool.close()
    pool.join()

    logger.info("Multi done")


def setup_parameters(batch_size, batching, files, num_cpu):
    if not batching:
        num_chunks = multi_utils.compute_initial_chunksize(len(files), num_cpu)
        fewer_than_cpu = num_chunks == len(files)
        max_depth = multi_utils.compute_max_depth(files, num_chunks, 0,
                                                  fewer_than_cpu)
    else:
        num_chunks = multi_utils.compute_initial_chunksize(batch_size, num_cpu)
        fewer_than_cpu = batch_size + 1 <= num_cpu
        max_depth = multi_utils.compute_max_depth(
            [0] * (batch_size + 1),
            num_chunks, 0, fewer_than_cpu)
    return num_chunks, fewer_than_cpu, max_depth


def distribute_tasks(files, pool, num_chunks, df_prev=None):
    """

    :param files:
    :param pool:
    :param num_chunks:
    :param df_prev:
    :return:
    """
    # logger.info("chunks: {} fewer: {} max_depth: {}".format(num_chunks, FEWER_THAN_CPU, MAX_DEPTH))
    logger.debug("Number of files: {}".format(len(files)))
    logger.info("Multi start...")
    df_list = pool.map(read_file, files)

    if df_prev is not None:
        logger.info("Adding previous dataframe")
        # df_list.append(df_prev)

    logger.info("Distributing tasks...")
    df = process_dataframes(pool, df_list, num_chunks, 0, df_prev)
    return df


def read_file(filename):
    """
    Initialize dataframe using specified filename, do some initial prep if necessary depending on global vars
    (specified via arguments)
    :param filename: filename to read, no exists_check because files are loaded from a specified directory
    :return: loaded (maybe modified) pandas dataframe
    """
    logging.info("Reading: {}".format(filename))
    df: DataFrame = pd.read_csv(filename, compression="gzip")
    # print(df.shape)
    # Drop journeys of length 1
    if DROP_ONES:
        logging.debug("Dropping ones")
        df.query("PageSeq_Length > 1", inplace=True)
    # Keep ONLY journeys of length 1
    elif KEEP_ONES:
        logging.debug("Keeping only ones")
        df.query("PageSeq_Length == 1", inplace=True)
    # If
    if DROP_ONE_OFFS:
        if "PageSequence" not in df.columns:
            sequence_preprocess(df)
            df.drop(DROPABLE_COLS, axis=1, inplace=True)
    return df


def generate_file_list(source_dir, stub):
    """
    Initialize list of files to read from a specified directory. If stub is not empty, filter files to be read
    based on whether their filename includes the stub.
    :param source_dir: Source directory
    :param stub: Filename stub for file filtering
    :return: a list of files
    """
    file_list = [os.path.join(source_dir, file) for file in os.listdir(source_dir)]
    if stub is not None:
        return [file for file in file_list if stub in file]
    else:
        return file_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Module that produces a merged, metadata-aggregated and '
                                                 'preprocessed dataset (.csv.gz), given a source directory '
                                                 'containing raw BigQuery extract dataset(s). Merging is '
                                                 'skipped if only one file is provided.')
    parser.add_argument('source_directory', help='Source directory for input dataframe file(s).')
    parser.add_argument('dest_directory', help='Specialized destination directory for output dataframe file.')
    parser.add_argument('output_filename', help='Naming convention for resulting merged dataframe file.')
    parser.add_argument('-doo', '--drop_one_offs', action='store_true',
                        help='Drop journeys occurring only once (on a daily basis, '
                             'or over approximately 3 day periods).')
    parser.add_argument('-kloo', '--keep_len_one_only', action='store_true',
                        help='Keep ONLY journeys with length 1 ie journeys visiting only one page.')
    parser.add_argument('-dlo', '--drop_len_one', action='store_true',
                        help='Drop journeys with length 1 ie journeys visiting only one page.')
    parser.add_argument('-f', '--filename_stub', default=None, type=str,
                        help='Filter files to be loaded based on whether their filenames contain specified stub.')
    parser.add_argument('-q', '--quiet', action='store_true', default=False, help='Turn off debugging logging.')
    args = parser.parse_args()

    DATA_DIR = os.getenv("DATA_DIR")
    source_directory = os.path.join(DATA_DIR, args.source_directory)
    dest_directory = os.path.join(DATA_DIR, args.dest_directory)
    final_filename = args.output_filename
    filename_stub = args.filename_stub

    LOGGING_CONFIG = os.getenv("LOGGING_CONFIG")
    logging.config.fileConfig(LOGGING_CONFIG)
    logger = logging.getLogger('make_dataset')

    if args.quiet:
        logging.disable(logging.DEBUG)

    if os.path.isdir(source_directory):
        # Set up variable values from parsed arguments
        DROP_ONE_OFFS = args.drop_one_offs
        DROP_ONES = args.drop_len_one
        KEEP_ONES = args.keep_len_one_only
        # pretty_tab = "".join(["\t" for i in range(12)])
        logger.info(
            "Data exclusion parameters:\nDrop one-off journeys: {}"
            "\nDrop journeys of length 1: {}"
            "\nKeep journeys only of length 1: {}".format(DROP_ONE_OFFS, DROP_ONES, KEEP_ONES))

        logger.info("Loading data...")

        to_load = generate_file_list(source_directory, filename_stub)
        if len(to_load) > 0:
            if len(to_load) <= BATCH_SIZE:
                SINGLE = True

            if not os.path.isdir(dest_directory):
                logging.info(
                    "Specified destination directory \"{}\" does not exist, creating...".format(dest_directory))
                os.mkdir(dest_directory)

            initialize_make(to_load, dest_directory, final_filename + ".csv.gz")
        else:
            logging.info(
                "Specified source directory \"{}\" contains no target files.".format(source_directory))

    else:
        logging.info("Specified source directory \"{}\" does not exist, cannot read files.".format(source_directory))

    # list2 = [("d", 3), ("t", 1), ("m", 2)], [("d", 3), ("t", 1), ("m", 2)], [("d", 3), ("t", 1), ("m", 2)]
    # print(list2)
    # print(aggregate_dict(list2))

