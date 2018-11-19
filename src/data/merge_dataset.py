# # -*- coding: utf-8 -*-

import argparse
import logging.config
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from pandas import DataFrame

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


def list_to_dict(metadata_list):
    """
    Transform metadata lists to dictionary aggregates
    :param metadata_list:
    :return:
    """
    return Counter([xs for xs in metadata_list])


def str_to_dict(metadata_str):
    """
    Transform metadata string eg mobile,desktop,mobile to [(mobile,2),(desktop,1)] dict-like
    list.
    :param metadata_str:
    :return: dict-like list of frequencies
    """
    return list_to_dict(metadata_str.split(','))


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


def taxon_preprocess(user_journey_df):
    """
    Bulk map functions for event frequency/counts.
    :param user_journey_df: dataframe
    :return: no return, columns added in place.
    """
    logger.info("Preprocess taxons...")
    logger.debug("Page_Event_List to Taxon_List...")
    user_journey_df['Taxon_List'] = user_journey_df['Page_Event_List'].map(lambda x: prep.extract_cd_components(x, 2))
    logger.debug("Page_Event_List to Taxon_Page_List...")
    user_journey_df['Taxon_Page_List'] = user_journey_df['Page_Event_List'].map(lambda x: prep.extract_pcd_list(x, 2))


def event_counters(user_journey_df):
    """
    Bulk map functions for event frequency/counts.
    :param user_journey_df: dataframe
    :return: no return, columns added in place.
    """
    # logger.debug("Computing number of event categories...")
    # user_journey_df['num_event_cats'] = user_journey_df['Event_List'].map(feat.count_event_cat)
    logger.debug("Computing frequency of event categories...")
    user_journey_df['Event_cats_agg'] = user_journey_df['Event_List'].map(feat.aggregate_event_cat)
    logger.debug("Computing frequency of event categories and actions...")
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

    if 'Page_Seq_Occurrences' not in user_journey_df.columns:
        logger.debug("Setting up Page_Seq_Occurrences...")
        user_journey_df['Page_Seq_Occurrences'] = user_journey_df.groupby('PageSequence')['Occurrences'].transform(
            'sum')

    # Count occurrences of de-looped journeys, most generic journey frequency metric.
    logger.debug("Aggregating de-looped journey occurrences...")
    user_journey_df['Occurrences_NL'] = user_journey_df.groupby('Page_Seq_NL')['Occurrences'].transform('sum')
    logger.debug("De-looped page sequence to list...")
    user_journey_df['Page_List_NL'] = user_journey_df['Page_Seq_NL'].map(
        lambda x: x.split(">>") if isinstance(x, str) else np.NaN)


def agg_dict(agg_from_dict, row_dict):
    for xs, value in row_dict.items():
        if xs in agg_from_dict.keys():
            agg_from_dict[xs] += value
        else:
            agg_from_dict[xs] = value
    return agg_from_dict


def aggregate(dataframe):
    metadata_counter = {}
    for agg in dataframe.columns:
        if agg in COUNTABLE_AGGREGATE_COLUMNS:
            logging.info("Agg {}".format(agg))
            metadata_counter[agg] = {}

    logging.info("Starting iteration...")
    for i, row in dataframe.iterrows():
        for agg in metadata_counter.keys():
            if row['Sequence'] in metadata_counter[agg].keys():
                metadata_counter[agg][row['Sequence']] = agg_dict(metadata_counter[agg][row['Sequence']],
                                                                  str_to_dict(row[agg]))
            else:
                metadata_counter[agg][row['Sequence']] = str_to_dict(row[agg])

        if i % 500000 == 0:
            logging.debug("At index: {}".format(i))
    return metadata_counter


def preprocess(dataframe: object, single: bool):
    """

    :param dataframe:
    :param single:
    :return:
    """
    logging.info("Dataframe shape: {}".format(dataframe.shape))

    if not single:
        logging.info("Working on multiple merged dataframes")
        metadata_counter = aggregate(dataframe)
    else:
        logging.info("Working on a single dataframe")
        for agg in dataframe.columns:

            if agg in COUNTABLE_AGGREGATE_COLUMNS:
                logging.info("Agg {}".format(agg))
                dataframe[agg] = dataframe[agg].map(lambda x: list(str_to_dict(x).items()))

    logging.info("Occurrences...")
    dataframe['Occurrences'] = dataframe.groupby('Sequence')['Occurrences'].transform('sum')

    if not single:
        bef = dataframe.shape[0]
        logger.debug("Current # of rows: {}. Dropping duplicate rows...".format(bef))
        dataframe.drop_duplicates(subset='Sequence', keep='first', inplace=True)
        after = dataframe.shape[0]
        logger.debug("Dropped {} duplicated rows.".format(bef - after))

        for agg in metadata_counter.keys():
            logger.info("Mapping {}, items: {}...".format(agg, len(metadata_counter[agg])))
            dataframe[agg] = dataframe['Sequence'].map(lambda x: list(metadata_counter[agg][x].items()))

    if DROP_ONE_OFFS:
        dataframe['Page_Seq_Occurrences'] = dataframe.groupby('PageSequence')['Occurrences'].transform('sum')
        bef = dataframe.shape[0]
        dataframe = dataframe[dataframe.Page_Seq_Occurrences > 1]
        after = dataframe.shape[0]
        logger.debug("Dropped {} one-off rows.".format(bef - after))


def initialize_make(files: list, destination: str, merged_filename: str):
    """

    :param files:
    :param destination:
    :param merged_filename:
    :return:
    """

    logging.info("Reading {} files...".format(len(files)))

    df = pd.concat([read_file(file) for file in files], ignore_index=True)

    preprocess(df, len(files) == 1)

    logging.debug(df.iloc[0])

    logging.debug("Saving merged dataframe...")
    path_to_file = os.path.join(destination, "merged_" + merged_filename)
    logger.info("Saving at: {}".format(path_to_file))
    df.to_csv(path_to_file, sep="\t", compression='gzip', index=False)


def read_file(filename):
    """
    Initialize dataframe using specified filename, do some initial prep if necessary depending on global vars
    (specified via arguments)
    :param filename: filename to read, no exists_check because files are loaded from a specified directory
    :return: loaded (maybe modified) pandas dataframe
    """
    logging.info("Reading: {}".format(filename))
    df: DataFrame = pd.read_csv(filename, compression="gzip")
    # logging.info("pre {}".format(df.shape))
    df.dropna(subset=['Sequence'], inplace=True)
    # logging.info("post {}".format(df.shape))
    # print(df.shape)
    # Drop journeys of length 1
    if DROP_ONES:
        logging.debug("Dropping ones...")
        df.query("PageSeq_Length > 1", inplace=True)
    # Keep ONLY journeys of length 1
    elif KEEP_ONES:
        logging.debug("Keeping only ones...")
        df.query("PageSeq_Length == 1", inplace=True)
    # If
    if DROP_ONE_OFFS:
        if "PageSequence" not in df.columns:
            sequence_preprocess(df)
            # df.drop(DROPABLE_COLS, axis=1, inplace=True)
    return df


def generate_file_list(source_dir, stub):
    """
    Initialize list of files to read from a specified directory. If stub is not empty, filter files to be read
    based on whether their filename includes the stub.
    :param source_dir: Source directory
    :param stub: Filename stub for file filtering
    :return: a list of files
    """
    file_list = sorted([os.path.join(source_dir, file) for file in os.listdir(source_dir)])
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
        logger.info(
            "Data exclusion parameters:\nDrop one-off journeys: {}"
            "\nDrop journeys of length 1: {}"
            "\nKeep journeys only of length 1: {}".format(DROP_ONE_OFFS, DROP_ONES, KEEP_ONES))

        logger.info("Loading data...")

        to_load = generate_file_list(source_directory, filename_stub)
        if len(to_load) > 0:

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
