import argparse
import logging.config
import os
from ast import literal_eval
from collections import Counter

import pandas as pd
from scipy import stats

AGGREGATE_COLUMNS = ['DeviceCategories', 'Event_cats_agg', 'Event_cat_act_agg']

NAVIGATE_EVENT_CATS = ['breadcrumbClicked',
                       'homeLinkClicked',
                       '/search',
                       'navDocumentCollectionLinkClicked',
                       'navAccordionLinkClicked',
                       'navLeafLinkClicked',
                       'navPolicyAreaLinkClicked',
                       'navServicesInformationLinkClicked',
                       'navSubtopicContentItemLinkClicked',
                       'navSubtopicLinkClicked',
                       'navTopicLinkClicked',
                       'relatedTaxonomyLinkClicked',
                       'stepNavHeaderClicked', 'stepNavLinkClicked', 'stepNavPartOfClicked']

# Useful for explicit event category and action matching, may extend in the future
NAVIGATE_EVENT_CATS_ACTS = [('relatedLinkClicked', 'Explore the topic')]


def device_count(x, device):
    return sum([value for item, value in x if item == device])


def has_related_event(sequence_str):
    return all(cond in sequence_str for cond in ["relatedLinkClicked", "Related content"])


def has_nav_event_cat(sequence_str):
    return any(event_cat in sequence_str for event_cat in NAVIGATE_EVENT_CATS)


def has_nav_event_cat_act(sequence_str):
    return any(
        event_cat in sequence_str and event_act in sequence_str for event_cat, event_act in NAVIGATE_EVENT_CATS_ACTS)


def map_device_counter(df):
    """
    Count the device-based occurrences per target device
    :param df:
    :return:
    """
    logging.info("Mapping device counts")
    df["DesktopCount"] = df['DeviceCategories'].map(lambda x: device_count(x, "desktop"))
    df["MobileCount"] = df['DeviceCategories'].map(lambda x: device_count(x, "mobile"))


def chi2_test(vol_desk, vol_mobile, vol_mobile_rel, vol_desk_rel):
    vol_mobile_no_rel = vol_mobile - vol_mobile_rel
    vol_desk_no_rel = vol_desk - vol_desk_rel
    obs = [[vol_mobile_rel, vol_mobile_no_rel], [vol_desk_rel, vol_desk_no_rel]]
    return stats.chi2_contingency(obs)


def compute_volumes(df, occ_cols):
    return (df[occ].sum() for occ in occ_cols)


def compute_percents(nums, denoms):
    if len(nums) == len(denoms):
        return (round((num * 100) / denom, 2) for num, denom in zip(nums, denoms))
    return -1


def compute_stats(df, df_filtered, occ_cols):
    logger.info("Computing occurrence-based statistics...")

    ind = ["All", "All_related", "Desktop", "Desktop_rel", "Mobile", "Mobile_rel"]
    cols = ["Volume", "Percentage", "Shape"]
    df_stats = pd.DataFrame(index=ind, columns=cols)

    vol_all, vol_desk, vol_mobile = compute_volumes(df, occ_cols)
    vol_all_related, vol_desk_rel, vol_mobile_rel = compute_volumes(df_filtered, occ_cols)

    percent_from_desk, percent_from_mobile = compute_percents([vol_desk, vol_mobile], 2 * [vol_all])

    percent_related, percent_from_desk_rel, percent_from_mobile_rel = compute_percents(
        [vol_all_related, vol_desk_rel, vol_mobile_rel],
        [vol_all, vol_desk, vol_mobile])

    df_stats["Volume"] = [vol_all, vol_all_related,
                          vol_desk, vol_desk_rel,
                          vol_mobile, vol_mobile_rel]
    df_stats["Percentage"] = [100, percent_related,
                              percent_from_desk, percent_from_desk_rel,
                              percent_from_mobile, percent_from_mobile_rel]

    # a, b, c, _ = chi2_test(vol_desk, vol_mobile, vol_mobile_rel, vol_desk_rel)

    return df_stats


def weight_seq_length(page_lengths, occurrences, name):
    length_occ = Counter()
    for length, occ in zip(page_lengths, occurrences):
        length_occ[length] += occ
    data = []
    for key, value in length_occ.items():
        for i in range(value):
            data.append(key)
    return pd.Series(data, name=name)


def list_zipper(df_list, count_cols, names, col_to_describe):
    return [[df_all[col_to_describe], df_all[count_col], name] for df_all, count_col, name in
            zip(df_list, count_cols, names)]


def describe_dfs(df_list_all, df_list_filtered, col_to_describe, count_cols):
    """

    :param df:
    :param df_related:
    :param col_to_describe:
    :return:
    """

    logger.info("Computing statistics for {}".format(col_to_describe))
    descriptive = pd.DataFrame()
    names_all = ["All_" + name for name in ["Journeys", "Desktop", "Mobile"]]
    names_rel = [name + "_Related" for name in ["Journeys", "Desktop", "Mobile"]]

    to_eval = list_zipper(df_list_all, count_cols, names_all, col_to_describe) + list_zipper(df_list_filtered,
                                                                                             count_cols,
                                                                                             names_rel, col_to_describe)

    for length, occ, name in to_eval:
        sr = weight_seq_length(length, occ, name).describe().apply(lambda x: format(x, '.3f'))
        descriptive[sr.name] = sr

    return descriptive


def column_eval(df):
    """
    Evaluate speficied columns as lists instead of strings. Compute Page_List lengths, if missing.
    :param df:
    :return: void, inplace
    """
    logger.info("Literal eval...")
    for column in AGGREGATE_COLUMNS:
        if column in df.columns and not isinstance(df[column].iloc[0], list):
            print("Working on column: {}".format(column))
            df[column] = df[column].map(literal_eval)
    if "PageSeq_Length" not in df.columns:
        logger.info("Computing PageSeq_Length...")
        df['Page_List'] = df['Page_List'].map(literal_eval)
        df['PageSeq_Length'] = df['Page_List'].map(len)


def initialize(filename, reports_dest):
    df = pd.read_csv(filename, sep="\t", compression="gzip")
    column_eval(df)
    # For dataframe files that include tablet devices
    df["TabletCount"] = df['DeviceCategories'].map(lambda x: device_count(x, "tablet"))
    df["Occurrences"] = df["Occurrences"] - df["TabletCount"]

    map_device_counter(df)

    df["Has_Related"] = df["Sequence"].map(has_related_event)

    # Journeys per device
    desktop_journeys = df[df.DesktopCount > 0]
    mobile_journeys = df[df.MobileCount > 0]

    # Related journeys, all/per device
    df_related = df[df["Has_Related"]]
    desk_rel_journeys = desktop_journeys[desktop_journeys["Has_Related"]]
    mobile_rel_journeys = mobile_journeys[mobile_journeys["Has_Related"]]

    occurrence_cols = ["Occurrences", "DesktopCount", "MobileCount"]

    df_stats = compute_stats(df, df_related, occurrence_cols)
    df_stats['Shape'] = [df.shape[0], df_related.shape[0], desktop_journeys.shape[0], desk_rel_journeys.shape[0],
                         mobile_journeys.shape[0], mobile_rel_journeys.shape[0]]

    descriptive_df = describe_dfs([df, desktop_journeys, mobile_journeys],
                                  [df_related, desk_rel_journeys, mobile_rel_journeys],
                                  "PageSeq_Length", occurrence_cols)

    df_stats.to_csv(os.path.join(reports_dest,  "device_rel_stats.csv"))
    descriptive_df.to_csv(os.path.join(reports_dest,  "PageSeq_Length" + "_describe.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Module to run analysis on user journeys in terms of a specific'
                                                 'event(s). For now focusing on \'Related content\' links. Reads'
                                                 'in data from the \'processed_journey\' directory.')
    parser.add_argument('input_filename', help='Source user journey file to analyse.')
    parser.add_argument('-q', '--quiet', action='store_true', default=False, help='Turn off debugging logging.')
    args = parser.parse_args()

    DATA_DIR = os.getenv("DATA_DIR")
    REPORTS_DIR = os.getenv("REPORTS_DIR")
    source_directory = os.path.join(DATA_DIR, "processed_journey")
    dest_directory = os.path.join(REPORTS_DIR, args.input_filename)
    input_file = os.path.join(source_directory, args.input_filename + ".csv.gz")

    LOGGING_CONFIG = os.getenv("LOGGING_CONFIG")
    logging.config.fileConfig(LOGGING_CONFIG)
    logger = logging.getLogger('user_journey_event_analysis')

    if args.quiet:
        logging.disable(logging.DEBUG)

    if os.path.isfile(input_file):
        if not os.path.isdir(dest_directory):
            logging.info(
                "Specified destination directory \"{}\" does not exist, creating...".format(dest_directory))
            os.mkdir(dest_directory)
            initialize(input_file, dest_directory)
        else:
            logging.info(
                "Specified destination directory \"{}\" exists, adding \'v2\' to results...".format(dest_directory))
