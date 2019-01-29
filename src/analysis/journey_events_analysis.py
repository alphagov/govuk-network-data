import argparse
import logging.config
import os
from ast import literal_eval
from collections import Counter

import pandas as pd
from scipy import stats


def device_count(x, device):
    return sum([value for item, value in x if item == device])


def has_related_event(x):
    return all(cond in x for cond in ["relatedLinkClicked", "Related content"])


def has_nav_event():
    return False


def map_counter(df):
    df["DesktopCount"] = df['DeviceCategories'].map(lambda x: device_count(x, "desktop"))
    df["MobileCount"] = df['DeviceCategories'].map(lambda x: device_count(x, "mobile"))


def split_dataframe(df):
    df["Has_Related"] = df["Sequence"].map(has_related_event)
    df_related = df[df["Has_Related"]]
    return df_related


def weight_seq_length(page_lengths, occurrences, name):
    length_occ = Counter()
    for length, occ in zip(page_lengths, occurrences):
        length_occ[length] += occ
    data = []
    for key, value in length_occ.items():
        for i in range(value):
            data.append(key)
    return pd.Series(data, name=name)


def chi2_test(vol_desk, vol_mobile, vol_mobile_rel, vol_desk_rel):
    vol_mobile_no_rel = vol_mobile - vol_mobile_rel
    vol_desk_no_rel = vol_desk - vol_desk_rel
    obs = [[vol_mobile_rel, vol_mobile_no_rel], [vol_desk_rel, vol_desk_no_rel]]
    return stats.chi2_contingency(obs)


def compute_volumes(df, columns):
    return (df[column].sum() for column in columns)


def compute_percent(nums, denom):
    return (round((num * 100) / denom, 2) for num in nums)


def compute_stats(df, df_related, df_stats):
    columns = ["Occurrences", "DesktopCount", "MobileCount"]
    vol_all, vol_desk, vol_mobile = compute_volumes(df, columns)
    vol_all_related, vol_desk_rel, vol_mobile_rel = compute_volumes(df_related, columns)

    percent_from_desk, percent_from_mobile = compute_percent([vol_desk, vol_mobile], vol_all)

    percent_related = round((vol_all_related * 100) / vol_all, 2)
    percent_from_desk_rel = round((vol_desk_rel * 100) / vol_desk, 2)
    percent_from_mobile_rel = round((vol_mobile_rel * 100) / vol_mobile, 2)
    # shape_all = df.shape[0]
    # shape_all_rel = df[df.Has_Related].shape[0]
    # shape_desk = desktop_journeys.shape[0]
    # shape_desk_rel = desktop_journeys[desktop_journeys.Has_Related].shape[0]
    # shape_mobile = mobile_journeys.shape[0]
    # shape_mobile_rel = mobile_journeys[mobile_journeys.Has_Related].shape[0]
    #
    # shapes = [shape_all, shape_all_rel,
    #           shape_desk, shape_desk_rel,
    #           shape_mobile, shape_mobile_rel]

    df_stats["Volume"] = [vol_all, vol_all_related,
                          vol_desk, vol_desk_rel,
                          vol_mobile, vol_mobile_rel]
    df_stats["Percentage"] = [100, percent_related,
                              percent_from_desk, percent_from_desk_rel,
                              percent_from_mobile, percent_from_mobile_rel]
    # df_stats["Shape"] = shapes

    a, b, c, d = chi2_test(vol_desk, vol_mobile, vol_mobile_rel, vol_desk_rel)

    return df_stats


def describe_dfs(to_eval):
    descriptive = pd.DataFrame()
    for length, occ, name in to_eval:
        sr = weight_seq_length(length, occ, name).describe().apply(lambda x: format(x, '.3f'))
        descriptive[sr.name] = sr
    return descriptive


def column_eval(cols, df):
    for column in cols:
        if not isinstance(df[column].iloc[0], list):
            print(column)
            df[column] = df[column].map(literal_eval)
    if "PageSeq_Length" not in df.columns:
        df['Page_List'] = df['Page_List'].map(literal_eval)
        df['PageSeq_Length'] = df['Page_List'].map(len)


def run(filename):
    df = pd.read_csv(filename, sep="\t", compression="gzip")

    column_eval([], df)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Module to run analysis on user journeys in terms of a specific'
                                                 'event(s). For now focusing on \'Related content\' links. Reads'
                                                 'in data from the \'processed_journey\' directory.')
    # parser.add_argument('source_directory', help='Source directory for input dataframe file(s).')
    # parser.add_argument('dest_directory', help='Specialized destination directory for output dataframe file.')
    parser.add_argument('input_filename', help='Source user journey file to analyse.')
    parser.add_argument('-doo', '--drop_one_offs', action='store_true',
                        help='Drop journeys occurring only once (on a daily basis, '
                             'or over approximately 3 day periods).')
    parser.add_argument('-kloo', '--keep_len_one_only', action='store_true',
                        help='Keep ONLY journeys with length 1 ie journeys visiting only one page.')
    parser.add_argument('-dlo', '--drop_len_one', action='store_true',
                        help='Drop journeys with length 1 ie journeys visiting only one page.')
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
        else:
            logging.info(
                "Specified destination directory \"{}\" exists, adding \'v2\' to results...".format(dest_directory))
