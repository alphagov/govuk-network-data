import argparse
import logging.config
import os
from ast import literal_eval
from scipy import stats



def column_eval(cols, df):
    for column in cols:
        if not isinstance(df[column].iloc[0], list):
            print(column)
            df[column] = df[column].map(literal_eval)
    if "PageSeq_Length" not in df.columns:
        df['Page_List'] = df['Page_List'].map(literal_eval)
        df['PageSeq_Length'] = df['Page_List'].map(len)


def device_count(x, device):
    return sum([value for item, value in x if item == device])


def is_related(x):
    return all(cond in x for cond in ["relatedLinkClicked", "Related content"])


def count_aggregate(df):
    df["DesktopCount"] = df['DeviceCategories'].map(lambda x: device_count(x, "desktop"))
    df["MobileCount"] = df['DeviceCategories'].map(lambda x: device_count(x, "mobile"))
    df["TabletCount"] = df['DeviceCategories'].map(lambda x: device_count(x, "tablet"))


def split_dataframe(df):
    df["Has_Related"] = df["Sequence"].map(is_related)
    ## Seperate out desktop and mobile journeys
    desktop_journeys = df[df.DesktopCount > 0]
    mobile_journeys = df[df.MobileCount > 0]

    desk_rel_journeys = desktop_journeys[desktop_journeys.Has_Related]
    mobile_rel_journeys = mobile_journeys[mobile_journeys.Has_Related]

def chi2_test(obs):
    chi2, p, dof, ex = stats.chi2_contingency(obs)
    print(chi2, p, dof, ex)

def compute_stats(df, df_related, desktop_journeys, mobile_journeys, desk_rel_journeys, mobile_rel_journeys, df_stats):
    vol_all = df.Occurrences.sum()
    vol_all_related = df[df.Has_Related].Occurrences.sum()
    # Number of journeys coming from desktops
    vol_desk = df["DesktopCount"].sum()
    # Number of journeys coming from mobiles
    vol_mobile = df["MobileCount"].sum()
    # Number of journeys coming from tablets
    vol_tablet = df["TabletCount"].sum()

    # Compute number of journeys from specific device that include related links
    # Don't base counting on occurrences, will include excluded device
    vol_desk_rel = desk_rel_journeys.DesktopCount.sum()
    vol_mobile_rel = mobile_rel_journeys.MobileCount.sum()

    vols = [vol_all, vol_all_related,
            vol_desk, vol_desk_rel,
            vol_mobile, vol_mobile_rel]

    percent_related = round((vol_all_related * 100) / vol_all, 2)
    percent_from_desk = round((vol_desk * 100) / df.Occurrences.sum(), 2)
    percent_from_mobile = round((vol_mobile * 100) / df.Occurrences.sum(), 2)
    percent_from_tablet = round((vol_tablet * 100) / df.Occurrences.sum(), 2)
    percent_from_desk_rel = round((vol_desk_rel * 100) / vol_desk, 2)
    percent_from_mobile_rel = round((vol_mobile_rel * 100) / vol_mobile, 2)

    percents = [100, percent_related,
                percent_from_desk, percent_from_desk_rel,
                percent_from_mobile, percent_from_mobile_rel]

    shape_all = df.shape[0]
    shape_all_rel = df[df.Has_Related].shape[0]
    shape_desk = desktop_journeys.shape[0]
    shape_desk_rel = desktop_journeys[desktop_journeys.Has_Related].shape[0]
    shape_mobile = mobile_journeys.shape[0]
    shape_mobile_rel = mobile_journeys[mobile_journeys.Has_Related].shape[0]

    shapes = [shape_all, shape_all_rel,
              shape_desk, shape_desk_rel,
              shape_mobile, shape_mobile_rel]

    df_stats["Volume"] = vols
    df_stats["Percentage"] = percents
    df_stats["Shape"] = shapes


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
