import pandas as pd
import os
import sys
import logging.config
import argparse
from ast import literal_eval
src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(src, "data"))
import preprocess as prep


COLUMNS_TO_KEEP = set(['Page_List_NL','Page_List','Occurrences','Seq_Occurrences'])


def read_file(filename):
    df = pd.read_csv(filename, compression="gzip")
    columns = set(df.columns.values)
    df.drop(list(columns-COLUMNS_TO_KEEP), axis=1, inplace=True)
    for column in COLUMNS_TO_KEEP:
        if isinstance(df[column].iloc[0], str) and "," in df[column].iloc[0]:
            logging.info("Working on literal_eval for \"{}\"".format(column))
            df[column] = df[column].map(literal_eval)
    return df


def unique_pages(user_journey_df):
    user_journey_df['Subpaths'] = user_journey_df['Page_List'].map(prep.subpaths_from_list)
    user_journey_df['Subpaths_NL'] = user_journey_df['Page_List_NL'].map(prep.subpaths_from_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Module that produces ')
    parser.add_argument('source_directory', help='Source directory for input dataframe file(s).')
    parser.add_argument('dest_directory', help='Specialized destination directory for output dataframe file.')
    parser.add_argument('output_filename', help='Naming convention for resulting merged dataframe file.')
    parser.add_argument('-f', '--filename_stub', default=None, type=str,
                        help='Filter files to be loaded based on whether their filenames contain specified stub.')
    parser.add_argument('-q', '--quiet', action='store_true', default=False, help='Turn off debugging logging.')
    args = parser.parse_args()

    DATA_DIR = os.getenv("DATA_DIR")
    source_directory = os.path.join(DATA_DIR, args.source_directory)
    dest_directory = os.path.join(DATA_DIR, args.dest_directory)

    final_filename = args.output_filename
    LOGGING_CONFIG = os.getenv("LOGGING_CONFIG")
    logging.config.fileConfig(LOGGING_CONFIG)
    logger = logging.getLogger('make_network_data')

    if args.quiet:
        logging.disable(logging.DEBUG)
