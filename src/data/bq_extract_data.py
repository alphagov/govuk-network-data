import argparse
import datetime
import fnmatch
import logging.config
import os

import pandas as pd


def find_query(query_arg, query_dir):
    for file in os.listdir(query_dir):
        if fnmatch.fnmatch(file, "*" + query_arg + "*"):
            return os.path.join(query_dir, file)


def read_query(filepath):
    with open(filepath, 'r') as file:
        lines = " ".join(line.strip("\n") for line in file)
    return lines


def change_timestamp(x, date, dialect):
    if dialect == "standard":
        return x.replace("TIME_STAMP", date.replace("-", ""))
    else:
        change = str("TIMESTAMP(\"") + date + "\"), " + str("TIMESTAMP(\"") + date + "\"))"
        return x.replace("TIME_STAMP", change)


def looped_query(query_from_file, date_range, exclude, pid, k_path, destination_dir, file_names, dialect):
    runs = len(date_range) - len(exclude)
    for i, date in enumerate(date_range):
        logger.info("RUN {} OUT OF {}".format(str(i + 1), runs))
        if date not in exclude:

            logger.info("Working on: {}".format(date))
            logger.info("Query start...")
            query_for_paths = change_timestamp(query_from_file, date, dialect)
            print(query_for_paths)
            df_in = pd.io.gbq.read_gbq(query_for_paths,
                                       project_id=pid,
                                       reauth=False,
                                       # verbose=True,
                                       private_key=k_path,
                                       dialect=dialect)

            file_name = os.path.join(destination_dir, file_names + "_" + str(date) + '.csv.gz')
            logger.info("Saving at: {}".format(file_name))
            df_in.to_csv(file_name, compression='gzip',
                         index=False)
            logger.info("Saved to file.")
        else:
            logger.info("Skipped target date: {}".format(date))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BigQuery extractor module')
    parser.add_argument('start_date', help='Start date in Y-m-d, eg 2018-12-31')
    parser.add_argument('end_date', help='End date in Y-m-d, eg 2018-12-31')
    parser.add_argument('dest_dir', help='Specialized destination directory for resulting dataframe file(s).')
    parser.add_argument('filename', help='Naming convention for resulting dataframe file(s).')
    parser.add_argument('query', help='Name of query to use, within queries directory.')
    args = parser.parse_args()

    # Logger setup
    LOGGING_CONFIG = os.getenv("LOGGING_CONFIG")
    logging.config.fileConfig(LOGGING_CONFIG)
    logger = logging.getLogger('extract')

    # BQ PROJECT SETUP
    ProjectID = 'govuk-bigquery-analytics'
    KEY_DIR = os.getenv("BQ_KEY_DIR")
    key_path = os.path.join(KEY_DIR, os.listdir(KEY_DIR)[0])

    # DATA DIRECTORIES
    QUERIES_DIR = os.getenv("QUERIES_DIR")
    DATA_DIR = os.getenv("DATA_DIR")
    dest_dir = os.path.join(DATA_DIR, args.dest_dir)

    # DATAFRAME FILENAME(S)
    filename = args.filename

    # DATES TO EVALUATE
    start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(args.end_date, '%Y-%m-%d')
    date_list = list(map(lambda x: x.strftime("%Y-%m-%d"), pd.date_range(start_date, end_date).tolist()))

    # RESOLVE QUERY FROM ARG
    query_path = find_query(args.query, QUERIES_DIR)

    # Set up the thing
    if not os.path.isdir(dest_dir):
        logging.info("Specified destination directory does not exist, creating...")
        os.mkdir(DATA_DIR, args.dest_dir)

    logger.info(
        "\n======\nStart date: {} \nEnd date: {} \nDestination directory: {}\
         \nFilename: {} \nQuery: {}\n======\n".format(
            start_date,
            end_date,
            dest_dir,
            filename,
            query_path))

    if query_path is not None:
        logger.info("Specified query exists, running...")
        query = read_query(query_path)
        looped_query(query, date_list, [], ProjectID, key_path, dest_dir, filename, "legacy")
