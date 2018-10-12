import argparse
import datetime
import fnmatch
import logging.config
import os
import traceback

import pandas as pd


def find_query(query_arg, query_dir):
    '''(str, str) -> str
    Return the relative paths of files in query_dir that contain query_arg string.
    >>>find_query('work', './')
    './work'
    '''
    for file in os.listdir(query_dir):
        if fnmatch.fnmatch(file, "*" + query_arg + "*"):
            return os.path.join(query_dir, file)


def read_query(filepath):
    """(str) -> str
    Opens the file at filepath for reading, removing /n
    before rejoining seperate lines with " " seperator.
    """
    with open(filepath, 'r') as file:
        lines = " ".join(line.strip("\n") for line in file)
    return lines


def change_timestamp(x, date, dialect):
    if dialect == "standard":
        return x.replace("TIME_STAMP", date.replace("-", ""))
    else:
        change = str("TIMESTAMP(\"") + date + "\"), " + str("TIMESTAMP(\"") + date + "\"))"
        return x.replace("TIME_STAMP", change)


def looped_query(query_from_file, date_range, exclude_dates, project_id, key_path, destination_dir, filename_stub,
                 dialect="legacy"):
    runs = len(date_range) - len(exclude_dates)

    logging.info(query_from_file)

    for i, date in enumerate(date_range):
        logger.info("RUN {} OUT OF {}".format(str(i + 1), runs))
        if date not in exclude_dates:
            df_in = None
            logger.info("Working on: {}".format(date))
            logger.info("Query start...")
            query_for_paths = change_timestamp(query_from_file, date, dialect)

            try:
                df_in = pd.io.gbq.read_gbq(query_for_paths,
                                           project_id=project_id,
                                           reauth=False,
                                           # verbose=True,
                                           private_key=key_path,
                                           dialect=dialect)
            except Exception as e:
                logging.error("Oops, gbq failed.\n======\n {} \n======\n".format(traceback.format_exc()))

            if df_in is not None:
                file_name = os.path.join(destination_dir, filename_stub + "_" + str(date) + '.csv.gz')
                logger.info("Saving at: {}".format(file_name))
                df_in.to_csv(file_name, compression='gzip',
                             index=False)
                logger.info("Saved to file.")
            else:
                logger.error("Nothing to save, query failed.")

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
    key_file_path = os.path.join(KEY_DIR, os.listdir(KEY_DIR)[0])

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

    # If dest_dir doesn't exist, create it.
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
        looped_query(query, date_list, [], ProjectID, key_file_path, dest_dir, filename)
