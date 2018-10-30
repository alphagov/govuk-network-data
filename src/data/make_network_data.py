import argparse
import logging.config
import os
import sys
from ast import literal_eval
from collections import Counter

import pandas as pd

src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(src, "data"))
import preprocess as prep

COLUMNS_TO_KEEP = ['Page_List_NL', 'Page_List', 'Occurrences', 'Page_Seq_Occurrences', 'Occurrences_NL']


def read_file(filename):
    """

    :param filename:
    :return:
    """
    logging.debug("Reading file {}...".format(filename))
    df = pd.read_csv(filename, compression="gzip")
    columns = set(df.columns.values)
    df.drop(list(columns - set(COLUMNS_TO_KEEP)), axis=1, inplace=True)
    for column in COLUMNS_TO_KEEP:
        if isinstance(df[column].iloc[0], str) and any(["," in val for val in df[column].values]):
            logging.debug("Working on literal_eval for \"{}\"".format(column))
            df[column] = df[column].map(literal_eval)
    return df


def generate_subpaths(user_journey_df):
    """

    :param user_journey_df:
    :return:
    """
    logging.debug("Setting up sub-paths column...")
    user_journey_df['Subpaths'] = user_journey_df['Page_List'].map(prep.subpaths_from_list)
    logging.debug("Setting up de-looped sub-paths column...")
    user_journey_df['Subpaths_NL'] = user_journey_df['Page_List_NL'].map(prep.subpaths_from_list)


def edgelist_from_subpaths(user_journey_df):
    """

    :param user_journey_df:
    :return:
    """
    logging.debug("Creating edge list from de-looped journeys (based on Subpaths_NL) ...")
    edgelist_counter = Counter()
    for tup in user_journey_df.itertuples():
        for edge in tup.Subpaths_NL:
            edgelist_counter[tuple(edge)] += tup.Occurrences_NL
    return edgelist_counter


def nodes_from_edgelist(edgelist):
    """

    :param edgelist:
    :return:
    """
    logging.debug("Creating node list...")
    node_list = set()
    for key, _ in edgelist.items():
        node_list.update(key)
    return sorted(list(node_list))


def create_node_edge_files(source_filename, dest_filename):
    """

    :param source_filename:
    :param dest_filename:
    :return:
    """
    df = read_file(source_filename)
    generate_subpaths(df)
    edges = edgelist_from_subpaths(df)
    nodes = nodes_from_edgelist(edges)
    logging.info("Number of nodes: {} Number of edges: {}".format(len(nodes), len(edges)))
    logging.info("Writing edge list to file...")
    with open(dest_filename + "_edges.csv", "w") as file:
        file.write("Source node, Destination Node, Weight\n")
        for key, value in edges.items():
            file.write(key[0] + "," + key[1] + "," + str(value) + "\n")
    logging.info("Writing node list to file...")
    with open(dest_filename + "_nodes.csv", "w") as file:
        for node in nodes:
            file.write(node + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Module that produces node and edge files given a user journey file.')
    parser.add_argument('source_directory', help='Source directory for input dataframe file(s).')
    parser.add_argument('input_filename', help='Source directory for input dataframe file(s).')
    parser.add_argument('dest_directory', help='Specialized destination directory for output files.')
    parser.add_argument('output_filename', help='Naming convention for resulting node and edge files.')
    parser.add_argument('-q', '--quiet', action='store_true', default=False, help='Turn off debugging logging.')
    args = parser.parse_args()

    DATA_DIR = os.getenv("DATA_DIR")
    source_directory = os.path.join(DATA_DIR, args.source_directory)
    input_filename = os.path.join(source_directory, (
        args.input_filename + ".csv.gz" if "csv.gz" not in args.input_filename else args.input_filename))
    dest_directory = os.path.join(DATA_DIR, args.dest_directory)

    output_filename = os.path.join(dest_directory, args.output_filename)
    LOGGING_CONFIG = os.getenv("LOGGING_CONFIG")
    logging.config.fileConfig(LOGGING_CONFIG)
    logger = logging.getLogger('make_network_data')

    if args.quiet:
        logging.disable(logging.DEBUG)

    if os.path.exists(input_filename):
        logger.info("Working on file {}:".format(input_filename))
        create_node_edge_files(input_filename, output_filename)
    else:
        logger.debug("Specified filename does not exist {}:".format(input_filename))
