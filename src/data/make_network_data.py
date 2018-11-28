import argparse
import gzip
import logging.config
import os
import sys
from ast import literal_eval
from collections import Counter
import re


import pandas as pd

src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(src, "data"))
import preprocess as prep

COLUMNS_TO_KEEP = ['Page_List_NL','PageSequence','Page_Seq_NL', 'Page_List', 'Occurrences', 'Page_Seq_Occurrences', 'Occurrences_NL']


def read_file(filename):
    """
    Read a dataframe compressed csv file, init as dataframe, drop unnecessary columns, prepare target columns
    to be evaluated as lists with literal_eval.
    :param filename: processed_journey dataframe
    :return: processed for list-eval dataframe
    """
    logger.debug("Reading file {}...".format(filename))
    df = pd.read_csv(filename, compression="gzip")
    columns = set(df.columns.values)
    # print(columns)
    df.drop(list(columns - set(COLUMNS_TO_KEEP)), axis=1, inplace=True)
    for column in COLUMNS_TO_KEEP:
        if isinstance(df[column].iloc[0], str) and any(["," in val for val in df[column].values]):
            logger.debug("Working on literal_eval for \"{}\"".format(column))
            df[column] = df[column].map(literal_eval)
    return df


def compute_occurrences(user_journey_df):
    user_journey_df['Page_Seq_Occurrences'] = user_journey_df.groupby('PageSequence')['Occurrences'].transform(
        'sum')
    user_journey_df['Occurrences_NL'] = user_journey_df.groupby('Page_Seq_NL')['Occurrences'].transform('sum')


def generate_subpaths(user_journey_df):
    """
    Compute lists of subpaths ie node-pairs/edges (where a node is a page) from both original and de-looped page_lists
    (page-hit only journeys)
    :param user_journey_df: user journey dataframe
    :return: inplace assign new columns
    """
    logger.debug("Setting up sub-paths column...")
    user_journey_df['Subpaths'] = user_journey_df['Page_List'].map(prep.subpaths_from_list)
    logger.debug("Setting up de-looped sub-paths column...")
    user_journey_df['Subpaths_NL'] = user_journey_df['Page_List_NL'].map(prep.subpaths_from_list)


def edgelist_from_subpaths(user_journey_df, delooped=False):
    """
    Generate a counter that represents the edge list. Keys are edges (node pairs) which represent a user going from
    first element of pair to second one), values are a sum of journey occurrences (de-looped occurrences since current
    computation is based on de-looped subpaths), ie number of times a user/agent went from one page (node) to another.
    :param user_journey_df: user journey dataframe
    :return: edgelist counter
    """
    subpath_default = 'Subpaths'
    occurrences_default = 'Page_Seq_Occurrences'

    if delooped:
        logger.debug("Creating edge list from de-looped journeys (based on Subpaths_NL) ...")
        subpath_default = 'Subpaths_NL'
        occurrences_default = 'Occurrences_NL'
    else:
        logger.debug("Creating edge list from original journeys (based on Subpaths) ...")

    edgelist_counter = Counter()

    node_id = {}
    id = 0
    for i, row in user_journey_df.iterrows():
        for edge in row[subpath_default]:
            edgelist_counter[tuple(edge)] += row[occurrences_default]
            for node in edge:
                if node not in node_id.keys():
                    node_id[node] = id
                    id += 1
    return edgelist_counter, node_id


def nodes_from_edgelist(edgelist):
    """
    Generate a node list (from edges). Internally represented as a set, returned as alphabetically sorted list
    :param edgelist: list of edges (node-pairs)
    :return: sorted list of nodes
    """
    logger.debug("Creating node list...")
    node_list = set()
    for key, _ in edgelist.items():
        node_list.update(key)
    return sorted(list(node_list))


def write_node_edge_files(source_filename, dest_filename, delooped):
    """
    Read processed_journey dataframe file, preprocess, compute node/edge lists, write contents of lists to file.
    :param source_filename: dataframe to be loaded
    :param dest_filename: filename prefix for node and edge files
    """
    df = read_file(source_filename)
    generate_subpaths(df)
    if any(re.search("Occurrences_NL|Page_Seq_Occurrences", col) for col in df.columns):
        compute_occurrences(df)
    edges, node_id = edgelist_from_subpaths(df, delooped)
    nodes = nodes_from_edgelist(edges)
    logger.info("Number of nodes: {} Number of edges: {}".format(len(nodes), len(edges)))
    logger.info("Writing edge list to file...")
    with gzip.open(dest_filename + "_edges.csv.gz", "w") as file:
        file.write("Source_node,Source_id,Destination_Node,Destination_id,Weight\n".encode())
        for key, value in edges.items():
            file.write("{},{},{},{},{}\n".format(key[0], node_id[key[0]], key[1], node_id[key[1]], value).encode())
    logger.info("Writing node list to file...")
    with gzip.open(dest_filename + "_nodes.csv.gz", "w") as file:
        file.write("Node,Node_id\n".encode())
        for node in nodes:
            file.write("{},{}\n".format(node, node_id[node]).encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Module that produces node and edge files given a user journey file.')
    parser.add_argument('source_directory', help='Source directory for input dataframe file(s).')
    parser.add_argument('input_filename', help='Source directory for input dataframe file(s).')
    parser.add_argument('dest_directory', help='Specialized destination directory for output files.')
    parser.add_argument('output_filename', help='Naming convention for resulting node and edge files.')
    parser.add_argument('-q', '--quiet', action='store_true', default=False, help='Turn off debugging logging.')
    # TODO: add option to choose between original and de-looped journeys (to compute edges)
    parser.add_argument('-d', '--delooped', action='store_true', default=False,
                        help='Use delooped journeys for edge and weight computation')

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
        logger.info("Working on file: {}".format(input_filename))
        write_node_edge_files(input_filename, output_filename, args.delooped)
    else:
        logger.debug("Specified filename does not exist: {}".format(input_filename))
