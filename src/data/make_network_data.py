import argparse
import gzip
import logging.config
import os
import re
import sys
from ast import literal_eval
from collections import Counter

import pandas as pd

src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(src, "data"))
import preprocess as prep

COLUMNS_TO_KEEP = ['Page_List', 'Page_List_NL', 'PageSequence', 'Page_Seq_NL', 'Occurrences', 'Page_Seq_Occurrences',
                   'Occurrences_NL']
NODE_ATTRIBUTES = ['Taxon_Page_List']
OCCURRENCES = ['Occurrences_NL', 'Page_Seq_Occurrences']


def read_file(filename, columns_to_read, collapse_search=False, use_delooped_journeys=False,
              drop_incorrect_occ=False, with_attribute=False):
    """
    Read a dataframe compressed csv file, init as dataframe, drop unnecessary columns, prepare target columns
    to be evaluated as lists with literal_eval.
    :param with_attribute:
    :param use_delooped_journeys:
    :param drop_incorrect_occ:
    :param filename: processed_journey dataframe
    :return: processed for list-eval dataframe
    """
    logger.debug("Reading file {}...".format(filename))
    df = pd.read_csv(filename, sep='\t', compression="gzip", skipinitialspace=True, usecols=columns_to_read)
    logger.debug("Read in {} columns...".format(df.columns))

    if drop_incorrect_occ and all(col in df.columns for col in OCCURRENCES):
        logger.debug("Dropping incorrect occurrence counts...")
        df.drop(['Occurrences_NL', 'Page_Seq_Occurrences'], axis=1, inplace=True)

    print(df.shape)
    print(df[df.Occurrences == 1].shape)
    # Sample 30% of one-off journeys and then use these indices to drop them
    indices = df[df.Occurrences == 1].sample(frac=0.3, random_state=1234).index
    print(len(indices))
    df.drop(indices, inplace=True)
    print(df.shape)

    logger.debug("Number of rows post one-off occurrence drop: {}".format(df.shape))

    if with_attribute:
        for attribute_column in NODE_ATTRIBUTES:
            logger.debug("Working on literal_eval for \"{}\"".format(attribute_column))
            df[attribute_column] = df[attribute_column].map(literal_eval)

    column_to_eval = 'Page_List'

    if use_delooped_journeys:
        column_to_eval = 'Page_List_NL'

    if isinstance(df[column_to_eval].iloc[0], str) and any(["," in val for val in df[column_to_eval].values]):
        logger.debug("Working on literal_eval for \"{}\"".format(column_to_eval))
        df[column_to_eval] = df[column_to_eval].map(literal_eval)

    if collapse_search:
        logger.debug("Collapsing /search nodes in \"{}\"".format(column_to_eval))
        df[column_to_eval] = df[column_to_eval].map(collapse_search_page)

    return df


def collapse_search_page(page_list):
    return [page for page in page_list if not (re.match(r"^/search[//?|/]\D*$", page) or page == "/search")]


def compute_occurrences(user_journey_df, page_sequence, occurrences):
    logging.debug("Computing specialized occurrences \"{}\" based on  \"{}\"...".format(occurrences, page_sequence))
    user_journey_df[occurrences] = user_journey_df.groupby(page_sequence)['Occurrences'].transform(
        'sum')


def generate_subpaths(user_journey_df, page_list, subpaths):
    """
    Compute lists of subpaths ie node-pairs/edges (where a node is a page) from both original and de-looped page_lists
    (page-hit only journeys)
    :param subpaths:
    :param page_list:
    :param user_journey_df: user journey dataframe
    :return: inplace assign new columns
    """
    logger.debug("Setting up \"{}\" based on  \"{}\"...".format(subpaths, page_list))
    user_journey_df[subpaths] = user_journey_df[page_list].map(prep.subpaths_from_list)


def edgelist_from_subpaths(user_journey_df, use_delooped_journeys=False):
    """
    Generate a counter that represents the edge list. Keys are edges (node pairs) which represent a user going from
    first element of pair to second one), values are a sum of journey occurrences (de-looped occurrences since current
    computation is based on de-looped subpaths), ie number of times a user/agent went from one page (node) to another.
    :param use_delooped_journeys:
    :param user_journey_df: user journey dataframe
    :return: edgelist counter
    """
    subpath_default = 'Subpaths'
    occurrences_default = 'Page_Seq_Occurrences'
    page_list_default = 'Page_List'
    page_sequence_default = 'PageSequence'

    if use_delooped_journeys:
        logger.debug("Creating edge list from de-looped journeys (based on Subpaths_NL) ...")
        subpath_default = 'Subpaths_NL'
        occurrences_default = 'Occurrences_NL'
        page_list_default = 'Page_List_NL'
        page_sequence_default = 'Page_Seq_NL'

    else:
        logger.debug("Creating edge list from original journeys (based on Subpaths) ...")

    if occurrences_default not in user_journey_df.columns:
        compute_occurrences(user_journey_df, page_sequence_default, occurrences_default)

    logger.debug("Dropping duplicates {}...".format(page_sequence_default))
    user_journey_df.drop_duplicates(page_sequence_default, keep="first", inplace=True)

    generate_subpaths(user_journey_df, page_list_default, subpath_default)
    edgelist_counter = Counter()

    ind_path = user_journey_df.columns.get_loc(subpath_default)
    ind_occ = user_journey_df.columns.get_loc(occurrences_default)

    for tup in user_journey_df.itertuples(index=False):
        for edge in tup[ind_path]:
            edgelist_counter[tuple(edge)] += tup[ind_occ]

    return edgelist_counter


def compute_node_attribute(user_journey_df):
    """

    :param user_journey_df:
    :return:
    """
    logger.debug("Identifying node taxons from \"Taxon_Page_List\"...")
    node_taxon_dict = {}
    for tup in user_journey_df.itertuples():
        for page, taxons in tup.Taxon_Page_List:
            if page not in node_taxon_dict.keys():
                node_taxon_dict[page] = taxons
    return node_taxon_dict


def nodes_from_edgelist(edgelist):
    """
    Generate a node list (from edges). Internally represented as a set, returned as alphabetically sorted list
    :param edgelist: list of edges (node-pairs)
    :return: sorted list of nodes
    """
    logger.debug("Creating node list...")
    nid = 0
    node_list = {}

    for keys, _ in edgelist.items():
        for key in keys:
            if key not in node_list.keys():
                node_list[key] = nid
                nid += 1
    return node_list


def compute_nodes_edges(source_filename, dest_filename, cols, collapse_search, use_delooped_journeys,
                        drop_incorrect_occ,
                        with_attribute):
    """
    Read processed_journey dataframe file, preprocess, compute node/edge lists, write contents of lists to file.
    :param collapse_search:
    :param with_attribute:
    :param drop_incorrect_occ:
    :param use_delooped_journeys:
    :param source_filename: dataframe to be loaded
    :param dest_filename: filename prefix for node and edge files
    """
    df = read_file(source_filename, cols, collapse_search, use_delooped_journeys, drop_incorrect_occ, with_attribute)
    edges = edgelist_from_subpaths(df, use_delooped_journeys)
    node_list = nodes_from_edgelist(edges)

    print(list(node_list.items())[0:10])

    default_edge_header = "Source_node\tSource_id\tDestination_node\tDestination_id\tWeight\n"
    default_node_header = "Node\tNode_id\n"
    node_attr = None

    if with_attribute:
        logger.debug("Creating node-attribute (taxon) dictionary...")
        node_attr = compute_node_attribute(df)
        default_edge_header = "Source_node\tSource_id\tDestination_node\tDestination_id\tWeight\tSource_Taxon\tDestination_Taxon\n"
        default_node_header = "Node\tNode_id\tNode_Taxon\n"

    logger.info("Number of nodes: {} Number of edges: {}".format(len(node_list), len(edges)))
    logger.info("Writing edge list to file...")

    edge_writer(dest_filename + "_edges.csv.gz", default_edge_header, edges, node_list, node_attr)
    node_writer(dest_filename + "_nodes.csv.gz", default_node_header, node_list, node_attr)


def node_writer(filename, header, node_id, node_attr):
    with gzip.open(filename, "w") as file:
        print(filename)
        file.write(header.encode())
        for node, nid in node_id.items():
            file.write("{}\t{}".format(node, nid).encode())
            if node_attr is not None:
                file.write("\t{}".format(node_attr[node]).encode())
            file.write("\n".encode())


def edge_writer(filename, header, edges, node_id, node_attr):
    with gzip.open(filename, "w") as file:
        print(filename)
        file.write(header.encode())
        for key, value in edges.items():
            file.write("{}\t{}\t{}\t{}\t{}".format(key[0], node_id[key[0]], key[1], node_id[key[1]], value).encode())
            if node_attr is not None:
                file.write("\t{}\t{}".format(node_attr[key[0]], node_attr[key[1]]).encode())
            file.write("\n".encode())


def check_header(filename):
    with gzip.open(filename, "rb") as reader:
        header = set(reader.readline().decode().replace("\n", "").split("\t"))
    return list(header.intersection(set(COLUMNS_TO_KEEP + NODE_ATTRIBUTES)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Module that produces node and edge files given a user journey file.')
    parser.add_argument('source_directory', default="", nargs="?", help='Source directory for input dataframe file(s).')
    parser.add_argument('input_filename', help='Source directory for input dataframe file(s).')
    parser.add_argument('dest_directory', default="", nargs="?",
                        help='Specialized destination directory for output files.')
    parser.add_argument('output_filename', help='Naming convention for resulting node and edge files.')
    parser.add_argument('-q', '--quiet', action='store_true', default=False, help='Turn off debugging logging.')
    parser.add_argument('-d', '--delooped', action='store_true', default=False,
                        help='Use delooped journeys for edge and weight computation')
    parser.add_argument('-i', '--incorrect', action='store_true', default=False,
                        help='Drop incorrect occurrences if necessary')
    parser.add_argument('-t', '--taxon', action='store_true', default=False,
                        help='Compute and include additional node attributes (only taxon for now).')
    parser.add_argument('-cs', '--collapse_search', action='store_true', default=False,
                        help='Remove /search? page hits.')
    parser.add_argument('-s', '--sampling', action='store_true', default=False,
                        help='Remove /search? page hits.')

    args = parser.parse_args()

    DATA_DIR = os.getenv("DATA_DIR")
    source_directory = os.path.join(DATA_DIR,
                                    args.source_directory if args.source_directory != "" else "processed_journey")
    input_filename = os.path.join(source_directory, (
        args.input_filename + ".csv.gz" if "csv.gz" not in args.input_filename else args.input_filename))
    dest_directory = os.path.join(DATA_DIR, args.dest_directory if args.dest_directory != "" else "processed_network")

    output_filename = os.path.join(dest_directory, args.output_filename)
    LOGGING_CONFIG = os.getenv("LOGGING_CONFIG")
    logging.config.fileConfig(LOGGING_CONFIG)
    logger = logging.getLogger('make_network_data')

    if args.quiet:
        logging.disable(logging.DEBUG)

    if os.path.exists(input_filename):
        logger.info("Working on file: {}".format(input_filename))
        logger.info("Using de-looped journeys: {}\nDropping incorrect occurrence counts: {}".format(args.delooped,
                                                                                                    args.incorrect))
        cols = check_header(input_filename)
        compute_nodes_edges(input_filename, output_filename, cols, args.collapse_search, args.delooped, args.incorrect,
                            args.taxon)
    else:
        logger.debug("Specified filename does not exist: {}".format(input_filename))
