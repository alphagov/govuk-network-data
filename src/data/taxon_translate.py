import argparse
import os

import pandas as pd


def recursive_parenting(taxon_df, content_id, parent_content_id, parent_list):
    """
    Recursively compute a taxon's parents
    :param taxon_df: taxon dataframe from content tagger (taxon json file)
    :param content_id: target taxon content_id
    :param parent_content_id: target taxon's parent content_id
    :param parent_list: incrementing list of parents
    :return: recursive call, aggregated list of parents if top level
    """
    if isinstance(parent_content_id, float) and len(parent_list) == 0:
        return []
    elif isinstance(parent_content_id, float):
        return [[parent_taxon, i + 1] for i, parent_taxon in enumerate(reversed(parent_list))]
    else:
        content_id = parent_content_id
        parent_content_id = taxon_df[taxon_df.content_id == parent_content_id].iloc[0].parent_content_id
        title = taxon_df[taxon_df.content_id == content_id].iloc[0].title
        parent_list.append([content_id, parent_content_id, title])
        return recursive_parenting(taxon_df, content_id, parent_content_id, parent_list)


def build_taxon_set(taxon_series):
    """
    Build set of unique taxons from the input taxon Series induced from the network node dataframe.
    :param taxon_series: Taxon column from the network node df, list of taxon content_id lists.
    :return: unique set containing taxon content_ids from nodes
    """
    return set([content_id for taxon_list in taxon_series for content_id in taxon_list])


def map_taxon_content_ids(target_taxon_df, nodes_df):
    """
    Extract taxons from node dataframe as a unique set of taxon content_ids and then compute their title, base_path
    (main component to be returned), level, parents (if any, else NaN) and finally the top-most parent.
    :param target_taxon_df: taxon dataframe from content tagger (taxon json file)
    :param nodes_df: dataframe with network nodes
    :return: dataframe containing taxon information
    """

    column_list = ['content_id', 'title', 'base_path', 'level', 'parents', 'level1_parent']
    taxon_level_df = pd.DataFrame(columns=column_list)

    taxon_set = build_taxon_set(nodes_df.Node_Taxon)

    for content_id in taxon_set:
        if target_taxon_df[target_taxon_df.content_id == content_id].shape[0] > 0:
            title = target_taxon_df[target_taxon_df.content_id == content_id].iloc[0].title
            base_path = target_taxon_df[target_taxon_df.content_id == content_id].iloc[0].base_path
            parent_list = pd.Series(recursive_parenting(target_taxon_df, content_id,
                                                        target_taxon_df[
                                                            target_taxon_df.content_id == content_id].parent_content_id.values[
                                                            0], []))
            current_level = len(parent_list) + 1
            level1_par = title
            if len(parent_list.values) > 0:
                level1_par = parent_list.values[0][0][2]
            taxon_level_df = pd.concat([taxon_level_df, pd.DataFrame([[content_id,
                                                                       title,
                                                                       base_path,
                                                                       current_level,
                                                                       parent_list.values,
                                                                       level1_par]], columns=column_list)])
    taxon_level_df.reset_index(drop=True, inplace=True)
    taxon_level_df.drop_duplicates(subset="content_id", keep="first", inplace=True)
    return taxon_level_df


def add_taxon_basepath_to_df(node_df, taxon_level_df):
    """
    Compute appropriate taxon base_paths for list of taxon content_ids and add to node dataframe.
    :param node_df: dataframe with network nodes
    :param taxon_level_df: dataframe containing taxon information (taxons nodes are tagged with)
    :return: augmented node dataframe, including taxon base_paths
    """
    content_basepath_dict = dict(zip(taxon_level_df.content_id, taxon_level_df.base_path))
    taxon_name_list = []
    for tup in node_df.itertuples():
        taxon_basepath = []
        for taxon in tup.Node_Taxon:
            if taxon in content_basepath_dict.keys():
                taxon_basepath.append(content_basepath_dict[taxon])
        taxon_name_list.append(taxon_basepath)
    node_df['Node_Taxon_basepath'] = taxon_name_list
    return node_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Module to translate taxon content_ids in node file to taxon base paths. In addition, ecursively '
                    'compute taxon'
                    'level, parents and top-most parents.')
    parser.add_argument('node_filename', help='Node input filename.')
    parser.add_argument('taxon_dir', help='Directory containing taxon json file.')
    parser.add_argument('taxon_output_filename', default="",
                        help='Naming convention for resulting taxon dataframe file. Includes taxons that nodes in node '
                             'file are tagged to.')
    parser.add_argument('-q', '--quiet', action='store_true', default=False, help='Turn off debugging logging.')
    args = parser.parse_args()

    DATA_DIR = os.getenv("DATA_DIR")
    nodes_path = os.path.join(DATA_DIR, "processed_data", args.node_filename + ".csv.gz")
    taxons_path = os.path.join(args.taxon_dir, "taxons.json.gz")

    if os.path.exists(taxons_path) and os.path.exists(nodes_path):
        print("Working on: {}".format(taxons_path))
        taxons_json_df = pd.read_json(taxons_path, compression="gzip")
        print("Working on: {} ".format(nodes_path))
        nodes_df = pd.read_csv(nodes_path, sep="\t", compression="gzip")

        taxon_df = map_taxon_content_ids(taxons_json_df, nodes_df)
        nodes_df = add_taxon_basepath_to_df(nodes_df, taxon_df)

        # overwrite option? should it be an option or default?
        nodes_df.to_csv(nodes_path.replace(".csv.gz", "_taxon_base_path.csv.gz"), sep="\t", compression="gzip",
                        index=False)
        # save taxon-specific dataframe
        taxon_output_path = os.path.join(DATA_DIR, "processed_data", args.taxon_output_filename)
        taxon_df.to_csv(taxon_output_path, compression="gzip", index=False)
    else:
        print("Files do not exist:\n {}: {},\n {}: {}".format(taxons_path, os.path.exists(taxons_path), nodes_path,
                                                              os.path.exists(nodes_path)))
