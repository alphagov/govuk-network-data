import argparse
import os

import pandas as pd


def recursive_parenting(df, content_id, parent_content_id, parent_list):
    """

    :param df:
    :param content_id:
    :param parent_content_id:
    :param parent_list:
    :return:
    """
    if isinstance(parent_content_id, float) and len(parent_list) == 0:
        return []
    elif isinstance(parent_content_id, float):
        return [[thing, i + 1] for i, thing in enumerate(reversed(parent_list))]
    else:
        content_id = parent_content_id
        parent_content_id = df[df.content_id == parent_content_id].iloc[0].parent_content_id
        title = df[df.content_id == content_id].iloc[0].title
        parent_list.append([content_id, parent_content_id, title])
        return recursive_parenting(df, content_id, parent_content_id, parent_list)


def build_taxon_set(taxon_series):
    """

    :param taxon_series:
    :return:
    """
    return set([content_id for taxon_list in taxon_series for content_id in taxon_list])


def map_taxon_content_ids(taxon_df, nodes_df):
    """

    :param nodes_list:
    :param taxon_path:
    :return:
    """

    column_list = ['content_id', 'title', 'level', 'parents', 'level1_parent']
    taxon_level_df = pd.DataFrame(columns=column_list)

    taxon_set = build_taxon_set(nodes_df.Node_Taxon)

    for content_id in taxon_set:
        if taxon_df[taxon_df.content_id == content_id].shape[0] > 0:
            title = taxon_df[taxon_df.content_id == content_id].iloc[0].title
            parent_list = pd.Series(recursive_parenting(taxon_df, content_id,
                                                        taxon_df[
                                                            taxon_df.content_id == content_id].parent_content_id.values[
                                                            0], []))
            current_level = len(parent_list) + 1
            level1_par = title
            if len(parent_list.values) > 0:
                level1_par = parent_list.values[0][0][2]
            taxon_level_df = pd.concat([taxon_level_df, pd.DataFrame([[content_id,
                                                                       title,
                                                                       current_level,
                                                                       parent_list.values,
                                                                       level1_par]], columns=column_list)])
    taxon_level_df.reset_index(drop=True, inplace=True)
    taxon_level_df.drop_duplicates(subset="content_id", keep="first", inplace=True)
    return taxon_level_df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Module to translate taxon content_ids in node file to names. Also recursively compute parents.')
    parser.add_argument('taxon_dir', help='File location of taxon json.')
    parser.add_argument('input_filename', help='Specialized destination directory for output dataframe file.')
    parser.add_argument('output_filename', default="", help='Naming convention for resulting merged dataframe file.')
    parser.add_argument('-q', '--quiet', action='store_true', default=False, help='Turn off debugging logging.')
    args = parser.parse_args()

    taxons_path = os.path.join(args.taxon_dir, "taxons.json.gz")
    nodes_path = os.path.join("", args.input_filename)

    if os.path.exists(taxons_path) and os.path.exists(nodes_path):
        taxon_df = pd.read_json(taxons_path, compression="gzip")
        node_df = pd.read_csv(nodes_path, sep="\t", compression="gzip")
        map_taxon_content_ids(taxon_df, node_df)
