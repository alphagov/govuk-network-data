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

    :param taxon_df:
    :param nodes_df:
    :return:
    """

    column_list = ['content_id', 'title', 'base_path', 'level', 'parents', 'level1_parent']
    taxon_level_df = pd.DataFrame(columns=column_list)

    taxon_set = build_taxon_set(nodes_df.Node_Taxon)

    for content_id in taxon_set:
        if taxon_df[taxon_df.content_id == content_id].shape[0] > 0:
            title = taxon_df[taxon_df.content_id == content_id].iloc[0].title
            base_path = taxon_df[taxon_df.content_id == content_id].iloc[0].base_path
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
                                                                       base_path,
                                                                       current_level,
                                                                       parent_list.values,
                                                                       level1_par]], columns=column_list)])
    taxon_level_df.reset_index(drop=True, inplace=True)
    taxon_level_df.drop_duplicates(subset="content_id", keep="first", inplace=True)
    return taxon_level_df


def add_taxon_basepath_to_df(node_df, taxons_df):
    """

    :param node_df:
    :param taxons_df:
    :return:
    """
    content_basepath_dict = dict(zip(taxons_df.content_id, taxons_df.base_path))
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
        description='Module to translate taxon content_ids in node file to names. Also recursively compute parents.')
    parser.add_argument('node_filename', help='Specialized destination directory for output dataframe file.')
    parser.add_argument('taxon_dir', help='File location of taxon json.')
    parser.add_argument('taxon_output_filename', default="",
                        help='Naming convention for resulting merged dataframe file.')
    parser.add_argument('-q', '--quiet', action='store_true', default=False, help='Turn off debugging logging.')
    args = parser.parse_args()

    DATA_DIR = os.getenv("DATA_DIR")
    nodes_path = os.path.join(DATA_DIR, "output", args.node_filename + ".csv.gz")
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
        taxon_output_path = os.path.join(DATA_DIR, "output", args.taxon_output_filename)
        taxon_df.to_csv(taxon_output_path, compression="gzip", index=False)
    else:
        print("files do not exist {} {}, {} {}".format(taxons_path,os.path.exists(taxons_path),nodes_path,os.path.exists(nodes_path)))
