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


if __name__ == "__main__":
    print()