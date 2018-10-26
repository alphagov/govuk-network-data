import numpy as np
import pandas as pd


def delete_vars(x):
    """
    Force object deletion
    :param x: object to delete
    """
    if isinstance(x, list):
        for xs in x:
            del xs
    del x


def compute_max_depth(test_list, chunks, depth, fewer_than_cpu):
    """
    Compute maximum recursive depth of process_dataframes, governs MAX_DEPTH global and at which point of execution
    one-off rows (based on Occurrence # of PageSequence) will be dropped.
    :param test_list: dummy list based on list of files to be read/processed.
    :param chunks: initial number of partitions
    :param depth: init = 0, increases with every recursive call
    :return: (int) maximum recursive depth
    """
    partitions = partition_list(test_list, chunks, fewer_than_cpu)
    if len(test_list) > 1:
        new_lst = [0 for _ in partitions]
        return compute_max_depth(new_lst, (lambda x: int(x / 2) if int(x / 2) > 0 else 1)(chunks), depth + 1, fewer_than_cpu)
    else:
        return depth


def compute_initial_chunksize(number_of_files, num_cpu):
    """

    :param num_cpu:
    :param number_of_files:
    :return:
    """
    if number_of_files > num_cpu:
        return int(number_of_files / 2)
    else:
        return number_of_files


def compute_batches(files, batchsize):
    """

    :param files:
    :param batchsize:
    :return:
    """

    if len(files) > int(np.ceil(batchsize * 1.5)):
        return True, merge_small_partition([files[i:i + batchsize] for i in range(0, len(files), batchsize)])
    else:
        return False, files


def merge_sliced_df(sliced_df_list: list, expected_size: int):
    """
    Merge dataframe slices (column pairs) when appropriate (codes match) and append to a list of merged dataframes.
    Due to order of columns, the Occurrences slice will be used as a basis for the merge.
    :param sliced_df_list: list of slices
    :param expected_size: number of dataframes that have been originally sliced
    :return: list of merged dataframes
    """
    final_list = [pd.DataFrame()] * expected_size
    # print([df.shape for i, df in sliced_df_list if i == 0])
    # i = dataframe code, dataframes may come from multiple files.
    for i, df in sliced_df_list:
        # print(df.columns)
        if len(final_list[i]) == 0:
            # print("new")
            final_list[i] = df.copy(deep=True)
        else:
            # print("merge")
            final_list[i] = pd.merge(final_list[i], df, how='left', on='Sequence')
    return final_list


def partition_list(dataframe_list: list, chunks: int, fewer_than_cpu):
    """
    Build a list of partitions from a list of dataframes. Based on indices.
    :param dataframe_list: list of dataframes
    :param chunks: number of indices lists to generate, len(partition_list)
    :return: partition list, list of lists containing indices
    """
    if chunks > 0:
        initial = [list(xs) for xs in np.array_split(list(range(len(dataframe_list))), chunks)]
        # print(initial)
        if len(initial) > 1 and not fewer_than_cpu:
            initial = merge_small_partition(initial)
        return initial
    else:
        return [[0]]


def merge_small_partition(partitions: list):
    """
    Merge small partitions of length 1 into previous partition, reduce number of recursive runs.
    :param partitions:
    :return:
    """
    to_merge = []
    for partition in partitions:
        if len(partition) == 1:
            to_merge.append(partition[0])
            partitions.remove(partition)
    if len(to_merge) >= 1:
        partitions[-1].extend(to_merge)
    return partitions


def slice_many_df(df_list, drop_one_offs, sliceable_cols, ordered=False):
    """
    Slice a list of dataframes into their columns. First list will consist of
    (df_number, [Sequence, PageSequence, Occurrences])
    slices, second list will consist of (df_number, [Sequence, AggregatableMetadata1]),
    (df_number, [Sequence, AggregatableMetadata2]) etc.
    Reduces size of dataframes passed on to worker processes, so they don't break.
    :param df_list:
    :param ordered:
    :return:
    """
    if not ordered:
        return [(i, df.iloc[:, ind].copy(deep=True)) for i, df in enumerate(df_list) for ind in
                slice_dataframe(df, drop_one_offs, sliceable_cols)]
    else:
        return [(i, df.iloc[:, ind].copy(deep=True)) for i, df in enumerate(df_list) for ind in
                slice_dataframe(df, drop_one_offs, sliceable_cols) if
                "Occurrences" in df.columns[ind]], [(i, df.iloc[:, ind].copy(deep=True)) for i, df in enumerate(df_list)
                                                    for ind in
                                                    slice_dataframe(df,drop_one_offs, sliceable_cols) if
                                                    "Occurrences" not in df.columns[ind]]


def slice_dataframe(df, drop_one_offs, sliceable_cols):
    """
    Computes the slices (column pairs) of dataframe
    :param df: dataframe to be sliced
    :return: list of dataframe slices
    """
    sliced_df = []
    for col in sliceable_cols:
        if col in df.columns:
            if col == "Occurrences":
                if drop_one_offs:
                    sliced_df.append(
                        [df.columns.get_loc("Sequence"), df.columns.get_loc("PageSequence"), df.columns.get_loc(col)])
                else:
                    sliced_df.append(
                        [df.columns.get_loc("Sequence"), df.columns.get_loc(col)])
            else:
                sliced_df.append([df.columns.get_loc("Sequence"), df.columns.get_loc(col)])
    return sliced_df
