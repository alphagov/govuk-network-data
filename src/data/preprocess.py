import re
import numpy as np


def clean_tuple(pe_str_tuple):
    """
    TODO: not sure why this is here... maybe quotes break things
    Transform raw SQL BigQuery string to list of page/event tuples:
    :param pe_str_tuple: a tuple, ideally length 2 (page1,eventCategory1<:<eventAction1)
    :return: tuple with quotes removed from each element
    """
    # if "http" not in tupes else re.sub(r"\"|\'", "", tupes)
    return [re.sub(r"\"|\'", "", tupes) for tupes in pe_str_tuple]


def bq_journey_to_pe_list(bq_journey_string):
    """
    Split a BigQuery string page1<<eventCategory1<:<eventAction1>>page2<<eventCategory2<:<eventAction2>>... into a
    list of tuples page_event_list = [(page1,eventCategory1<:<eventAction1), (page2,eventCategory2<:<eventAction2),
    ...] The event string eg eventCategory1<:<eventAction1 is further split at a later stage. Nothing is dropped,
    number of page1<<eventCategory1<:<eventAction1 instances and number of page-event tuples should be equal.
    :param bq_journey_string:
    :return: The list of page-event tuples.
    """
    # TODO: fix line below, for now it replaces string in a weird /search query within journey (uncommon)
    bq_journey_string = bq_journey_string.replace(">>iii....", "")
    page_event_list = []
    for hit in bq_journey_string.split(">>"):
        # Old delimiter: split("//")
        page_event_tup = clean_tuple(hit.split("<<"))
        # For len==3 Taxon present within bq_journey_string
        if len(page_event_tup) == 2 or len(page_event_tup) == 3:
            page_event_list.append(tuple(page_event_tup))
        else:
            # TODO remove in future
            print("Error, tuple split generated too many elements.")
            print("Overall BigQuery string:", bq_journey_string)
            print("Too long page_event tuple:", page_event_tup)
            # Add in dummy variable for debugging and to avoid empty lists
            # Useful for inspecting real data, uncomment if desired
            # page_event_list.append(("page1","eventCategory<:<eventAction"))
            # TODO remove in future
            # if any(["http" in tup for tup in page_event_tup]):
            #     page_event_list.append((page_event_tup[0], "::".join(page_event_tup[1:])))
            # else:
            #     page_event_list.append(("::".join(page_event_tup[:-1]), page_event_tup[-1]))
    return page_event_list


def reindex_pe_list(page_event_list):
    """
    TODO: not used right now
    Reindex and de-loop page_event_list if necessary. Used when absolute hit position within journey
    needs to be evaluated.
    If that's the case, page_list and event_list generators should be run based on this list, not
    page_event_list itself.
    :param page_event_list:
    :return:
    """
    if len(page_event_list) > 0:
        position_dict = [(0, page_event_list[0])]
        for i, (page, event) in enumerate(page_event_list[1:]):
            # print(i)
            if page != page_event_list[i][0]:
                index = position_dict[-1][0]
                position_dict.append((index + 1, (page, event)))
            elif page == page_event_list[i][0] and (event != position_dict[-1][1][1]):
                position_dict.append((position_dict[-1][0], (page, event)))
        return position_dict
    return np.NaN


def split_event(event_str):
    """
    Split eventCategory<:<eventAction pair into a tuple. The if conditions are superfluous, there in the case
    something breaks due to delimiter being present in the str. (rare now)
    :param event_str: string tuple from
    page_event_list.
    EVENT::NULL::NULL
   (EVENT_NULL,EVENT_NULL)
    :return: tuple(eventCat,EventAct)
    """
    event_tuple = tuple(event_str.split("<:<"))
    if len(event_tuple) == 2:
        return event_tuple
    if len(event_tuple) == 3:
        if "NULL" in event_tuple[1]:
            return tuple((event_tuple[0] + "_" + event_tup for event_tup in event_tuple[1:]))
        else:
            return tuple(event_tuple[1:])
    if len(event_tuple) > 3:
        print("Event tuple has more than two elements:", event_tuple)
        print("Original:", event_str)
        # event_tuple = (event_tuple[0], "<<".join(event_tuple[1:]))
    if len(event_tuple) == 2:
        print("Event tuple has only one element:", event_tuple)
        print("Original:", event_str)


def extract_pe_components(page_event_list, i):
    """
    Extract page_list or event_list from page_event_list
    :param page_event_list: list of (page,event) tuples
    :param i: 0 for page_list 1, for event_list
    :return: appropriate hit_list
    """
    hit_list = []
    # page_event is a tuple
    for page_event in page_event_list:
        if i == 0 and page_event[1] == "PAGE<:<NULL<:<NULL":
            hit_list.append(page_event[i])
        elif i == 1:
            hit_list.append(split_event(page_event[i]))
    return hit_list


def taxon_string_to_list(taxon_string):
    return tuple(taxon_string.split(","))


def extract_cd_components(page_event_list, i):
    """
    TODO: probably add functionality as a condition to extract_pe_components
    Extract cd_list from page_event_cd_list
    :param page_event_list: list of (page,event) tuples
    :param i: 0 for page_list 1, for event_list
    :return: appropriate hit_list
    """
    # page_event_cd is a tuple
    # For initial taxon implementation
    return [page_event_cd[i] for page_event_cd in page_event_list]


def extract_page_cd_components(page_event_list, i):
    """
    TODO: probably add functionality as a condition to extract_pe_components
    Extract cd_list from page_event_cd_list
    :param page_event_list: list of (page,event) tuples
    :param i: 0 for page_list 1, for event_list
    :return: appropriate hit_list
    """
    # page_event_cd is a tuple
    # For initial taxon implementation
    return [(page_event_cd[0], page_event_cd[i]) for page_event_cd in page_event_list]


def collapse_loop(page_list):
    """
    Remove A>>A>>B page loops from page_list. Saved as new dataframe column.
    :param page_list: the list of pages to de-loop
    :return: de-loop page list
    """
    return [node for i, node in enumerate(page_list) if i == 0 or node != page_list[i - 1]]


# Network things, should probably be moved somewhere else
def start_end_page(page_list):
    """
    Find start and end pages (nodes) in a list of page hits
    :param page_list: list of page hits
    :return: start and end nodes
    """
    if len(page_list) == 1:
        return page_list[0]
    else:
        return page_list[0], page_list[-1]


def subpaths_from_list(page_list):
    """
    Build node pairs (edges) from a list of page hits
    :param page_list: list of page hits
    :return: list of all possible node pairs
    """
    return [[page, page_list[i + 1]] for i, page in enumerate(page_list) if i < len(page_list) - 1]


def start_page(page_list):
    """
    First page/node in a list of page hits
    :param page_list: list of page hits
    :return: First page
    """
    return page_list[0]


def end_page(page_list):
    """
    Last page/node in a list of page hits
    :param page_list: list of page hits
    :return: last page
    """
    return page_list[-1]


def start_end_subpath_list(subpath_list):
    """
    First and last page from list of node pairs
    :param subpath_list: list of node pairs
    :return: first and last page
    """
    return subpath_list[0][0], subpath_list[-1][-1]


def start_end_edges_subpath_list(subpath_list):
    """
    First/last node pairs (edges) from list of node pairs
    :param subpath_list: list of node pairs
    :return: first and last node pairs
    """
    return subpath_list[0], subpath_list[-1]
