import re
import numpy as np

"""

"""


def clean_tuple(pe_str_tuple):
    """
    Transform raw SQL BigQuery string to list of page/event tuples
    :param pe_str_tuple:
    :return:
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

    bq_journey_string = bq_journey_string.replace(">>iii....", "")
    page_event_list = []
    for hit in bq_journey_string.split(">>"):
        # Old delimiter: split("//")
        page_event_tup = clean_tuple(hit.split("<<"))
        if len(page_event_tup) == 2:
            page_event_list.append(tuple(page_event_tup))
        else:
            print("error")
            print(bq_journey_string)
            print(page_event_tup)
            # if any(["http" in tup for tup in page_event_tup]):
            #     page_event_list.append((page_event_tup[0], "::".join(page_event_tup[1:])))
            # else:
            #     page_event_list.append(("::".join(page_event_tup[:-1]), page_event_tup[-1]))
    return page_event_list


def reindex_pe_list(page_event_list):
    """
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
    :return: tuple(eventCat,EventAct)
    """
    event_tuple = tuple(event_str.split("<:<"))
    if len(event_tuple) > 2:
        print("more than two")
        print(event_tuple)
        # event_tuple = (event_tuple[0], "<<".join(event_tuple[1:]))
    if len(event_tuple) == 1:
        print(event_str)
        print("this is a one", event_tuple)
    return event_tuple


def extract_pe_components(page_event_list, i):
    """

    :param page_event_list:
    :param i:
    :return:
    """
    hit_list = []
    for page_event in page_event_list:
        if i == 0:
            if page_event[1] == "NULL<:<NULL":
                hit_list.append(page_event[0])
        else:
            hit_list.append(split_event(page_event[i]))
    return hit_list





def collapse_loop(page_list):
    """

    :param page_list:
    :return:
    """
    return [node for i, node in enumerate(page_list) if i == 0 or node != page_list[i - 1]]


# Network things, should probably be moved somewhere else
def start_end_page(page_list):
    """

    :param page_list:
    :return:
    """
    return page_list[0], page_list[-1]


def subpaths_from_list(page_list):
    """

    :param page_list:
    :return:
    """
    return [[page, page_list[i + 1]] for i, page in enumerate(page_list) if i < len(page_list) - 1]


def start_page(page_list):
    return page_list[0]


def start_end_subpath_list(subpath_list):
    return subpath_list[0][0], subpath_list[-1][-1]



