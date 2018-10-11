# # -*- coding: utf-8 -*-

import argparse
import logging.config
import os
import pprint
import re
from collections import Counter


# Transform raw SQL BigQuery string to list of page/event tuples
def clean_tuple(x):
    return [re.sub(r":|\"|\'", "", xs) for xs in x]


def bq_journey_to_list(x):
    journey_list = []
    for xs in x.split(">>"):
        page_event_tup = clean_tuple(xs.split("::"))
        if len(page_event_tup) == 2:
            journey_list.append(tuple(page_event_tup))
        else:
            journey_list.append(("::".join(page_event_tup[:-1]), page_event_tup[-1]))
    return journey_list


# Requires rework: this function deloops the sequence (A >> A >> A => A).
# If you don't want this, uncomment the else.
# def deloop_event_page_list(x):
#     position_dict = []
#     for i,(page,event) in enumerate(x):
# #         print(i,(page,event))
#         if i==0 or page != x[i-1][0]:
#             index = i
#             if len(position_dict)>0:
#                 index = position_dict[-1][0]
#             position_dict.append((index+1,(page,event)))
#         elif  page == x[i-1][0] and event!='NULL//NULL':
#             prev_event = position_dict[-1]
#             position_dict.append((prev_event[0],(prev_event[1][0],event)))
#         elif  page == x[i-1][0] and event =='NULL//NULL' and position_dict[-1][0] != 'NULL//NULL':
#             position_dict.append((prev_event[0],(prev_event[1][0],event)))
# #         else:
# #             position_dict.append((index+1,(page,event)))
#     return position_dict


def split_event_cat_act(event_list):
    return [tuple(event.split("//")) for event in event_list]


def extract_pe_components(page_event_list, i):
    return [page_event[i] for page_event in page_event_list]


# Counts for events
def count_event_categories(event_list):
    return len(set([cat for cat, _ in event_list]))


def count_event_actions(event_list, kind, category):
    return [action for cat,action in event_list if cat == category].count(kind)


def aggregate_event_categories(event_list):
    return [(key, value) for key, value in Counter([cat for cat, _ in event_list]).items()]


def aggregate_event_actions(event_list):
    return [(key, value) for key, value in Counter([(cat, act) for cat, act in event_list]).items()]


# Transform metadata lists to dictionary aggregates
def list_to_dict(x):
    local_dict = Counter()
    for xs in x.split(','):
        local_dict[xs]+=1
    return list(local_dict.items())

# TODO: needs more work, right now it is dependant on hardcoded df column specification
def zip_aggregate_metadata(user_journey_df):
    col = []
    for tup in user_journey_df.itertuples():
        locs = tup.Locations.split(',')
        langs = tup.Languages.split(',')
        devs = tup.DeviceCategories.split(',')
        zipped_meta_counter = Counter()
        for loc,lang,dev in zip(locs,langs,devs):
            zipped_meta_counter[(loc,lang,dev)] += 1
        col.append(list(zipped_meta_counter.items()))

    user_journey_df['AggMeta'] = col


# Loop-related functions
def has_loop(page_list):
    return any(i == j for i, j in zip(page_list, page_list[1:]))


def collapse_loop(page_list):
    return [node for i, node in enumerate(page_list) if i == 0 or node != page_list[i - 1]]


def add_loop_columns(user_journey):
    logger.info("Collapsing loops...")
    user_journey['Sequence_List_No_Loops'] = user_journey['Sequence_List'].map(collapse_loop)
    # logger.info("Has loop...")
    # user_journey['Has_Loop'] = user_journey['Sequence_List'].map(has_loop)
    # logger.info("To string...")
    # user_journey['Sequence_No_Loops'] = user_journey['Sequence_List_No_Loops'].map(list_to_path_string)
    logger.info("Aggregating de-looped journey occurrences...")
    user_journey['Occurrences_No_Loop'] = user_journey.groupby('Sequence_No_Loop') \
        ['Occurrences'].transform('sum')


# repetitions
def has_repetition(journey_list):
    return len(set(journey_list)) != len(journey_list)


if __name__ == "__main__":
    print(split_event_cat_act(["ffyesno//yes","NULL//NULL","NULL//NULL"]))

    eventlist = [("NULL","NULL"),("ffyesno","yes"),("ffyesno","no"),("ffman","no"),("ffyesno","no")]

    print(aggregate_event_actions(eventlist))
    print(aggregate_event_categories(eventlist))
    
    # parser = argparse.ArgumentParser(description='BigQuery extractor module')
    # parser.add_argument('source_dir', help='Source directory for input dataframe file(s).')
    # parser.add_argument('dest_dir', help='Specialized destination directory for output dataframe file.')
    # parser.add_argument('filename', help='Naming convention for resulting merged dataframe file.')
    # args = parser.parse_args()
    #
    # Logger setup
    LOGGING_CONFIG = os.getenv("LOGGING_CONFIG")
    logging.config.fileConfig(LOGGING_CONFIG)
    logger = logging.getLogger('make_dataset')
    #
    # DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data')
    # source_dir = ""
    # dest_dir = ""
    #
    # logger.info("Loading data")
    #
    # to_load = [os.path.join(source_dir, file) for file in os.listdir(source_dir)]
    #
    # pprint.pprint(to_load)
