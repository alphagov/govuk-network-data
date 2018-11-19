import gzip
import logging.config
import os
import sys

import argparse

src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(src, "data"))
sys.path.append(os.path.join(src, "features"))
import preprocess as prep
import build_features as feat


def count_lines(filepath):
    logging.info("Counting lines...")
    with gzip.open(filepath, "r") as f:
        for i, l in enumerate(f):
            pass
    return i


def read_write_file(input_path,output_path):
    with gzip.open(output_path, "w") as file:
        with gzip.open(input_path, "r") as read:
            df_columns = read.readline().decode().replace("\n", "").split("\t")
            print(df_columns)
            if 'PageSequence' not in df_columns:
                other_columns.insert(2, 'PageSequence')
            sequence_index = df_columns.index("Sequence")
            logging.info("Write headers...")
            all_cols = df_columns + other_columns
            print(all_cols)
            write_to_file = "\t".join(all_cols) + "\n"
            logging.info("Iteration...")

            for i, line in enumerate(read):

                line = line.decode().replace("\n", "")
                row = line.split("\t")

                for element in row:
                    if not isinstance(element, str):
                        write_to_file += str(element) + "\t"
                    else:
                        write_to_file += "\"" + str(element) + "\"" + "\t"

                sequence = row[sequence_index]
                # logging.info("Writing sequence columns:"
                # Page event
                page_event_list = prep.bq_journey_to_pe_list(sequence)
                write_to_file += "\"" + str(page_event_list) + "\"" + "\t"
                # Page list
                page_list = prep.extract_pe_components(page_event_list, 0)
                write_to_file += "\"" + str(page_list) + "\"" + "\t"

                if 'PageSequence' not in df_columns:
                    write_to_file += "\"" + ">>".join(page_list) + "\"" + "\t"

                # logging.info("Writing events columns"
                event_list = prep.extract_pe_components(page_event_list, 1)
                write_to_file += "\"" + str(event_list) + "\"" + "\t"
                write_to_file += "\"" + str(feat.count_event_cat(event_list)) + "\"" + "\t"
                write_to_file += "\"" + str(feat.aggregate_event_cat(event_list)) + "\"" + "\t"
                write_to_file += "\"" + str(feat.aggregate_event_cat_act(event_list)) + "\"" + "\t"

                # logging.info("Writing taxon_list"
                write_to_file += "\"" + str(prep.extract_cd_components(page_event_list, 2)) + "\"" + "\t"
                write_to_file += "\"" + str(prep.extract_pcd_list(page_event_list, 2)) + "\"" + "\t"

                # logging.info("Writing loop column stuff"
                de_looped = prep.collapse_loop(page_list)
                write_to_file += "\"" + str(de_looped) + "\"" + "\t"
                write_to_file += "\"" + ">>".join(de_looped) + "\""

                write_to_file += "\n"

                if i % 500000 == 0:
                    logging.info("At index: {}".format(i))
                    file.write(write_to_file.encode())
                    write_to_file = ""
                    file.flush()

                if i == num_lines - 1 and write_to_file != "":
                    logging.info("At index via last: {}".format(i))
                    file.write(write_to_file.encode())
                    write_to_file = ""
                    file.flush()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Module that produces a metadata-aggregated and '
                                                 'preprocessed dataset (.csv.gz), given a merged file.')
    parser.add_argument('in_file', help='Source directory for input dataframe file(s).')

    args = parser.parse_args()

    LOGGING_CONFIG = os.getenv("LOGGING_CONFIG")
    logging.config.fileConfig(LOGGING_CONFIG)
    logger = logging.getLogger('read_write')

    DATA_DIR = os.getenv("DATA_DIR")

    read_path = os.path.join(DATA_DIR, "output", args.in_file+".csv.gz")
    write_path = os.path.join(DATA_DIR, "output", args.in_file.replace("merged", "preprocessed"))

    other_columns = ['Page_Event_List', 'Page_List', 'Event_List', 'num_event_cats', 'Event_cats_agg',
                     'Event_cat_act_agg', 'Taxon_List', 'Taxon_Page_List', 'Page_List_NL', 'Page_Seq_NL']

    if os.path.isfile(read_path):
        logging.info("Reading from \"{}\" and writing to \"{}\"...".format(read_path, write_path))
        num_lines = count_lines(read_path)
        logging.info("Number of rows in dataframe: {}".format(num_lines))
        logging.info("Reading, processing, writing file...")
        read_write_file(read_path, write_path)
    else:
        logging.error("Input file \"{}\" does not exist.".format(read_path))
