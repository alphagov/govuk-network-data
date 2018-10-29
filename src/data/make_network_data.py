import pandas as pd
import
src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(src, "data"))
sys.path.append(os.path.join(src, "features"))
import multiprocess_utils as multi_utils
import preprocess as prep

COLUMNS_TO_KEEP = set(['Page_List_NL','Occurrences'])


def read_file(filename):
    df = pd.read_csv(filename, compression="gzip")
    columns = set(df.columns.values)
    df.drop(list(columns-COLUMNS_TO_KEEP), axis=1, inplace=True)
    return df


def unique_pages(user_journey_df):
    user_journey_df['subpaths'] = user_journey_df['Page_List_NL'].

if __name__ == "__main__":
    print("hey")
