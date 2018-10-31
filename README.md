# govuk-network-data
> Data pipeline for extraction and preprocessing of BigQuery user journey data.

A data pipeline for extracting and preprocessing BigQuery user journey data. The data captures the sequences of pages visited by users and how often these journeys occur. Additional metadata and Event data (what users do on their journey) is also provided.  

# Installing / Getting started

You need access and a key to GOV.UK BigQuery analytics (govuk-bigguery-analytics) to extract raw data for this pipeline. People to ask about this are senior performance analysts.

Clone this repo and then set up your python 3.6.0 virtual environment 

## Python version
Python 3.6.0

## Virtual environment
Create a new python 3.6.0 virtual environment using your favourite virtual environment manager (you may need pyenv to specify python version). 

Source the environment variables from the .envrc file either using direnv (`direnv allow`) or `source .envrc`.  
You can check they're loaded using `echo $NAME_OF_ENVIRONMENT_VARIABLE` or `printenv`.

Then install required python packages:  

`pip install -r requirements.txt`

## Where to put your BigQuery key

`mkdir key`

then put the json file in there

# What this does

This package arms data scientists with the tools to answer the hardest questions that people are asking about the sequence of pages that users are visiting and the type of behaviour those users are displaying.  

* A data pipeline that produces data in a convenient format to explore the GOV.UK page sequences or journies that users travel in a session.   
* Express this data as a graph with pages visited expressed as nodes and directed movement between pages as edges.   

<img src="network_data_pipeline.png" width="200" height="300" />

# Extracting raw data from big query
This produces a compressed csv in the destination directory (raw_bq_extract) where each row is a specific user journey (including events). However this raw data is messy and needs preprocessing to be analytically useful (see next section: 'Converting raw big query data to processed_journey data').

- Run `python src/data/bq_extract_data.py --help` to list required positional arguments:  
  - __start_date__ - Start date in Y-m-d, eg 2018-12-31
  - __end_date__ - End date in Y-m-d, eg 2018-12-31
  - __dest_dir__ - Specialized destination directory for resulting dataframe
              file(s).
  - __filename__ - Naming convention for resulting dataframe file(s).
  - __query__ - Name of query to use, within queries directory.
- Other optional arguments:
  - The default SQL dialect is legacy so specify `--standard` if needed. 
  - Set verbosity as quiet `--quiet` to reduce logging output.

First, save your sql query 'query_name.sql' in the $QUERIES_DIR directory.  

Here's an example of a command execution: 

`python src/data/bq_extract_data.py 2018-10-18 2018-10-18 raw_output_dir test prelim_meta_standard_query_with_pageseq --standard`  
In the above example, the SQL query exists as 'prelim_meta_standard_query_with_pageseq.sql' in the $QUERIES_DIR directory.

# Converting raw big query data to processed_journey data

This creates a csv where each row is a processed user journey and has session information rolled into it.   
For example if 3 users went A -> B -> C on different devices, then this would be represented as a single row with a column containing a device dictionary (See table below: DeviceCategories).

This processing script can also merge different inputs such as data extracts from different days.

- Run `python src/data/make_dataset.py --help` to list required positional arguments:  
  - __source_directory__ - Source directory for input dataframe file(s).  
  - __dest_directory__ - Specialized destination directory for output dataframe. 
                        file.
  - __output_filename__ - Naming convention for resulting merged dataframe file.__start_date__ - Start date in Y-m-d, eg 2018-12-31

- Other optional arguments:

  - __-doo, --drop_one_offs__ - Drop rare journeys occurring only once per input file. If merging multiple inputs, the merge will occur before the count (so that single journeys occuring in each file are aggregated before counting). Then the default behaviour is to drop in batches, approximately every 3 days to fit in memory and compute constraints.  
  - __-kloo, --keep_len_one_only__ -Keep ONLY journeys with length 1 ie journeys visiting only one page.  
  - __-dlo, --drop_len_one__  Drop journeys with length 1 ie journeys visiting only one page.  
  - __-f FILENAME_STUB, --filename_stub FILENAME_STUB__ -If merging multiple inputs, filename_stub is the unique prefix in their filenames which identify this group of inputs.  
  - __-q, --quiet__ -Turn off debugging logging.  
  
Here's an example of a command execution:  
`python src/data/make_dataset.py raw_bq_extract processed_journey test_output -doo`

Here's some definitions of the columns in the resulting dataframe:

| Column  | Description  |
|:---|:---|
| Sequence  |  Big query generated sequence of events & page hits |
| PageSequence | sequence of pages without events separated by >> |
| Occurrences  |  Number of times (sessions) the Sequence was identified|
| Page_Seq_Occurences  | Number of times (sessions) the PageSequence was identified|
| DeviceCategories  |  List of tuples (dictionary-like) where the key is the device (str) and the value is the number of sequences performed on each device (int) |
| Dates |  List of tuples (dictionary-like) where the key is date (YYYYMMDD) and the value is the number of times the sequence occurred in that date (int) |
| Page_Event_List | from Sequence -> list of tuples of (page url, event). Where it's a page hit, event==PAGE_NULL |
| Page_List | List of urls from PageSequence|
| PageSequence_internal |For debugging: will be dropped|
| Event_List | list of tuples each containing (event category, event action)|
| num_event_cats | Number of event types (categories) identified in sequence  |
| Event_cats_agg | List of tuples each containing (event category, its frequency)  |
| Event_cat_act_agg | List of nested tuples each containing ((event category, event action), its frequency)  |
| Page_List_NL | Page list without self-loops  |
| Page_Seq_NL | Page Seqence without self-loops  |
| Occurrences_NL | Number of sequence occurrences without self-loops  |

# Converting processed_journey data to functional network data

This creates two compressed csvs, one containing edges (and their weights = occurrences) and the other nodes.
These can be converted into many graph file formats or be read into graph processing software directly.

- Run `python src/data/make_network_data.py -h` to list required positional arguments:  
  - __source_directory__ - Source directory for input dataframe file(s).  
  - __input_filename__ - Source filename for input dataframe file(s).
  - __dest_directory__ - Specialized destination directory for output files.
  - __output_filename__ - Naming convention for resulting node and edge files.

- Other optional arguments:
  - __-q, --quiet__ -Turn off debugging logging.  

You need to create a destination directory for the node and edge files:  
`mkdir data/network_data`
  
Here's an example of a command execution:  
`python src/data/make_network_data.py processed_journey test_output network_data test`

where processed_journey is the directory containing output from make_dataset, test_output is 
test_output.csv.gz, network_data is the directory that the node and edge files will be exported to 
and test is the prefix for the node and edge filenames.

# Developing

```shell

git clone https://github.com/ukgovdatascience/govuk-network-data
cd govuk-network-data/

```

## Unit tests
pytest is used for unit testing. Install using pip: 

`pip install -U pytest`

Following installation navigate to the appropriate folder to run tests. For example to run tests on functions associated with the data extraction pipeline, go to `./src/data/` and run:  

`python -m pytest -v tests/`  
or
`python3 -m pytest -v tests/`  


## Contributing
See `CONTRIBUTING.md`  

## References

## License

