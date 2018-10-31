# govuk-network-data
> Making better use of BigQuery user journey data.

A data pipeline for extracting and preprocessing BigQuery user journey data. The data captures the sequences of pages
 visited by users and how often these journeys occur. Additional metadata and Event data (what users do on their journey)
  is also provided.  

# Installing / Getting started
When following this guidance, code is executed in the terminal unless specified otherwise.  

You need permissions and a key to GOV.UK BigQuery analytics (govuk-bigguery-analytics) to extract raw data for this 
pipeline. People to ask about this are senior performance analysts or search the GDS wiki for guidance.

Clone this repo using:  
 
`git clone git@github.com:ukgovdatascience/govuk-network-data.git`  

in your terminal.  

## Where to put your BigQuery key
After cloning the repo, navigate to it using:  

`cd govuk-network-data`  

Next, create a directory to hold your private key.

`mkdir key`

then place the private key (.json) in this folder. There should only be one key.  

## Python version
You will need the python interpreter version [Python 3.6.0](https://www.python.org/downloads/release/python-360/).  

## Virtual environment
Create a new python 3.6.0 virtual environment using your favourite virtual environment 
manager (you may need `pyenv` to specify python version; which you can get using `pip install pyenv`). 

If new to python, an easy way to do this is using the PyCharm community edition and opening this repo as a project. 
You can then specify what python interpreter to use (as 
explained [here](https://stackoverflow.com/questions/41129504/pycharm-with-pyenv)).  

## Setting Environment variables
Source the environment variables from the `.envrc` file either using direnv (`direnv allow`) or `source .envrc`.  
You can check they're loaded using `echo $NAME_OF_ENVIRONMENT_VARIABLE` or `printenv`.  

## Using pip to install necessary packages
Then install required python packages:  

`pip install -r requirements.txt`  

We provide you with more than the minimal number of packages you need to run the data pipeline. We provide some 
convenience packages for reviewing notebooks etc.  

Alternatively, you can review the packages that are imported and manually install those that you think are necessary 
using `pip install` if you want more control over the process. 

## BigQuery cost caveat

You are now ready to use this package to pipe data from BigQuery through a pandas dataframe and 
output as a bunch of compressed csv. Consider the cost of the query you intend to run and read all
community guidance beforehand.  

# What this does

This package arms data scientists with the tools to answer the hardest questions that people are asking about thei
 sequence of pages that users are visiting and the type of behaviour those users are displaying.  

* A data pipeline that produces data in a convenient format to explore the GOV.UK page sequences or journies that 
users travel in a session.   
* Express this data as a graph with pages visited expressed as nodes and directed movement between pages as edges.   

![alt text](network_data_pipeline.png)



# Extracting raw data from big query
This produces a compressed csv in the destination directory (raw_bq_extract) where each row is a specific user journey
 (including events). However this raw data is messy and needs preprocessing to be analytically useful
  (see next section: 'Converting raw big query data to processed_journey data').

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

First, save your sql query 'query_name.sql' in the `$QUERIES_DIR` directory.  

Here's an example of a command execution (please consider your query carefully, as this is not free!): 

`python src/data/bq_extract_data.py 2018-10-18 2018-10-18 raw_output_dir test prelim_meta_standard_query_with_pageseq --standard`  

In the above example, the SQL query exists as `prelim_meta_standard_query_with_pageseq.sql` in the `$QUERIES_DIR` directory.

## Managing expectations

The test code will take awhile to run (less than 10 mins). You should use `caffeinate` or prevent your machine
 from sleeping during this period.  
 
> Don't panic

While the code is running it will log it's progress. This will appear in the terminal. 
Remember that the 200 code in the Debug level logging tells us that the request was
 received and understood and is being processed.  
 
 Upon completion you should be notified as to the number of rows, time to run and where the output 
 was saved.

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

git pull
git checkout -b feature/something-awesome
# make changes
git push origin feature/something-awesome
# create pull request with branch on Github
# request a review

```
Develop as a separate branch and push to Github. Create a pull request and ensure all unit tests pass. 
Create new units tests for any extra functions.  
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
See LICENSE
