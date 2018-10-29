# govuk-network-data
> Data pipeline for extraction and preprocessing of BigQuery user journey data.

A data pipeline for extracting and preprocessing BigQuery user journey data. The data captures the sequences of pages visited by users and how often these journies occur. Additional metadata and Event data (what users do on their journey) is also provided.  

# Installing / Getting started

A quick introduction of the minimal setup you need to get a hello world up &
running.

This should probably include setting up the computing environment and the necessary permissions required. Who to ask for them or point to a wiki page. 

```shell
packagemanager install awesome-project
awesome-project start
awesome-project "Do something!"  # prints "Nah."
```

Here you should say what actually happens when you execute the code above.

# Initial configuration

## Python version
Python 3.6.0

## Virtual environment

Source the environment variables from the .envrc file either using direnv (`direnv allow`) or `source .envrc`.  
You can check they're loaded using `echo $NAME_OF_ENVIRONMENT_VARIABLE` or `printenv`.

Then install required python packages:  

`pip install -r requirements.txt`

## Where to put your BigQuery key

`mkdir key`

then put the json file in there

# Features

This package arms data scientists with the tools to answer the hardest questions that people are asking about the sequence of pages that users are visiting and the type of behaviour those users are displaying.  

* A data pipeline that produces data in a convenient format to explore the GOV.UK page sequences or journies that users travel in a session.   
* Express this data as a graph with pages visited expressed as nodes and directed movement between pages as edges.   

## Configuration

Here you should write what are all of the configurations a user can enter when
using the project.

#### Argument 1
Type: `String`  
Default: `'default value'`

State what an argument does and how you can use it. If needed, you can provide
an example below.

Example:
```bash
awesome-project "Some other value"  # Prints "You're nailing this readme!"
```

#### Argument 2
Type: `Number|Boolean`  
Default: 100

Copy-paste as many of these as you need.


## Running a big query extract
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

## Creating a csv with each row a processed user journey with session information rolled into it 

For example is 3 users went A -> B -> C on different devices, then this would be a single row with a column containing a device dictionary where the device is the key and the value is the number of sessions using that device. E.g., `{'mobile':1, 'tablet':1, 'desktop':1}`.

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

