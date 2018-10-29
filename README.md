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

`python src/data/bq_extract_data.py 2018-10-18 2018-10-18 test_dir test prelim_meta_standard_query_with_pageseq --standard`  
In the above example, the SQL query exists as 'prelim_meta_standard_query_with_pageseq.sql' in the $QUERIES_DIR directory.

## Creating a csv with each row a user journey with sessions rolled into it

For example is 3 users went A -> B -> C on different devices, then this would be a single row with each device listed in that column

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

