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
Either source the environment variables from the .envrc file either using direnv (`direnv allow`) or `source .envrc` in the command line or add this EnvFile to the pycharm project run configurations, as described here: 
https://stackoverflow.com/questions/42708389/how-to-set-environment-variables-in-pycharm

You can check they're loaded using `echo $NAME_OF_ENVIRONMENT_VARIABLE` or `printenv`.  

You might notice that there are two (data directories) provided for your convenience:  
* __DATA_DIR__ which will point to your local data dir in this project.  
* __GDRIVE_DATADIR__ which points to our teams Google Drive data dir for this work. 
If you have access and the necessary software ([Google File Stream](https://support.google.com/a/answer/7491144?hl=en)) 
and permissions,
 then you can access the data from here rather than downloading it locally to your
  machine first.  
  
 Within these `data` dir are the following sub-dir:  
 -  __raw-bq-extract__ to hold the output from extract_data.py
- __processed_journey__ to hold the output from make_dataset.py
- __processed_network__ to hold the output from make_functional_network_data.py

## Using pip to install necessary packages
Then install required python packages:  

`pip install -r requirements.txt`  

We provide you with more than the minimal number of packages you need to run the data pipeline. We provide some 
convenience packages for reviewing notebooks etc.  

Alternatively, you can review the packages that are imported and manually install those that you think are necessary 
using `pip install` if you want more control over the process (and are a confident user). 

## BigQuery cost caveat

You are now ready to use this package to pipe data from BigQuery through a pandas dataframe and save your output locally as 
several compressed csv files (containing tabular data with tab-separation: tsv was necessary as the page urls can contain commas).
 Consider the cost of the query you intend to run and read all
community guidance beforehand.  

# What this does

This package equips data scientists with the tools to answer the hardest questions that people are asking about the sequence of pages that users are visiting and the type of behaviour those users are displaying.

* A data pipeline that produces data in a convenient format to explore the GOV.UK page sequences or journeys that 
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

`python src/data/bq_extract_data.py 2018-10-18 2018-10-18 raw_bq_extract raw_output_filename prelim_meta_standard_query_with_pageseq --standard`  

In the above example, the SQL query exists as `prelim_meta_standard_query_with_pageseq.sql` in the `$QUERIES_DIR` directory.

## Which query to use?

This depends on your question. `prelim_meta_standard_query_with_pageseq.sql` should be your default. It should cost about 
the same as `standard_query` but is slightly cleaner and has some additional meta data. You'll have to 
review the queries yourself to elucidate the precise differences. When you are more familiar you can write your own 
custom queries. Prior to using custom queries in the pipeline you should write them with the standard BigQuery 
Google compute tools so that you get an estimate of the cost of the query.  

## Managing expectations

The test code will take awhile to run (less than 10 mins). You should use `caffeinate` or prevent your machine
 from sleeping during this period.  
 
> Don't panic

While the code is running it will log its progress. This will appear in the terminal. 
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

  - __-doo, --drop_one_offs__ - Drop rare journeys occurring only once per input file. 
  If merging multiple inputs, the merge will occur before the count (so that single journeys occuring in each file are
   aggregated before counting). Then the default behaviour is to drop in batches, approximately every 3 days to fit in
    memory and compute constraints.  
  - __-kloo, --keep_len_one_only__ -Keep ONLY journeys with length 1 ie journeys visiting only one page.  
  - __-dlo, --drop_len_one__  Drop journeys with length 1 ie journeys visiting only one page.  
  - __-f FILENAME_STUB, --filename_stub FILENAME_STUB__ -If merging multiple inputs, filename_stub is the unique prefix in their filenames which identify this group of inputs.  
  - __-q, --quiet__ -Turn off debugging logging.  
  - __-h, --help__ show the help message and exit  
  
Here's an example of a command execution:  
`python src/data/make_dataset.py raw_bq_extract processed_journey processed_filename -doo`

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
| Taxon_List | List of taxons of pages |   

## Analysing this data

For help getting started analysing and visualising this type of data using R or Python,
 see our notebooks in `notebooks/eda`.
 
## Using processed journey data for A/B tests 
See the [govuk_ab_analysis](https://github.com/ukgovdatascience/govuk_ab_analysis) repo for some scripts to do this analysis.

`notebooks/eda/generate_ab_rl_mvp.ipynb` is a notebook to analyse an A/B test, using data generated by the `stnd_taxon_ab.sql` query.
`notebooks/eda/generating_ab_test_data_with_workings.ipynb` is a notebook with some more workings of how we derived our metrics.


# Analysing journey events

For a reproducible analytical pipeline approach, you can pass a processed_journey file to `journey_events_analysis.py`.
  It creates a dir with the name of said `processed_filename` in reports and then puts 2 csvs in there.  
 
`python src/analysis/journey_events_analysis.py processed_filename`

# Converting processed_journey data to functional network data

This creates two compressed csvs with tab-seperation (as some page urls have commas in), one containing edges (and their weights = occurrences) and the other nodes.
These can be converted into many graph file formats or be read into graph processing software directly.

- Run `python src/data/make_network_data.py -h` to list required positional arguments:  
  - __source_directory__ - Source directory for input dataframe file(s).  
  - __input_filename__ - Source filename for input dataframe file(s).
  - __dest_directory__ - Specialized destination directory for output files.
  - __output_filename__ - Naming convention for resulting node and edge files.

- Other optional arguments:
  - __-h, --help__      show this help message and exit
  - __-q, --quiet__     Turn off debugging logging.
  - __-d, --delooped__  Use delooped journeys for edge and weight computation
  - __-i, --incorrect__ Drop incorrect occurrences if necessary
  - __-t, --taxon__     Compute and include additional node attributes (only taxon
                    for now).
  

You need to create a destination directory for the node and edge files:  
`mkdir data/processed_network`
  
Here's an example of a command execution:  
`python src/data/make_network_data.py processed_journey processed_filename processed_network network_filename -d -i -t`

where processed_journey is the directory containing output from make_dataset, processed_filename is 
processed_filename.csv.gz, processed_network is the directory that the node and edge files will be exported to 
and network_filename is the prefix for the node and edge filenames.

## Using the network data

The two dataframes for nodes and edges represent the minimal way to produce a network in a [tidy fashion](https://www.data-imaginist.com/2017/introducing-tidygraph/). 

### Python

One option is to use python for the exploration of the data. You can read each csv.gz in with:

```python
import pandas as pd

nodes = pd.read_csv('../data/processed_network/network_filename_nodes.csv.gz', sep='\t')
```

This is explored further in some of the `notebooks/network_analysis`, where we use provide a tutorial with `networkx`.  

### Docker and Neo4j

Given the size of the data (if over more than a few days) you might consider building a graph database to speed up 
your analysis (the nodes and edges csv format is also amenable to standard network science tools).  

Install Docker on your machine using software management centre (if new to Docker, we suggest you do the tutorial). 

From the terminal run:  

```bash

docker run \
    --publish=7474:7474 --publish=7687:7687 \
    --volume=$HOME/neo4j/data:/data \
    --volume=$HOME/neo4j/logs:/logs \
    --volume=$HOME/neo4j/import:/import \
    --env=NEO4J_AUTH=none \
    neo4j
    
    # for docker neo4j setup see
    # https://neo4j.com/developer/docker-23/
```

Open the local host `http://localhost:7474/browser/` in your browser after the instance has started up. You'll see Neo4j 
running locally.  

### Move your nodes and edges csv
Notice how one of the arguments creates an import folder in the newly created `neo4j` dir. This 
is in your `$HOME` dir. We need to move the .csv we wish to load into neo4j into the  aforementioned 
`/import` dir. Copy them across.  

### Restart neo4j

Stop the neo4j instance using Ctrl + C in the terminal where it is running. Restart it using the above code chunk.  

### Load the network into neo4j

Open the local host `http://localhost:7474/browser/` in your browser after the instance has started up. You'll see Neo4j 
running locally. There's a prompt where you can enter Cypher commands (the Neo4j language). Run the following code 
to load in your data, adjusting for filename differences and different header names.  

#### Clear any nodes or edges

Ensure a clean graph database by clearing any old nodes or edges stored.  

```bash

MATCH (n)
DETACH DELETE n

```

We can now be confident loading our data in. However, due to changes to the output files being 
tab-separated files rather than comma-separated (as page urls had commas), the below code needs some modifications. 
Specifically we need to a Cypher command to acknowledge the tsv-ness,
 see [here for help](http://bigdatums.net/2016/12/17/load-tab-delimited-file-neo4j/). This fix has not been tested yet.   

#### Nodes

Here our csv has the header "url".  

```bash
// Create nodes for each unique page
USING PERIODIC COMMIT
LOAD CSV WITH HEADERS FROM "file:///test_nodes.csv" AS row
CREATE (:Page {url:row.url});

```

This should populate the graph database with nodes (which are page urls in our case). We then index our nodes to speed things up 
when creating relationships.  

```bash

CREATE INDEX ON :Page(url);

```

#### Edges

Here our csv for edges has the headers; "source", "destination" (both page urls) and "weight" (occurrences). This will 
take a few seconds to run for one days data.

```bash

USING PERIODIC COMMIT
LOAD CSV WITH HEADERS FROM "file:///test_edges.csv" AS row
MATCH (source:Page {url: row.source})
MATCH (destination:Page {url: row.destination})
MERGE (source)-[:MOVES_TO {occurrences:row.weight}]->(destination);

```

You can check the correct graph database has been instantiated by calling the schema, or viewing some nodes and edges.  

```bash

CALL db.schema();
```

or 

```bash
MATCH (n) RETURN n LIMIT 500
```

This should look like a bunch of nodes and edges with direction. You can add weights to the edges and colour the nodes by 
Page metadata type if so inclined.  

There's plenty of software available to manage this type of data, don't feel constrained to use Neo4j
 
### Nuanced queries

Neo4j is very fast. You can run nuanced queries quickly. For example:   

```

MATCH (n)
WHERE n.url STARTS WITH '/government/organisations/department-for-education'
RETURN n
LIMIT 500;

```

Consult the [Neo4j](https://neo4j.com/docs/developer-manual/current/cypher/clauses/) manual for further guidance.

### Visualising the network

People like visualisations, use Gephi or any of the plenty of suitable tools for doing this. 
See the `notebooks/network_analysis/networkx_tutorial_govuk` for some code to do this.    


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
pytest is used for unit testing. If you haven't installed the requirements, then install using pip: 

`pip install -U pytest`

Following installation navigate to the appropriate folder to run tests. For example to run tests on functions associated with the data extraction pipeline, go to `./src/data/` and run:  

`python -m pytest -v tests/`  
or
`python3 -m pytest -v tests/`  

### Testing the pipeline

Some functions change panda dataframes with specific column names.

# Contributing
See `CONTRIBUTING.md`  

## License
See LICENSE
