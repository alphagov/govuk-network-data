# govuk-network-data
> Data pipeline for extraction and preprocessing of BigQuery user journey data.

A data pipeline for extracting and preprocessing BigQuery user journey data. The data captures the sequences of pages visited by users and how often these journies occur. Additional metadata and Event data (what users do on their journey) is also provided.  

# Installing / Getting started

A quick introduction of the minimal setup you need to get a hello world up &
running.

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

`pip install -r requirements.txt`

## Where to put your BigQuery key

`mkdir key`

then put the json file in there

# Features

## Running a big query extract

`python src/data/bq_extract_data.py '2018-10-15' '2018-10-15' 'test_dir' 'test' 'simple_test.sql'`

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

## Features

What's all the bells and whistles this project can perform?
* What's the main functionality
* You can also do another thing
* If you get really randy, you can even do this

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

## Contributing
See `CONTRIBUTING.md`  

## References

## License
The code in this project is licensed under MIT license.
