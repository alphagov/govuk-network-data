# Python setup for MacOS

This is a quick run through on how to set up Python on your machines. We'll be
using `pip` to install our packages, and `pyenv` with its `pyenv-virtualenv`
plugin to manage different Python versions and virtual environments,
respectively.

Python virtual environments allow you to create an isolated environment. This
can have its own dependencies (different packages, different versions)
completely separate from every other environment.

These instructions have been adapted from [The Hitchhiker's Guide to Python](https://docs.python-guide.org/starting/install3/osx/).
Further detail about `pyenv-virtualenv` can be found in its [documentation](https://github.com/pyenv/pyenv-virtualenv#pyenv-virtualenv).

By default, MacOS has Python 2 installed, but we need Python 3.

Install [Homebrew](https://brew.sh/) using Terminal.
```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

Install the latest version of Python 3 using Homebrew; this should also install
`pip` for you automatically.
```
brew install python
```

Add your newly-installed Python to PATH, and validate your Python 3 version has
been installed.
```
echo 'export PATH="/usr/local/opt/python/libexec/bin:$PATH"' >> ~/.bash_profile
python --version  # as of Oct 2019, this should be Python 3.7.4 on Homebrew
```

Use Homebrew to install `pyenv`, and its `pyenv-virtualenv` plugin, and add required
lines to your `.bash_profile`.
```
brew install pyenv
brew install pyenv-virtual

echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bash_profile
```

Create a new Python virtual environment running Python 3.6.9; we'll call this
virtual environment `govuk-network-data`.
```
pyenv virtualenv 3.6.9 govuk-network-data
```

You need to activate this virtual environment before install packages and using
it.
```
pyenv activate govuk-network-data
```

Now install packages as listed in the `requirements.txt` file in this
repository.
```
pip install -r <<<PATH TO requirements.txt>>>
```

To deactivate the virtual environment run the following code:
```
pyenv deactivate
```
