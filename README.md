##Installation
	1. Copy `credentials.py.template` as `credentials.py`
	2. `python setup.py install`  # Creates your database and installs packages
	3. `python data_db/create_db.py --initialize`  # Builds your database

##Database information
	Enter your database with `psql neural -h 127.0.0.1 -d neural`
	If error occur due to path issue use this `export PYTHONPATH=$PYTHONPATH:$(pwd)`

##Dependencies for Deconvolution
	Currently, the deconvolution pipeline requires support from the OASIS repository, (work cited as [Friedrich J, Zhou P, Paninski L. Fast Online Deconvolution of Calcium Imaging Data. PLoS Comput Biol. 2017; 13(3):e1005423]). To use this, please go to https://github.com/j-friedrich/OASIS and clone the repository in the main folder containing /abo_data. Do this by typing git clone https://github.com/j-friedrich/OASIS.git in your bash window and then setting it up by following the installation instructions in the repository.

TODO: How to create datasets.