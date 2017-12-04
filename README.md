##Installation
	1. Copy `credentials.py.template` as `credentials.py`
	2. `python setup.py install`  # Creates your database and installs packages
	3. `python data_db/create_db.py --initialize`  # Builds your database

##Database information
	Enter your database with `psql neural -h 127.0.0.1 -d neural`
	If error occur due to path issue use this `export PYTHONPATH=$PYTHONPATH:$(pwd)`


TODO @pachaya: Fill this out and provide a How-To for creating datasets.
