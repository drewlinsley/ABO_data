#!/usr/bin/env python
import sshtunnel
import argparse
import psycopg2
import psycopg2.extras
import psycopg2.extensions
import credentials
from allen_config import Allen_Brain_Observatory_Config
sshtunnel.DAEMON = True  # Prevent hanging process due to forward thread
main_config = Allen_Brain_Observatory_Config()


class data_db(object):
    def __init__(self, config):
        self.status_message = False
        self.db_schema_file = 'db/db_schema.txt'
        # Pass config -> this class
        for k, v in config.items():
            setattr(self, k, v)

    def __enter__(self):
        if main_config.db_ssh_forward:
            forward = sshtunnel.SSHTunnelForwarder(
                credentials.machine_credentials()['ssh_address'],
                ssh_username=credentials.machine_credentials()['username'],
                ssh_password=credentials.machine_credentials()['password'],
                remote_bind_address=('127.0.0.1', 5432))
            forward.start()
            self.forward = forward
            self.pgsql_port = forward.local_bind_port
        else:
            self.forward = None
            self.pgsql_port = ''
        pgsql_string = credentials.postgresql_connection(str(self.pgsql_port))
        self.pgsql_string = pgsql_string
        self.conn = psycopg2.connect(**pgsql_string)
        self.conn.set_isolation_level(
            psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        self.cur = self.conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print exc_type, exc_value, traceback
            self.close_db(commit=False)
        else:
            self.close_db()
        if main_config.db_ssh_forward:
            self.forward.close()
        return self

    def close_db(self, commit=True):
        self.conn.commit()
        self.cur.close()
        self.conn.close()

    def return_status(
            self,
            label,
            throw_error=False):
        """
        General error handling and status of operations.
        ::
        label: a string of the SQL operation (e.g. 'INSERT').
        throw_error: if you'd like to terminate execution if an error.
        """
        if label in self.cur.statusmessage:
            print 'Successful %s.' % label
        else:
            if throw_error:
                raise RuntimeError('%s' % self.cur.statusmessag)
            else:
                'Encountered error during %s: %s.' % (
                    label, self.cur.statusmessage
                    )

    def recreate_db(self, run=False):
        """Initialize the DB with db_schema_file."""
        if run:
            db_schema = open(self.db_schema_file).read().splitlines()
            for s in db_schema:
                t = s.strip()
                if len(t):
                    self.cur.execute(t)

    def populate_db_with_cell_stim(self, namedict):
        """
        Add cell stim info to the db.
        ::
        experiment_name: name of experiment to add
        parent_experiment: linking a child (e.g. clickme) -> parent (ILSVRC12)
        """
        self.cur.executemany(
            """
            INSERT INTO cells
            (
            cell_specimen_id,
            session,
            drifting_gratings,
            locally_sparse_noise,
            locally_sparse_noise_four_deg,
            locally_sparse_noise_eight_deg,
            natural_movie_one,
            natural_movie_two,
            natural_movie_three,
            natural_scenes,
            spontaneous,
            static_gratings,
            cell_output_npy
            )
            VALUES
            (
            %(cell_specimen_id)s,
            %(session)s,
            %(drifting_gratings)s,
            %(locally_sparse_noise)s,
            %(locally_sparse_noise_four_deg)s,
            %(locally_sparse_noise_eight_deg)s,
            %(natural_movie_one)s,
            %(natural_movie_two)s,
            %(natural_movie_three)s,
            %(natural_scenes)s,
            %(spontaneous)s,
            %(static_gratings)s,
            %(cell_output_npy)s
            )
            """,
            namedict)
        if self.status_message:
            self.return_status('INSERT')

    def populate_db_with_rf(self, namedict):
        """
        Add cell RF info to the db.
        ::
        experiment_name: name of experiment to add
        parent_experiment: linking a child (e.g. clickme) -> parent (ILSVRC12)
        """
        self.cur.executemany(
            """
            INSERT INTO rf
            (
            cell_specimen_id,
            lsn_name,
            experiment_container_id,
            found_on,
            found_off,
            alpha,
            number_of_shuffles,
            on_distance,
            on_area,
            on_overlap,
            on_height,
            on_center_x,
            on_center_y,
            on_width_x,
            on_width_y,
            on_rotation,
            off_distance,
            off_area,
            off_overlap,
            off_height,
            off_center_x,
            off_center_y,
            off_width_x,
            off_width_y,
            off_rotation,
            cre_line,
            structure,
            age,
            imaging_depth
            )
            VALUES
            (
            %(cell_specimen_id)s,
            %(lsn_name)s,
            %(experiment_container_id)s,
            %(found_on)s,
            %(found_off)s,
            %(alpha)s,
            %(number_of_shuffles)s,
            %(on_distance)s,
            %(on_area)s,
            %(on_overlap)s,
            %(on_height)s,
            %(on_center_x)s,
            %(on_center_y)s,
            %(on_width_x)s,
            %(on_width_y)s,
            %(on_rotation)s,
            %(off_distance)s,
            %(off_area)s,
            %(off_overlap)s,
            %(off_height)s,
            %(off_center_x)s,
            %(off_center_y)s,
            %(off_width_x)s,
            %(off_width_y)s,
            %(off_rotation)s,
            %(cre_line)s,
            %(structure)s,
            %(age)s,
            %(imaging_depth)s
            )
            """,
            namedict)
        if self.status_message:
            self.return_status('INSERT')

    def select_cells_by_rf_coor(self, namedict):
        """
        Select cells by rf coordinates.
        """
        self.cur.execute(
            """
            SELECT * FROM rf
            WHERE
            on_center_x >= %s and
            on_center_x < %s and
            on_center_y >= %s and
            on_center_y < %s
            """
            %
            (
                namedict['x_min'],
                namedict['x_max'],
                namedict['y_min'],
                namedict['y_max'])
            )
        if self.status_message:
            self.return_status('INSERT')
        return self.cur.fetchall()

    def gather_data_by_rf_coor(self, namedict):
        """
        Select cells by rf coordinates.
        """
        eq = ''
        if 'cre_line' in namedict:
            eq += " and lower(cre_line) LIKE '%%%s%%'"  % namedict['cre_line'].lower()
        if 'structure' in namedict:
            eq += " and lower(structure)='%s'" % namedict['structure'].lower()
        if 'imaging_depth' in namedict:
            eq += " and imaging_depth=%s" % namedict['imaging_depth']

        self.cur.execute(
            """
            SELECT DISTINCT ON (cells.cell_specimen_id) * FROM rf
            INNER JOIN cells on cells.cell_specimen_id=rf.cell_specimen_id
            WHERE
            on_center_x >= %s and
            on_center_x < %s and
            on_center_y >= %s and
            on_center_y < %s
            %s
            """
            %
            (
                namedict['rf_coordinate_range']['x_min'],
                namedict['rf_coordinate_range']['x_max'],
                namedict['rf_coordinate_range']['y_min'],
                namedict['rf_coordinate_range']['y_max'],
                eq)
            )
        if self.status_message:
            self.return_status('INSERT')
        return self.cur.fetchall()

    def proc_stim_query(
            self,
            stim_string,
            stim_query,
            logical=' or ',
            column=None,
            sub_query=None):
        if len(stim_string) > 0:
            stim_string += logical
        if column is not None:
            stim_query = '%s=\'' % column + stim_query + '\''
        if sub_query is not None:
            stim_query = '(%s and (%s))' % (stim_query, sub_query)
        stim_string += stim_query
        return stim_string

    def gather_data_by_rf_coor_and_stim(
            self,
            namedict,
            stimuli_filter=None,
            session_filter=None):
        """
        Select cells by rf coordinates.
        """
        eq = ''
        if 'cre_line' in namedict:
            eq += " and lower(cre_line) LIKE '%%%s%%'"  % namedict['cre_line'].lower()
        if 'structure' in namedict: 
            eq += " and lower(structure)='%s'" % namedict['structure'].lower()
        if 'imaging_depth' in namedict:
            eq += " and imaging_depth=%s" % namedict['imaging_depth']

        # Create stimulus filter
        stim_string = ''
        if stimuli_filter is not None:
            for stim_query in stimuli_filter:
                stim_string = self.proc_stim_query(
                    stim_string,
                    stim_query)
            if len(stim_string) > 0 and session_filter is None:
                stim_string = ' and ' + stim_string
            print 'Querying stimuli by: %s.' % stim_string

        # Create session filter
        if session_filter is not None:
            session_string = ''
            for session_query in session_filter:
                session_string = self.proc_stim_query(
                    session_string,
                    session_query,
                    sub_query=stim_string,
                    column='session')
            if len(session_string) > 0:
                session_string = ' and (' + session_string
                print 'Querying session by: %s.' % session_string
                stim_string = session_string
                stim_string += ')'
        else:
            stim_string = ' and (%s)' % stim_string.split('and ')[-1]

        # Query DB -- Note the distinct flag on inner join!
        self.cur.execute(
            """
            SELECT * FROM rf
            INNER JOIN cells on cells.cell_specimen_id=rf.cell_specimen_id
            WHERE
            on_center_x >= %s and
            on_center_x < %s and
            on_center_y >= %s and
            on_center_y < %s
            %s
            %s
            """
            %
            (
                namedict['rf_coordinate_range']['x_min'],
                namedict['rf_coordinate_range']['x_max'],
                namedict['rf_coordinate_range']['y_min'],
                namedict['rf_coordinate_range']['y_max'],
                eq,
                stim_string)
            )
        if self.status_message:
            self.return_status('INSERT')
        return self.cur.fetchall()


def initialize_database():
    """Initialize the psql database from the schema file."""
    config = credentials.postgresql_connection()
    with data_db(config) as db_conn:
        db_conn.recreate_db(run=True)
        db_conn.return_status('CREATE')


def get_cells_by_rf(list_of_dicts):
    """Query cells by their RF centers."""
    config = credentials.postgresql_connection()
    queries = []
    with data_db(config) as db_conn:
        for d in list_of_dicts:
            queries += [db_conn.select_cells_by_rf_coor(d)]
    return queries


def get_cells_all_data_by_rf(list_of_dicts):
    """Get all data for cells by their RF centers."""
    config = credentials.postgresql_connection()
    queries = []
    with data_db(config) as db_conn:
        for d in list_of_dicts:
            queries += [db_conn.gather_data_by_rf_coor(d)]
    return queries


def get_cells_all_data_by_rf_and_stimuli(rfs, stimuli, sessions=None):
    """Get all data for cells by their RF centers."""
    config = credentials.postgresql_connection()
    queries = []
    with data_db(config) as db_conn:
        for it_rf in rfs:
            queries += [
                db_conn.gather_data_by_rf_coor_and_stim(
                    it_rf,
                    stimuli,
                    sessions)
            ]
    return queries


def add_cell_data(
        cell_rf_dict,
        list_of_cell_stim_dicts):
    """Add a cell to the databse.

    Inputs:::
    cell_rf_dict: dictionary containing cell_id_number and its RF properties.
    list_of_cell_stim_dicts: a list of dictionaries, each containing the cell's
        id + a pointer to a data numpy file and a boolean for the stimuli it
        contains.
    ------------------------------
    For a given cell, e.g., cell_1

    cell_rf_dict = {
        'cell_id': 1,
        'rf': big
    }
    list_of_cell_stim_dicts[
        {
            'cell_id': 1,
            'session': A,
            'drifting_gratings': True,
            'ALL OTHER COLUMNS': False,
            'cell_npy': os.path.join(data_dir, '%s_%s_%s.npy' % (
                cell_id, session, stimulus))
        },
        {
            'cell_id': 1,
            'session': B,
            'drifting_gratings': True,
            'ALL OTHER COLUMNS': False,
            'cell_npy': os.path.join(
                data_dir, '%s_%s_%s.npy' % (cell_id, session, stimulus))
        }
    ]
    """
    config = credentials.postgresql_connection()
    with data_db(config) as db_conn:
        db_conn.populate_db_with_rf([cell_rf_dict])
        db_conn.populate_db_with_cell_stim(list_of_cell_stim_dicts)


def get_performance(experiment_name):
    config = credentials.postgresql_connection()
    with data_db(config) as db_conn:
        perf = db_conn.get_performance(experiment_name=experiment_name)
    return perf


def main(
        initialize_db):
    if initialize_db:
        print 'Initializing database.'
        initialize_database()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initialize",
        dest="initialize_db",
        action='store_true',
        help='Recreate your database.')
    args = parser.parse_args()
    main(**vars(args))
