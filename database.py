import time
import psutil
import logging
import itertools
import pandas as pd
import psycopg2 as pg2
import connectorx as cx
import multiprocessing as mp
from sqlalchemy.sql import text
from sqlalchemy import event, DDL
from database_config import settings
from cbpro_tickers import cbpro_tickers
from sqlalchemy_views import CreateView
from HistoricalData import HistoricalData
from sqlalchemy.exc import ProgrammingError
from psycopg2.errors import InvalidSchemaName
from sqlalchemy import create_engine, Table, MetaData
from sqlalchemy.schema import CreateSchema, DropSchema
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy_utils import create_database, database_exists, drop_database


class Database(object):
    SQL_ALCHEMY_DATABASE_URL = f'postgresql+psycopg2://{settings.database_username}:{settings.database_password}@{settings.database_hostname}:{settings.database_port}/{settings.database_name}'
    CONNECTORX_DATABASE_URL = f'postgresql://{settings.database_read_only_username}:{settings.database_read_only_password}@{settings.database_hostname}:{settings.database_port}/{settings.database_name}'

    def __init__(self, logger: logging.Logger, debug=False):
        self.engine = create_engine(self.SQL_ALCHEMY_DATABASE_URL)

        if debug:
            self.drop_database(logger=logger)

            self.create_database(logger=logger)
            self.grant_schema_permission(logger=logger, schema_name='public')
            self.create_schema(logger=logger)

            try:
                connection, cursor = self.connect()
                cursor.execute(f"DROP SCHEMA IF EXISTS public CASCADE")
                self.disconnect(connection=connection, cursor=cursor)
                if self.schema_exists(logger=logger, schema_name='public'):
                    logger.error('public schema not dropped')
                else:
                    logger.info('public schema dropped successfully')
            except ProgrammingError as e:
                logger.error(f"Failed to drop public schema: {e}")

    def __repr__(self):
        return 'CryptoJutsu Database'

    def grant_schema_permission(self, logger, schema_name):
        try:
            connection, cursor = self.connect()
            cursor.execute(f'GRANT CREATE ON DATABASE "{settings.database_name}" TO {settings.database_username};')
            cursor.execute(f'GRANT USAGE ON SCHEMA "{schema_name}" TO {settings.database_username};')
            cursor.execute(f'ALTER SCHEMA "{schema_name}" OWNER TO {settings.database_username};')
            logger.info(f"Granted temporary permissions to {settings.database_username} to drop schema {schema_name}.")
            self.disconnect(connection=connection, cursor=cursor)
        except Exception as e:
            logger.error(f"Failed to grant permissions: {e}")
            raise

    def revoke_drop_schema_permission(self, logger):
        try:
            connection, cursor = self.connect()
            cursor.execute(f"REVOKE CREATE ON DATABASE {settings.database_name} FROM {settings.database_username};")
            cursor.execute(f"REVOKE USAGE ON SCHEMA public FROM {settings.database_username};")
            cursor.execute(f"REVOKE DROP ON SCHEMA public FROM {settings.database_username};")
            logger.info(f"Revoked temporary permissions from {settings.database_username}.")
            self.disconnect(connection=connection, cursor=cursor)
        except Exception as e:
            logger.error(f"Failed to revoke permissions: {e}")

    @staticmethod
    def connect():
        connection = pg2.connect(database=settings.database_name,
                                 user=settings.database_username,
                                 password=settings.database_password,
                                 host=settings.database_hostname,
                                 port=settings.database_port)

        cursor = connection.cursor()
        return connection, cursor

    @staticmethod
    def disconnect(connection, cursor):
        connection.commit()
        cursor.close()
        connection.close()

    @staticmethod
    def speed_read(query):
        return cx.read_sql(conn=Database.CONNECTORX_DATABASE_URL, query=query)

    def create_database(self, logger: logging.Logger):
        engine = create_engine(self.SQL_ALCHEMY_DATABASE_URL)
        if not database_exists(engine.url):
            create_database(engine.url)
            logger.info('CryptoJutsu Database Created')

    def drop_database(self, logger: logging.Logger):
        engine = create_engine(self.SQL_ALCHEMY_DATABASE_URL)
        if database_exists(engine.url):
            drop_database(engine.url)
            logger.info('DEBUG MODE')

    def schema_exists(self, logger: logging.Logger, schema_name: str):
        schema_investigation = f'SELECT * from information_schema.schemata where schema_name = \'{schema_name}\';'
        schema_exists = False
        connection, cursor = self.connect()

        try:
            cursor.execute(schema_investigation)
            results = cursor.fetchall()
        except BaseException as e:
            logger.info(
                f'Database query failed. There may be no schema with the name {schema_name} - {e}')
            results = [[0]]

        if not results:
            results = [[0]]

        self.disconnect(connection=connection, cursor=cursor)

        try:
            if results[0][0]:
                schema_exists = True
        except BaseException as e:
            logger.info(e)
            schema_exists = False

        return schema_exists

    def create_schema(self, logger: logging.Logger):
        if not self.schema_exists(logger=logger, schema_name=settings.candlesticks_schema):
            connection, cursor = self.connect()
            cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{settings.candlesticks_schema}"')
            self.disconnect(connection=connection, cursor=cursor)
            self.grant_schema_permission(logger=logger, schema_name=settings.candlesticks_schema)
            if self.schema_exists(logger=logger, schema_name=settings.candlesticks_schema):
                logger.info(f'"{settings.candlesticks_schema}" schema created')
            else:
                raise ValueError(f'{settings.candlesticks_schema} Schema not created')

    def drop_schema(self, logger: logging.Logger):
        if self.schema_exists(logger=logger, schema_name=settings.candlesticks_schema):
            schema_investigation = f'SELECT table_name from information_schema.tables where table_schema = \'{settings.candlesticks_schema}\';'
            cnx = pg2.connect(user=settings.database_username, password=settings.database_password,
                              host=settings.database_hostname, port=settings.database_port,
                              database=settings.database_name)
            cursor = cnx.cursor()
            cursor.execute(schema_investigation)
            results = cursor.fetchall()
            if len(results):
                tables = results
            else:
                tables = []
            cursor.close()
            cnx.close()

            for table in tables:
                self.drop_table(table_name=table[0], logger=logger)

            try:
                self.engine.execute(DropSchema(settings.candlesticks_schema))
            except ProgrammingError as e:
                logger.info(f'Failed to drop schema: {settings.candlesticks_schema} - {e}')
            except InvalidSchemaName:
                pass

    def table_exists(self, table_name: str):
        validation_query = f'SELECT table_name FROM information_schema.tables WHERE table_schema = \'{settings.candlesticks_schema}\' AND table_name = \'{table_name}\' ORDER BY table_schema DESC'
        table_exists = False
        connection, cursor = self.connect()

        try:
            cursor.execute(validation_query)
            results = cursor.fetchall()
        except:
            results = [[0]]

        if not results:
            results = [[0]]

        self.disconnect(connection=connection, cursor=cursor)

        try:
            if results[0][0]:
                table_exists = True
        except BaseException as e:
            print(e)
            table_exists = False

        return table_exists

    def view_exists(self, view_name: str):
        validation_query = f'SELECT table_name FROM information_schema.tables WHERE table_schema = \'{settings.candlesticks_schema}\' AND table_name = \'{view_name}_missing_minutes\' ORDER BY table_schema DESC'
        table_exists = False
        connection, cursor = self.connect()

        try:
            cursor.execute(validation_query)
            results = cursor.fetchall()
        except:
            results = [[0]]

        if not results:
            results = [[0]]

        self.disconnect(connection=connection, cursor=cursor)

        try:
            if results[0][0]:
                table_exists = True
        except BaseException as e:
            print(e)
            table_exists = False

        return table_exists

    def drop_table(self, table_name, logger: logging.Logger):
        if self.table_exists(table_name=table_name):
            drop_query = f'DROP TABLE "{settings.candlesticks_schema}"."{table_name}";'
            connection, cursor = self.connect()

            try:
                cursor.execute(drop_query)
                self.disconnect(connection=connection, cursor=cursor)
                logger.info(f'Table: {table_name} successfully dropped')
            except BaseException as e:
                logger.info(f'Table: {table_name} could not be dropped - {e}')

    def drop_view(self, view_name, logger: logging.Logger):
        if self.view_exists(view_name=view_name):
            drop_query = f'DROP VIEW "{settings.candlesticks_schema}"."{view_name}_missing_minutes";'
            connection, cursor = self.connect()

            try:
                cursor.execute(drop_query)
                self.disconnect(connection=connection, cursor=cursor)
                logger.info(f'View: {view_name}_missing_minutes successfully dropped')
            except BaseException as e:
                logger.info(f'View: {view_name}_missing_minutes could not be dropped - {e}')

    def create_table(self, ticker: str, logger: logging.Logger):
        if not self.table_exists(table_name=ticker):
            connection, cursor = self.connect()
            query = f'CREATE TABLE "{settings.candlesticks_schema}"."{ticker}" ("Time_Stamp" timestamp without time zone NOT NULL, "Date" date NOT NULL, "Time" time without time zone NOT NULL, "Open" double precision NOT NULL, "High" double precision NOT NULL, "Low" double precision NOT NULL, "Close" double precision NOT NULL, "Volume" double precision NOT NULL, PRIMARY KEY ("Time_Stamp")); ALTER TABLE IF EXISTS "{settings.candlesticks_schema}"."{ticker}" OWNER to {settings.database_username}'
            cursor.execute(query=query)
            self.disconnect(connection=connection, cursor=cursor)
            logger.info(f'Table "{settings.candlesticks_schema}"."{ticker}" successfully created')

    def build_candlesticks(self, start_date: str, end_date: str, logger: logging.Logger, cores: int = mp.cpu_count()):
        # check if data being requested already exists in order to avoid reloading data
        tickers = [i.upper() for i in cbpro_tickers]
        for ticker in tickers:
            self.create_table(ticker=ticker, logger=logger)

        ticker_dict = cbpro_to_sql(tickers=tickers, start_date=start_date, end_date=end_date, cores=cores, logger=logger)
        return ticker_dict


def cbpro_get(ticker: str, day: str, shared_dict: dict, logger: logging.Logger):
    shared_dict['cbpro'] = True
    try:

        try:
            data = HistoricalData(
                ticker=ticker,
                granularity=60,
                start_date=(pd.Timestamp(day) - pd.Timedelta("1 Days")).strftime("%Y-%m-%d-%H-%M"),
                end_date=day,
                verbose=False
            ).retrieve_data()

        except BaseException as e:
            print(f'Error retrieving data from Coinbase Pro - {e}')
            time.sleep(60)
            data = cbpro_get(ticker=ticker, day=day, shared_dict=shared_dict, logger=logger)

        shared_dict['cbpro'] = False
        return data
    except BaseException as e:
        raise ValueError(f'Financial Modeling Prep does not have {ticker} data from {day} - {e}')


def get_day(ticker: str, day: str, logger: logging.Logger):
    try:
        query = f"SELECT * FROM \"{settings.candlesticks_schema}\".\"{ticker}\" WHERE \"Date\" = CAST('{day}' AS date)"
        result = cx.read_sql(
            f'postgresql://{settings.database_username}:{settings.database_password}@{settings.database_hostname}:{settings.database_port}/{settings.database_name}',
            query)
        return result
    except BaseException as e:
        raise ValueError(f'{mp.current_process().name}: {ticker} Select Query Failed - {e}')


def insert_new_data(ticker, data, logger: logging.Logger):
    if 'Time_Stamp' == [*data.index.names][0]:
        data.reset_index(inplace=True)

    engine = create_engine(
        f'postgresql+psycopg2://{settings.database_username}:{settings.database_password}@{settings.database_hostname}:{settings.database_port}/{settings.database_name}'
    )

    data = data.drop_duplicates(subset=['Time_Stamp'], keep='last')

    # Check if 'Time_Stamp' is already the index, if not, set it as the index
    if 'Time_Stamp' != [*data.index.names][0]:
        data.set_index('Time_Stamp', inplace=True)

    if 'index' in [*data.columns]:
        data.drop('index', axis=1, inplace=True)

    data.to_sql(con=engine, schema=settings.candlesticks_schema, name=ticker, if_exists='append')
    return True


def fill_in_missing_rows(ticker: str, core: int, logger: logging.Logger):
    ticker_time = time.time()

    data = Database.speed_read(
        f'SELECT "Time_Stamp", "Date", "Time", "Open", "High", "Low", "Close", "Volume" FROM "Candlesticks"."{ticker}"'
    )

    df = data.copy()
    required_columns = ["Time_Stamp", "Date", "Time", "Open", "High", "Low", "Close", "Volume"]
    if not all(column in df.columns for column in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    # Convert 'Time_Stamp' to datetime if it is not already
    if not pd.api.types.is_datetime64_any_dtype(df['Time_Stamp']):
        df['Time_Stamp'] = pd.to_datetime(df['Time_Stamp'])

    # Set 'Time_Stamp' as the index
    df = df.set_index('Time_Stamp')

    # Create a complete time index from the start to the end of the DataFrame with a frequency of one minute
    complete_time_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='T')

    # Reindex the DataFrame to the complete time index, filling missing rows with NaN
    df = df.reindex(complete_time_index)

    # Fill missing 'Date' and 'Time' columns
    df['Date'] = df.index.date
    df['Time'] = df.index.time

    # Forward fill the 'Close' column to get the last recorded close value for missing rows
    df['Close'] = df['Close'].ffill()

    # Fill 'Open', 'High', 'Low' columns with the last recorded close value
    df['Open'] = df['Close']
    df['High'] = df['Close']
    df['Low'] = df['Close']

    # Fill missing 'Volume' values with 0
    df['Volume'] = df['Volume'].fillna(0)

    # Reset the index to get 'Time_Stamp' back as a column
    df = df.reset_index().rename(columns={'index': 'Time_Stamp'})
    try:
        engine = create_engine(
            f'postgresql+psycopg2://{settings.database_username}:{settings.database_password}@{settings.database_hostname}:{settings.database_port}/{settings.database_name}'
        )

        for i in range(df.shape[0]):
            if df.loc[i, 'Time_Stamp'] in list(data['Time_Stamp'].values):
                df.drop(i, axis=0, inplace=True)

        df.set_index('Time_Stamp').to_sql(con=engine, schema=settings.candlesticks_schema, name=ticker, if_exists='append')

        logger.info(
            f'{mp.current_process().name}: {ticker} took {time.time() - ticker_time} seconds to fill in {df.shape[0]} rows'
        )

        data = Database.speed_read(
            f'SELECT "Time_Stamp", "Date", "Time", "Open", "High", "Low", "Close", "Volume" FROM "Candlesticks"."{ticker}"'
        )
        
        missing_minutes = pd.date_range(start=data['Time_Stamp'].min(), end=data['Time_Stamp'].max(),
                                        freq='T').difference(data['Time_Stamp']).size

        if missing_minutes:
            print("No missing minutes in the dataset.")
        else:
            print(f"There are {missing_minutes} missing minutes in the dataset.")

    except Exception as e:
        raise ValueError(
            f'{mp.current_process().name}: {ticker} failed to interpolate missing data - {e}')


mp.process.active_children()


def batch(ticker: str, start: str, stop: str, core: int, logger: logging.Logger, shared_dict: dict):
    p = psutil.Process()
    p.cpu_affinity([core])
    print(f'{mp.current_process().name}: Building {ticker} table in {settings.database_name} DB from {start} to {stop}')
    days = int(abs((pd.Timestamp(stop) - pd.Timestamp(start)).days))
    day = pd.Timestamp(stop).strftime('%Y-%m-%d')
    sql_data = False
    while not sql_data:
        final_df = pd.DataFrame(columns=['Open', 'High', 'Close', 'Low', 'Volume', 'Time_Stamp', 'Date', 'Time'])
        consecutive_fails = 0
        ticker_time = time.time()
        current_day = 0
        while current_day < days:
            day_time = time.time()
            try:
                while True:
                    if not shared_dict['cbpro']:
                        df = cbpro_get(
                            ticker=ticker,
                            day=pd.Timestamp(day).strftime('%Y-%m-%d-%H-%M'),
                            shared_dict=shared_dict,
                            logger=logger
                        )
                        break

                if [*df.columns] == ['low', 'high', 'open', 'close', 'volume']:
                    df.reset_index(inplace=True)
                    df.rename(
                        columns={
                            'low': 'Low',
                            'high': 'High',
                            'open': 'Open',
                            'close': 'Close',
                            'volume': 'Volume',
                            'time': 'Time_Stamp'
                        },
                        inplace=True
                    )

                    df = df[['Open', 'High', 'Close', 'Low', 'Volume', 'Time_Stamp']]
                    df['Time_Stamp'] = pd.to_datetime(df['Time_Stamp'])
                    df['Date'] = df['Time_Stamp'].apply(lambda time_stamp: time_stamp.strftime('%Y-%m-%d'))
                    df['Time'] = df['Time_Stamp'].apply(lambda time_stamp: time_stamp.strftime('%H:%M:%S'))
                    df['Time'] = pd.to_datetime(df['Time'])

                final_df = pd.concat([df, final_df], axis=0)

                consecutive_fails = 0
                print(
                    f'{mp.current_process().name}: {current_day + 1} / {days} --- {day} took {time.time() - day_time} seconds')
                current_day += 1
                day = (pd.Timestamp(day) - pd.Timedelta('1 Days')).strftime('%Y-%m-%d')

            except BaseException as e:
                print(f'{mp.current_process().name}: {ticker} {day} Failed preprocessing - {e}')
                consecutive_fails += 1
                if consecutive_fails > 6:
                    logger.info(f'{mp.current_process().name}: {ticker} is not a live ticker before {day}')
                    print(f'{mp.current_process().name}: {ticker} is not a live ticker before {day}')
                    break
                else:
                    day = (pd.Timestamp(day) - pd.Timedelta('1 Days')).strftime('%Y-%m-%d')

        final_df['Date'] = pd.to_datetime(final_df['Date'])
        final_df['Time'] = pd.to_datetime(final_df['Time'])
        final_df['Date'] = final_df['Date'].apply(lambda time_stamp: time_stamp.strftime('%Y-%m-%d'))
        final_df['Time'] = final_df['Time'].apply(lambda time_stamp: time_stamp.strftime('%H:%M:%S'))
        final_df.reset_index(inplace=True, drop=True)

        if final_df.shape[0]:
            data_inserted = insert_new_data(ticker=ticker, data=final_df, logger=logger)
        else:
            data_inserted = False

        if data_inserted:
            sql_data = True
            logger.info(
                f'{mp.current_process().name}: {ticker} {day} - {stop} uploaded to database in {time.time() - ticker_time} seconds')
            print(
                f'{mp.current_process().name}: {ticker} {day} - {stop} uploaded to database in {time.time() - ticker_time} seconds')
        else:
            raise ValueError(
                f'{mp.current_process().name}: {ticker} {start} - {stop} Failed to insert data too many times')

    fill_in_missing_rows(ticker=ticker, core=core, logger=logger)
    query = f"SELECT * FROM \"{settings.candlesticks_schema}\".\"{ticker}\" ORDER BY \"Time_Stamp\""
    data_dict = cx.read_sql(
        conn=f'postgresql://{settings.database_username}:{settings.database_password}@{settings.database_hostname}:{settings.database_port}/{settings.database_name}',
        query=query).to_dict()

    for key in data_dict.keys():
        shared_dict[ticker]['data'][key] = data_dict[key]


def cbpro_to_sql(tickers: list, start_date: str, end_date: str, cores: int, logger: logging.Logger):
    ticker_time = time.time()
    if end_date is None:
        stop_ts = pd.to_datetime(time.time() * 10 ** 9).tz_localize('UTC').tz_convert('US/Eastern')
    else:
        stop_ts = pd.Timestamp(end_date)

    stop_str = stop_ts.strftime('%Y-%m-%d')

    start_ts = pd.Timestamp(start_date)
    start_str = start_ts.strftime('%Y-%m-%d')
    core_counter = 0
    processes = {}
    manager = mp.Manager()
    shared_dict = manager.dict()
    shared_dict['cbpro'] = False
    mp.process._process_counter = itertools.count(1)
    for ticker in tickers:
        shared_dict[ticker] = {'days': manager.list(), 'data': manager.dict()}
        core_counter += 1
        if core_counter == cores:
            mp.process._process_counter = itertools.count(1)
            core_counter = 1

        process = mp.Process(target=batch, args=(ticker, start_str, stop_str, core_counter, logger, shared_dict))
        if core_counter in processes.keys():
            processes[core_counter].append(process)
        else:
            processes[core_counter] = [process]

    current_batch = []
    processing = True
    while processing:
        for process in processes.keys():
            if not len(processes[process]):
                continue
            p = processes[process].pop(0)
            current_batch.append(p)

        if len(current_batch):
            for b in current_batch:
                b.start()

            for b in current_batch:
                b.join()

            for b in current_batch:
                b.kill()

            current_batch = []
        else:
            break

    ticker_dict = {}
    for ticker in tickers:
        ticker_df = pd.DataFrame(
            {col: [*shared_dict[ticker]['data'][col].values()] for col in shared_dict[ticker]['data'].keys()})
        ticker_dict[ticker] = {'days': [day for day in shared_dict[ticker]['days']], 'data': ticker_df}

    logger.info(f'{pd.Timedelta(pd.Timestamp(stop_ts.strftime("%Y-%m-%d")) - pd.Timestamp(start_date)).days} Days of {tickers} took {time.time() - ticker_time} seconds to upload to SQL')
    return ticker_dict
