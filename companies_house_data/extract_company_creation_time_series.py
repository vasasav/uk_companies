"""
Given simplified table, created by `extract_data.py`, group similar post-codes together and then extract ethe
number of companies created in that postcode collection on different months. Use the same monthly step for all
postcode address groups. Usage example

```
python extract_company_creation_time_series.py \
    --input_db_name=filtered_combined_data.parquet \
    --incorporation_period_start_date=2000-01-01 \
    --incorporation_period_end_date=2024-06-30
    --batch_save_h5_file=extracted_time_series_batch.h5
```

Example code to extract output:

```
import h5py

with h5py.File('extracted_time_series_batch.h5', 'r') as fh:
    time_series_mat=fh['time_series_mat'][:]
    simplified_pc_list=[pc.decode('utf-8') for pc in fh['utf-8_simplified_pc_list'][:]]

print(time_series_mat.shape)
```
"""

import duckdb as ddb
import numpy as np
import argparse
import typing as tp
import h5py


def create_monthly_count_tb(
        dbcon,
        output_table_name: str,
        period_start_date_str: str,
        period_end_date_str: str,
        relative_date_str: str,
        source_table: str='companies_house',
):
    """
    Create a table with simplified post-codes, months, and number of companies generated
    under these simplified postcodes. The simplified postcodes are created by throwing away
    some of the trailing characters. This leads to simplified postcodes that get assigned
    to georgraphically co-located addresses. The table is created in the DuckDB memory. Schema

    | simplified_pc | inc_monthly_count | relative_month |
    |---------------|-------------------|----------------|
    | GU73          |           1       |       95       |
    etc.

    To work, the function requires the `source_table` with:
        -  company_number
        - address_post_code
        - inc_date
    To be present

    Only the companies with incorporation date that falls within the specified period, will be
    grouped together. The incorporation date well be given as `relative_month` - number of
    months since the `relative_date_str`

    :param dbcon: initialized DuckDB connector with `source_table` inside it
    :param output_table_name: name of the table that will be created in DuckDB
    :param period_start_date_str: start date of the incorporation period YYYY-MM-DD
    :param period_end_date_str: end date of the incorporation period YYYY-MM-DD
    :param relative_date_str: date relative to which the incorporation period will be reported YYYY-MM-DD
    :param source_table: name of the table from which the data will be extracted
    :return: same dbcon
    """
    dbcon.execute(
        f"""
        CREATE TABLE {output_table_name} AS
        WITH
        post_code_vw AS (
            SELECT
                company_number,
                address_post_code,
                inc_date,
                DATE_DIFF('DAY', DATE '{relative_date_str}', inc_date) AS rel_day_count,
                DATE_DIFF('MONTH', DATE '{relative_date_str}', inc_date) AS rel_month_count,
                DATE_DIFF('YEAR', DATE '{relative_date_str}', inc_date) AS rel_year_count,
                REPLACE(SUBSTRING(address_post_code, 1, STRLEN(address_post_code)-2), ' ', '') AS simplified_pc
            FROM {source_table}
            WHERE
                inc_date >= DATE '{period_start_date_str}'
                AND inc_date <= DATE '{period_end_date_str}' 
                AND address_post_code IS NOT NULL
        )

        SELECT
            simplified_pc,
            COUNT(DISTINCT company_number) AS inc_monthly_count,
            rel_month_count AS relative_month
        FROM post_code_vw
        GROUP BY simplified_pc, rel_month_count
        """
    )

    return dbcon

###########################################

def extract_incroporation_series_batch(
        dbcon,
        inc_table_name: str,
        sort_hash_salt: str='basic_salt',
        int_batch_start_pos: int=0,
        int_batch_stop_pos: tp.Optional[int]=None,
        min_relative_month: tp.Optional[int]=None,
        max_relative_month: tp.Optional[int]=None
)->tp.Tuple[tp.List[str], np.ndarray]:
    """
    Extract time series data on company creation dates as N*M matrix, with N postcodes
    and M monthly time-steps, from min_relative_month ... max_relative_month
    if months are not specified, these will be taken as minimum and maximum possible

    Data is extracted from `inc_table_name` from dbcon. The postcodes get ordered
    by hash (with hash-salt `sort_hash_salt`) and then the selected batch
    is int_batch_start_pos...int_batch_stop_pos

    :param dbcon: connector to DuckDB. We expect to have table `inc_table_name` in there
                    see `create_monthly_count_tb`
    :param inc_table_name: name of the table to extract the data from
    :param sort_hash_salt: hash salt for selecting batch
    :param int_batch_start_pos:  batch start position
    :param int_batch_stop_pos: batch end position
    :param min_relative_month: minimum of the relative months to consider,
                if None, the minimum possible month is taken
    :param max_relative_month: maximum of the relative months to consider,
                if None, the maximum possible month is taken
    :return: [list of postodes], [N*M numpy matrix]
    """

    # get a list of all postcodes ordered by the hash-salt
    # and prepare the selection window
    pc_df = dbcon.execute(
        f"""
        WITH
        -- assign numbers to post-codes
        sorted_pc_vw AS (
            SELECT
                simplified_pc,
                ROW_NUMBER() OVER(ORDER BY HASH(CONCAT(simplified_pc, '{sort_hash_salt}'))) AS sorted_pc_order
            FROM (SELECT DISTINCT simplified_pc FROM {inc_table_name})
        )
        
        SELECT
            *
        FROM sorted_pc_vw
        """
    ).fetchdf()

    if int_batch_stop_pos is None:
        int_batch_stop_pos = pc_df.sorted_pc_order.max()

    # get the min-max months
    minmax_df = dbcon.execute(
        f"""
            WITH
            -- assign numbers to post-codes
            sorted_pc_vw AS (
                SELECT
                    simplified_pc,
                    ROW_NUMBER() OVER(ORDER BY HASH(CONCAT(simplified_pc, '{sort_hash_salt}'))) AS sorted_pc_order
                FROM (SELECT DISTINCT simplified_pc FROM {inc_table_name})
            )
            ,
            -- selected post-codes
            chosen_pc_vw AS (
                SELECT
                    simplified_pc
                FROM sorted_pc_vw
                WHERE 
                    sorted_pc_order >= {int_batch_start_pos}
                    AND
                    sorted_pc_order <= {int_batch_stop_pos}
            )
            ,
            -- select matching records
            chosen_vw AS (
                SELECT
                    *
                FROM {inc_table_name}
                WHERE simplified_pc IN (SELECT simplified_pc FROM chosen_pc_vw)
            )
            ,
            -- min/max relative month
            minmax_vw AS (
                SELECT
                    MIN(relative_month) AS min_relative_month,
                    MAX(relative_month) AS max_relative_month
                FROM chosen_vw
            )
            
            SELECT
                *
            FROM minmax_vw
            """
    ).fetchdf()

    if min_relative_month is None:
        min_relative_month = minmax_df.min_relative_month.iloc[0]

    if max_relative_month is None:
        max_relative_month = minmax_df.max_relative_month.iloc[0]

    # get the number of incorporated companies as table with arrays
    arr_df = dbcon.execute(
        f"""
            WITH
            -- assign numbers to post-codes
            sorted_pc_vw AS (
                SELECT
                    simplified_pc,
                    ROW_NUMBER() OVER(ORDER BY HASH(CONCAT(simplified_pc, '{sort_hash_salt}'))) AS sorted_pc_order
                FROM (SELECT DISTINCT simplified_pc FROM {inc_table_name})
            )
            ,
            -- selected post-codes
            chosen_pc_vw AS (
                SELECT
                    simplified_pc
                FROM sorted_pc_vw
                WHERE 
                    sorted_pc_order >= {int_batch_start_pos}
                    AND
                    sorted_pc_order <= {int_batch_stop_pos}
            )
            ,
            -- select matching records
            chosen_vw AS (
                SELECT
                    *
                FROM {inc_table_name}
                WHERE simplified_pc IN (SELECT simplified_pc FROM chosen_pc_vw)
            )
            ,
            -- months
            pc_month_vw AS (
                SELECT
                    M.* AS relative_month,
                    P.*
                FROM GENERATE_SERIES({min_relative_month}, {max_relative_month}) AS M
                CROSS JOIN chosen_pc_vw AS P
            )
            ,
            -- get counts for all the months
            chosen_all_months_vw AS (
                SELECT
                    M.simplified_pc,
                    M.relative_month,
                    IFNULL(C.inc_monthly_count, 0) AS inc_monthly_count,
                FROM pc_month_vw AS M
                LEFT JOIN chosen_vw AS C
                    ON C.relative_month=M.relative_month AND C.simplified_pc=M.simplified_pc
            )
            
            -- aggregate counts into arrays
            
            SELECT
                simplified_pc,
                ARRAY_AGG(inc_monthly_count ORDER BY relative_month) AS inc_monthly_count_arr
            FROM chosen_all_months_vw
            GROUP BY simplified_pc
            """
    ).fetchdf()

    # extract results as a list of N post-codes
    # and an N*M matrix with observations for M months
    pc_list = arr_df.simplified_pc.values
    inc_counts = np.stack(arr_df.inc_monthly_count_arr.values, axis=0)

    ## extrcact array with all the months present
    return pc_list, inc_counts

#######################

def main(
        input_db_name: str,
        period_start_date_str: str,
        period_end_date_str: str,
        batch_hash_salt: str='42fish',
        int_batch_start_pos: int=0,
        int_batch_stop_pos: tp.Optional[int]=None,
        batch_save_h5_file: tp.Optional[str]=None
)->tp.Tuple[tp.List[str], np.ndarray]:
    """
    Shorten postcodes to get the simplified postcodes, that group co-located companies. Then extract
    time-series of number of companies registered under each simplified postcode, monthly. Return
    results as list of simplified postcodes, and an N*M integer array for N simplified postcodes
    and M number of months

    There is an option to save output as an H5 file which will contain:
        time_series_mat - an N*M matrix full of integers
        batch_pc_list - array of simplified post code files, encoded as UTF-8

    :param input_db_name: source database, see `extract_incroporation_series_batch`
    :param period_start_date_str: start of the period within which to consider time series, see `extract_incroporation_series_batch`
    :param period_end_date_str: end of the period within which to consider time series, see `extract_incroporation_series_batch`
    :param batch_hash_salt: hash salt to order the simplified postcodes, to form the batch for extraction
    :param int_batch_start_pos: start position (integer) of the simplified postcodes batch, for extraction (inclusive)
    :param int_batch_stop_pos: stop position (integer) of the simplified postcodes batch, for extraction (exclusive)
    :param batch_save_h5_file: destination for the extracted batch file.
    :return: list of simplified post-codes, N*M time series matrix
    """

    # create a db connector
    print('Creating DB connector ... ', end='')
    dbcon = ddb.connect()
    dbcon.execute(f"CREATE TABLE companies_house AS FROM '{input_db_name}';")
    print('Done')

    ############
    print('Grouping companies under postcodes ... ', end='')
    intermediate_pc_aggregate_table = 'intermediate_pc_aggregate_table'
    create_monthly_count_tb(
        dbcon=dbcon,
        output_table_name=intermediate_pc_aggregate_table,
        period_start_date_str=period_start_date_str,
        period_end_date_str=period_end_date_str,
        relative_date_str=period_start_date_str
    )
    print('Done')

    ####

    print('Extracting time series batch ... ', end='')
    batch_pc_list, batch_counts_mat = extract_incroporation_series_batch(
        dbcon=dbcon,
        inc_table_name='intermediate_pc_aggregate_table',
        int_batch_start_pos=int_batch_start_pos,
        int_batch_stop_pos=int_batch_stop_pos,
        sort_hash_salt=batch_hash_salt
    )
    print('Done')

    ### saving results
    print('Saving result ... ', end='')
    if batch_save_h5_file is not None:
        # save the batch
        with h5py.File(batch_save_h5_file, 'w') as fh:
            fh.create_dataset('time_series_mat', data=batch_counts_mat)

            # use
            # ` with h5py.File('test.h5', 'r') as h5_fh: var=[pc.decode('utf-8') for pc in h5_fh['utf-8_simplified_pc_list'][:]];`
            # to extract
            fh.create_dataset('utf-8_simplified_pc_list', data=[pc.encode('utf-8') for pc in batch_pc_list])
            fh.create_dataset('utf-8_period_start_date_str', data=period_start_date_str.encode('utf-8'))
            fh.create_dataset('utf-8_period_end_date_str', data=period_end_date_str.encode('utf-8'))

        print(f'Done. {batch_save_h5_file}')
    else:
        print('Not needed')

    return batch_pc_list, batch_counts_mat

###########################################

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description="Extract time series of company creations")
    # argumets
    parser.add_argument(
        '--input_db_name',
        type=str,
        help='File name for the database with company house data (after `extract_data.py`)',
        required=True
    )
    #
    parser.add_argument(
        '--incorporation_period_start_date',
        type=str,
        help='Start date for the incorporation period under consideration (YYYY-MM-DD)',
        default='2000-01-01'
    )
    #
    parser.add_argument(
        '--incorporation_period_end_date',
        type=str,
        help='End date for the incorporation period under consideration (YYYY-MM-DD)',
        default='2024-06-01'
    )
    #
    parser.add_argument(
        '--batch_hash_salt',
        type=str,
        help='Random string that will be used as salt to order simplified post-codes for batch selection',
        default='42fish'
    )
    #
    parser.add_argument(
        '--int_batch_start_pos',
        type=int,
        help='Start position for the batch that will be extracted (inclusive)',
        default=0
    )
    #
    parser.add_argument(
        '--int_batch_stop_pos',
        type=int,
        help='Stop position for the batch that will be extracted (exclusive)',
        default=None
    )
    #
    parser.add_argument(
        '--batch_save_h5_file',
        type=str,
        help='Name of the H5 file into which the extracted batch will be saved',
        default='extracted_time_series_batch.h5'
    )
    # Parse the arguments
    args = parser.parse_args()
    # Extract the argument values

    main(
        input_db_name=args.input_db_name,
        period_start_date_str=args.incorporation_period_start_date,
        period_end_date_str = args.incorporation_period_end_date,
        batch_hash_salt=args.batch_hash_salt,
        int_batch_start_pos=args.int_batch_start_pos,
        int_batch_stop_pos=args.int_batch_stop_pos,
        batch_save_h5_file=args.batch_save_h5_file
    )

