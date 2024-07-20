"""
Will find every ZIP file in the directory, treat is as part of the BasicCompanyData export, load it
and incoroporate all into a single lightly processed parquet file. `combined_data.parquet` & `filtered_combined_data.parquet`

Most properties here are hard-coded since we are so close to data here

Usage example

```
python extract_data.py --output_db_name=combined_data.parquet
```
"""

import duckdb
import os
import subprocess as sproc
import argparse

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description="Extract company house data from CSVs into a single PARQUET file")
    # Add the --output_db_name argument
    parser.add_argument(
        '--output_db_name',
        type=str,
        help='Name of the output database',
        default='combined_data.parquet'
    )
    # Parse the arguments
    args = parser.parse_args()
    # Extract the value of --output_db_name
    output_db_name = args.output_db_name

    print('Unzipping files')

    # get the list of ZIP files in the current directory and unzip them if needed
    dir_path = os.path.abspath('.')
    csv_path_list = []

    # prepare db context
    con = duckdb.connect()

    # Traverse the directory structure
    file_counter = 1
    tb_name_list = []
    for root, dirs, files in os.walk(dir_path):
        for file_path in files:
            if file_path.lower().endswith('.zip'):
                zip_path = os.path.join(dir_path, file_path)
                suggested_csv_path = zip_path.replace('.zip', '.csv')

                # extract files from archives if needed
                if not os.path.isfile(suggested_csv_path):
                    print(f'\textracting {suggested_csv_path}...', end='')
                    result = sproc.run(
                        ["unzip", zip_path, '-d', dir_path],
                        check=True,  # Raises an exception if the command exits with a non-zero status
                        capture_output=True  # Captures stdout and stderr
                    )
                    print('done')
                else:
                    print(f'\t{suggested_csv_path} already exists')

                csv_path_list.append(suggested_csv_path)

                # load into database
                cur_tb_name = f'compnanies_house_tb_{file_counter}'
                con.execute(
                    f"""
                    CREATE TABLE {cur_tb_name} AS FROM '{suggested_csv_path}';
                    """
                )
                tb_name_list.append(cur_tb_name)
                file_counter += 1
                print(f'\tLoaded table {cur_tb_name}')

    # combine into a single table
    combine_sql = 'SELECT * FROM ('
    combine_sql += ' UNION ALL '.join([f' SELECT * FROM {tb_name} ' for tb_name in tb_name_list])
    combine_sql += ')'

    # export to a single clean file
    export_file = output_db_name.replace('.parquet', '') + '.parquet'
    con.execute(
        f"""
        COPY ({combine_sql}) TO {export_file} (FORMAT PARQUET);
        """
    )
    print(f'Results have been exported to {export_file}')

    # export a cleaner version of data with chosen useful columns
    filt_export_file = f'filtered_{export_file}'
    con.execute(
        f"""
        COPY (
            SELECT
                T.CompanyName AS company_name,
                T.CompanyNumber AS company_number,
                T.URI AS company_uri,
                T."SICCode.SicText_1" AS sic_code,
                T.CompanyStatus AS company_status,
                T.CompanyCategory AS company_category,
                T."RegAddress.AddressLine1" AS address_line,
                T."RegAddress.PostTown" AS address_town,
                T."RegAddress.PostCode" AS address_post_code,
                T.IncorporationDate AS inc_date,
                T."Accounts.AccountCategory" AS acc_cat
            FROM ({combine_sql}) AS T
        ) TO {filt_export_file} (FORMAT PARQUET);
        """
    )
    print(f'Filtered results have been exported to {filt_export_file}')

