# uk_companies

Statistics on company incorporation dates from UK Companies House. 

## Setup steps

0. Assuming presence of libraries such as `numpy`, `matplotlib`, `scipy`, `h5py`, `duckdb`, `pandas`

1. Navigate to []companies house website](https://download.companieshouse.gov.uk/en_output.html) and download the basic data. Here we are expecting it to be downloaded as multiple files, e.g. `BasicCompanyData-2024-07-01-part1_7.zip` etc... Place files into `companies_house_data`

2. Run `python extract_data.py --output_db_name=combined_data.parquet` to extract, clean and combine the companies house data into a single PARQUET file. The file of interest will be `filtered_combined_data.parquet`

3. Run 
```
python extract_company_creation_time_series.py \
    --input_db_name=filtered_combined_data.parquet \
    --incorporation_period_start_date=2000-01-01 \
    --incorporation_period_end_date=2024-06-30
    --batch_save_h5_file=extracted_time_series_batch.h5
```
