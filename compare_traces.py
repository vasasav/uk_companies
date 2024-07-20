"""
Display and number of companies incorporated as specific addresses using Streamlit. Assuming streamlit is
installed, display with:

```
streamlit run  compare_traces.py -- --time_series_h5=companies_house_data/extracted_time_series_batch.h5
```
"""

import streamlit as st
import pandas as pd
import os
import argparse
import h5py
import time
import plotly as ply
import plotly.express as plx
import plotly.graph_objects as pgo
import numpy as np
import poisson_trace_stats as pts
import scipy.stats as sp_st


######################
def main(
        time_series_h5: str
):
    st.title('Displaying company house data for grouped postcodes')

    with st.spinner('Loading data...'):
        # load time traces
        with h5py.File(time_series_h5, 'r') as fh:
            time_series_mat = fh['time_series_mat'][:]
            simplified_pc_list = [pc.decode('utf-8') for pc in fh['utf-8_simplified_pc_list'][:]]
            period_start_date_str = fh['utf-8_period_start_date_str'][()].decode('utf-8')
            period_end_date_str = fh['utf-8_period_end_date_str'][()].decode('utf-8')

    ########
    selected_grouped_pc = st.selectbox('Choose an option:', simplified_pc_list)

    # extract rate for the trace

    i_grouped_pc = simplified_pc_list.index(selected_grouped_pc)
    count_arr = np.squeeze(time_series_mat[i_grouped_pc,:])
    time_arr = np.arange(len(count_arr))

    with st.spinner('Computing rate...'):
        # load time traces
        time_for_rate_arr, rate_arr = pts.rate_trace_extract(
            count_arr=count_arr,
            time_arr=time_arr
        )

    ######### plot selected trace


    fig = pgo.Figure()

    conf_interval_color = 'rgb(100, 50, 100)'

    fig.add_trace(pgo.Scatter(
        x=time_for_rate_arr,
        y=rate_arr,
        mode='lines',
        fill='tonexty',
        name=f'per month rate',
        line={'color': 'rgba(0, 0, 0, 0)'},
        fillcolor='rgba(0,0,0,0)'
    ))
    fig.add_trace(pgo.Scatter(
        x=time_for_rate_arr,
        y=sp_st.poisson(rate_arr).ppf(0.95),
        mode='lines',
        name=f'rate confidence interval',
        fill='tonexty',
        line={'color': 'rgba(0, 0, 0, 0)'},
        fillcolor=conf_interval_color
    ))
    fig.add_trace(pgo.Scatter(
        x=time_for_rate_arr,
        y=sp_st.poisson(rate_arr).ppf(0.05),
        mode='lines',
        name=f'rate confidence interval',
        fill='tonexty',
        line={'color': 'rgba(0, 0, 0, 0)'},
        fillcolor=conf_interval_color
    ))
    fig.add_trace(pgo.Scatter(
        x=time_for_rate_arr,
        y=rate_arr,
        mode='lines',
        fill='tonexty',
        name=f'per month rate',
        line={'color': 'rgb(255, 150, 200)'},
        fillcolor='rgba(0,0,0,0)'
    ))

    fig.add_trace(pgo.Scatter(
        x=time_arr,
        y=count_arr,
        mode='markers',
        name=f'inc. counts {selected_grouped_pc}'
    ))

    fig.update_layout(
        showlegend=True,
        xaxis={
            'rangeslider': {'visible': True},
            'title': 'time in months'
        },
        yaxis={
            #'autorange': True,
            'fixedrange': False,
            'title': 'incorporated company counts'
        }
        ,
    )

    st.plotly_chart(fig)

    #########

######################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Load company incorporation data, grouped by postcode, and serve it as Streamlit app.'
    )
    #
    parser.add_argument(
        '--time_series_h5',
        type=str,
        help='path to H5 with time series of company creation counts',
        required=True
    )
    #
    args = parser.parse_args()

    main(
        time_series_h5=args.time_series_h5,
    )
