import pandas as pd
import numpy as np
import numpy.random as npr
import numpy.polynomial as np_poly


import scipy as sp
import scipy.stats as sp_st
import scipy.special as sp_sp

import sklearn as sk
import sklearn.linear_model as sk_lm

import typing as tp

##################################################



def polynomial_rate_trend_predict(
    count_arr: tp.List[int],
    time_arr: tp.List[float],
    predict_time: float,
    max_rate: float=1e4,
    poisson_fit_poly_degree: int=2,
    poisson_fit_max_iter_count: int=1000
)->float:
    """
    Given count data and time at which this count data was observed, use poisson regression to predict the rate 
    at time predict_time.

    So the poisson rate on i-th time is modelled as

    lambda_i=a^{(0)} + a^{(1)}*t_i+a^{(2)}*t_i^2 + ....

    :param count_arr: array of counts
    :param time_arr: array of times for the counts in `count_arr`
    :param predict_time: time for which the poisson rate prediction should be generated
    :param poisson_fit_poly_degree: degree of polynomial to fit for the dependence of rate on time. Must be at least 1, i.e. linear model
    :param poisson_fit_max_iter_count: parameter for the poisson regression, number of iterations to consider
    :return: predicted poisson rate for time `predict_time`
    """
    
    assert poisson_fit_poly_degree >= 0, 'Polynomial degree must be greater than zero'
    assert len(count_arr) > poisson_fit_poly_degree, 'Number of points needed to fit the regression constants has to be at least equal to number of constants'
    assert len(count_arr)==len(time_arr)

    if poisson_fit_poly_degree == 0:
        return np.mean(count_arr)

    ######
    # prepare the delay matrix, i.e. the design matrix
    # where each column is the time-array raised to the corresponding power
    # skipping the zeroth power because the Poisson regressor will be allowed
    # to fit the intercept explicitly
    delay_mat = np.zeros([len(count_arr), poisson_fit_poly_degree], dtype=float)
    #
    for i_order in range(poisson_fit_poly_degree):
        delay_mat[:, i_order] = time_arr[:]**(i_order+1)

    # fit Poisson regressor
    poisson_reg = sk_lm.PoissonRegressor(max_iter=poisson_fit_max_iter_count)
    poisson_reg.fit(delay_mat, count_arr)

    # predict the expected value, which for Poisson distribution is the rate    
    predict_arr = np.array([predict_time**(i_order+1) for i_order in range(poisson_fit_poly_degree)])[None,:]
    predicted_rate = np.clip(np.squeeze(poisson_reg.predict(predict_arr)), a_min=0.0, a_max=max_rate)

    return predicted_rate

##########################

def rate_trace_extract(    
    count_arr: tp.List[int],
    time_arr: tp.Optional[tp.List[float]]=None,
    poisson_fit_window_size: int=6,
    poisson_fit_poly_degree: int=2,
    poisson_fit_max_iter_count: int=1000
)->tp.Tuple[tp.List[float], tp.List[float]]:
    """
    Extract possion rate of counts for a trace of counts. For each position, select a window of preceding counts, 
    and use `polynomial_rate_trend_predict` to predict the rate for the current position

    Returns the poisson rates and the times for these rates. The rates for times that
    occur within the `poisson_fit_window_size`, at the beginning, are not computed,
    so the size of returned arrays is not len(count_arr), but len(count_arr)-poisson_fit_window_size

    :param count_arr: array of counts
    :param time_arr: array of times for the counts in `count_arr`
    :param poisson_fit_window_size: Window of preceding counts that will be used to estimate the current poisson rate
    :param poisson_fit_poly_degree: see `polynomial_rate_trend_predict` 
    :param poisson_fit_max_iter_count: see `polynomial_rate_trend_predict`
    :return:   predicted_time_arr - times
                predicted_rate_arr - rates
    """
    
    assert len(count_arr) > poisson_fit_window_size, 'Fit window size has to be smaller than the full count time series'
    assert (time_arr is None) or (len(time_arr)==len(count_arr)), \
        'If time array is given, the length of the time array must be the same as that of count array'
    
    predicted_rate_arr = np.zeros(len(count_arr)-poisson_fit_window_size, dtype=float)
    predicted_time_arr = np.zeros_like(predicted_rate_arr)

    # prepare time array, if it missing
    # assume equispaced data with later times occuring in higher indices
    if time_arr is None:
        time_arr = np.arange(len(count_arr)).astype(float)

    # order counts by increasing time
    i_temporal_order = np.argsort(time_arr)
    time_arr = time_arr[i_temporal_order]
    count_arr = count_arr[i_temporal_order]

    # step through time and count arrays
    for i_pt in range(poisson_fit_window_size, len(count_arr)):
        prior_count_arr = count_arr[(i_pt-poisson_fit_window_size):i_pt]
        
        # arrange time in such a way that it
        # is measured relative to time in the step just before
        # the current step
        # so if window size is six, then
        # rel_prior_time_arr = [-5, -4, -3, -2, -1, 0]
        # and rel_cur_time = 1
        # this should produce stable numerical behaviour
        rel_prior_time_arr = time_arr[(i_pt-poisson_fit_window_size):i_pt]
        rel_prior_time_arr -= time_arr[i_pt-1]
        rel_cur_time = time_arr[i_pt] - time_arr[i_pt-1]
    
        # get the current rate
        cur_rate = polynomial_rate_trend_predict(
            count_arr=prior_count_arr,
            time_arr=rel_prior_time_arr,
            poisson_fit_poly_degree=poisson_fit_poly_degree,
            predict_time=rel_cur_time,
            max_rate=np.mean(prior_count_arr)*10
        )
        #
        predicted_rate_arr[i_pt-poisson_fit_window_size] = cur_rate
        predicted_time_arr[i_pt-poisson_fit_window_size] = time_arr[i_pt] 

    return predicted_time_arr, predicted_rate_arr


##############################


def simulate_cumulative_likelihood_sample(
    rate_option_arr: tp.List[float],
    sample_count: int=1000,
    entropy_bin_count: int=20,
    eps: float=1e-6
)->tp.Tuple[tp.List[float], float]:
    """
    Given an array of possible Poisson rates, simulate random counts with these rates
    record the likelihood of such counts, under these rates. Return the likelihood
    array and the entropy of discritized distirbution of likelihoods. For continuous
    distributions, in the limit, such distribution of likelihoods, should
    tend to uniform. 

    :param count_arr: array of options for poisson rates
    :param sample_count: number of samples to draw for each histogram of likelihoods
    :param entropy_bin_count: number of bins for the histogram of likelihoods
    :param eps: number deemed to be small enough (in comparison with likelihoods)
    :return:   cumulative likelihood array, entropy computed from the histogram of this array
    """

    assert sample_count>0
    assert entropy_bin_count>1
    assert len(rate_option_arr)>0

    rate_arr = npr.choice(rate_option_arr, replace=True, size=sample_count)
    count_arr = sp_st.poisson(rate_arr).rvs()
    
    cumulative_likelihood_arr = sp_st.poisson(rate_arr).cdf(count_arr)

    bin_counts, _ = np.histogram(cumulative_likelihood_arr, bins=np.linspace(0-eps, 1+eps, entropy_bin_count+1))
    hist_entropy = np.sum([np.log(c)/c for c in bin_counts if c>0])
    
    return cumulative_likelihood_arr, hist_entropy


#######################

################
def compute_histogram_entropy_poisson(
    poisson_rate_arr: tp.List[float],
    counts_arr: tp.List[int],
    histogram_bin_count: int=20,
    eps: float=1e-6
)->float:
    """
    Given an array of poisson rates and the corresponding array of counts
    compute the cumulative likelihood of these counts under these rates. 
    Bin the cumulative likelihood into a histogram, then compute the
    entropy of that histogram

    :param poisson_rate_arr: array of rates for the poisson distribution
    :counts_arr: counts for these rates (observed)
    :histogram_bin_count: number of bins in the histogram of the cumulative likelihoods
    :param eps: small number compared to likelihoods
    :return: entropy of the histogram
    """
    
    cum_lkhd_arr = sp_st.poisson(poisson_rate_arr).cdf(counts_arr)
    bin_counts, _ = np.histogram(cum_lkhd_arr, bins=np.linspace(0-eps, 1+eps, histogram_bin_count+1))
    measured_hist_entropy = np.sum([np.log(c)/c for c in bin_counts if c>0])

    return measured_hist_entropy