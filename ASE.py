import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.stats import multicomp
from scipy.optimize import minimize
from scipy.stats import norm, chi2


# loglikelihood_theta( ) computes the negative log-likelihood of observed allele-specific counts under a given value for theta (the over-dispersion parameter). 
# The likelihood is computed under the assumption that the number of reads mapping to the reference allele (h_1) follows a beta-binomial distribution, given the total number of reads.
# df is a data-frame corresponding to a unique sample (or ID), which contains allele-specific expression levels for each gene within that sample. 
def loglikelihood_theta(theta, df):
    pi = 0.5   # expected proportion of reads mapping to the reference allele 
    ll = 0     # log-likelihood of observing allele-specific reads across all genes for a given ID
    for gene in df.index:
        vec1 = np.arange(max(0, df.loc[gene, 'h1'] - 1) + 1)
        vec2 = np.arange(max(0, df.loc[gene, 'total'] - df.loc[gene, 'h1'] - 1) + 1)
        vec3 = np.arange(1, df.loc[gene, 'total'])
        part1 = np.log(pi + vec1*theta)
        part2 = np.log(1 - pi + vec2*theta)
        part3 = np.log(1 + vec3*theta)
        ll += np.sum(part1) + np.sum(part2) - np.sum(part3)
    return -ll   # negative log-likelihood   

# estimate_theta( ) minimizes loglikelihood_theta( ) with respect to theta for each sample and returns the maximum likelihood estimator theta_est. 
def estimate_theta(df):
    theta_mle = minimize(loglikelihood_theta, x0 = (1, ), bounds = ((1e-10,1.5), ), args = (df, ), method = 'L-BFGS-B')
    if not theta_mle.success:
        theta_mle = minimize(loglikelihood_theta, x0 = (1, ), bounds = ((0,10), ), args = (df, ), method = 'L-BFGS-B')
    theta_est = theta_mle.x[0]
    return theta_est

# For a given sample-gene pair, loglikelihood_pi( ) computes the negative log-likelihood of observing the data under a given value for pi and the maximum-likelihood estimator of theta.
def loglikelihood_pi(pi, theta_est, row):
    # h1: number of reads mapping to the reference allele
    # total: total number of reads mapping to a given gene
    h1, total = row 
    vec1 = np.arange(max(0, h1 - 1) + 1)
    vec2 = np.arange(max(0, total - h1 - 1) + 1)
    vec3 = np.arange(1, total)
    part1 = np.log(pi + vec1*theta_est)
    part2 = np.log(1 - pi + vec2*theta_est)
    part3 = np.log(1 + vec3*theta_est)
    ll = np.sum(part1) + np.sum(part2) - np.sum(part3)
    return -ll   # negative log-likelihood    

# estimate_pi( ) minimizes loglikelihood_pi( ) with respect to pi for each sample-gene pair and returns the maximum likelihood estimator pi_est.
def estimate_pi(row):
    minObj = minimize(loglikelihood_pi, x0 = (0.4, ), bounds = ((0.0000001, 0.999999), ), method = 'L-BFGS-B', 
                                args = (row['theta_est'], row[['h1', 'total']]))
    return minObj.x[0]

# beta_se( ) computes the standard error of beta_est. 
# beta_est is the estimator of beta, which is equal to logit(pi); i.e., allelic imbalance.
# This function uses the Fisher Information value combined with the delta method to approximate the standard error.
def beta_se(row):
    theta_est, pi_est, h1, total = row
    vec1 = np.arange(max(0, h1 - 1) + 1)
    vec2 = np.arange(max(0, total - h1 - 1) + 1)
    part1 = -(1/(pi_est + vec1*theta_est)**2)
    part2 = -(1/(1 - pi_est + vec2*theta_est)**2)
    FisherInfo = -(np.sum(part1) + np.sum(part2))    # Fisher Information    
    se = np.sqrt(1/FisherInfo)*(1/(pi_est*(1 - pi_est)))
    return se 

# likelihoodRatioTest( ) performs the following likelihood ratio test:
# H_0: pi = 0.5  vs  H_1: pi != 0.5, 
# and returns a chi-squared test-statistic with 1 degree of freedom.
# Rejecting H_0 is an indicator of allelic imbalance and hence at least one regulatory mutation in the vicinity of the gene.
def likelihoodRatioTest(row):
    likelihood_null = loglikelihood_pi(0.5, row['theta_est'], row[['h1', 'total']])
    likelihood_mle = loglikelihood_pi(row['pi_est'], row['theta_est'], row[['h1', 'total']])
    return -2*(likelihood_mle - likelihood_null)

# For a given gene, aggregate_beta( ) uses the inverse-variance weighting method, across all the individuals, to combine beta_est values and obtain a single aggregate beta value for that gene. 
# The Z-statistic := beta_agg/se(beta_agg) is computed to assess the following hypothesis:
# H_0: beta = 0  vs  H_1: beta != 0
def aggregate_beta(df):
    df['beta_inv_var'] = 1/(df.beta_se)**2
    agg_beta = np.sum(df.beta_est*df.beta_inv_var)/np.sum(df.beta_inv_var)    # aggregate beta computed for a given gene
    agg_beta_se = np.sqrt(1/np.sum(df.beta_inv_var))    # standard error for agg_beta
    agg_pvalue = 2*norm.cdf(-abs(agg_beta/agg_beta_se))
    return pd.Series([agg_beta, agg_beta_se, agg_pvalue], index = ['agg_beta', 'agg_beta_se', 'agg_pvalue'])

# Columns of data:
# ID: sample or individual
# h_1: number of reads mapping to the reference allele
# h_2: number of reads mapping to the alternate allele
# total: total number of reads overlapping the gene

# Statistical Model:
# h_1 ~ Binomial(total, pi)
# pi is assumed to be a random variable following a Beta distribution.
# This results in an over-dispersion parameter, theta. 
data = pd.read_csv('./Indiv_AF_L.txt', sep = ' ')
data.set_index('gene', inplace = True)

# Estimating theta for a given ID 
theta_series = data.groupby('ID').apply(estimate_theta)
data = pd.merge(data, theta_series.to_frame('theta_est'), left_on = 'ID', right_index = True, how = 'inner')

# Estimating pi, beta, and standard error of beta for a given gene-ID pair
data['pi_est'] = data.apply(lambda row: estimate_pi(row), axis = 1)
data['beta_est'] = np.log(data.pi_est/(1 - data.pi_est))
data['beta_se'] = data.loc[:, ['theta_est', 'pi_est', 'h1', 'total']].apply(lambda row: beta_se(row), axis = 1)
data['likelihood_ratio_pvalue'] = 1 - chi2.cdf(data.apply(lambda row: likelihoodRatioTest(row), axis = 1), 1)

# Estimating a weighted average of beta for each gene
aggregate_ASE = data.groupby(level = 0).apply(aggregate_beta)

# Correcting p-values for multiple testing
adjustedPvalues = multicomp.multipletests(aggregate_ASE.agg_pvalue, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)[0]
