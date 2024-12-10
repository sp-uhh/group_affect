from sklearn.mixture import GaussianMixture
from statistics import mean, variance, median, mode

import numpy as np

def get_aggregate_features_for_mimicry(distance_array, features=[min, max, mean, mode, median, variance]):
    mimicry_features = []
    for feature in features:
        mimicry_features.append(feature(distance_array))
    return mimicry_features

#Nanninga et al. -> IMplemented for Convergence Mixture Gauss based Mimicry,
def learn_mixuture_gaussian_model(indiv_data, n_components=1, covariance_type='full'):
    model = GaussianMixture(n_components=n_components, covariance_type=covariance_type).fit(indiv_data)
    return model

def get_log_likelihood_features_in_model(individual1, individual2):
    distance_array=[]
    model = learn_mixuture_gaussian_model(individual1.reshape(-1, 1), 1, "full")
    distance_array.extend(model.score_samples(individual2.reshape(-1, 1)))
    return distance_array

# Distance Based Mimicry - Based on Oyku's Thesis
def get_mimicry_sq_distance(individual1, individual2, agg_features):
    #From Indiv1's Sample 0 to n-1
    mimicry_features = []
    for channel in range(individual1.shape[1]):
        channel_len = individual1.shape[0]
        curr_channel1, curr_channel2 = individual1[:channel_len-1,channel], individual2[1:,channel]
        current_distance = (curr_channel1-curr_channel2)**2
        mimicry_features.extend(get_aggregate_features_for_mimicry(current_distance, agg_features))
    return mimicry_features


def get_asymmetric_mimicry(individual1, individual2, model_type="sq-distance"):
    mimicry_features=[]
    if model_type == "sq-distance":
        mimicry_features = get_mimicry_sq_distance(individual1, individual2,
                                                   agg_features=[min, max, mean, variance])
    elif model_type == "mix-gaussian":
        mimicry_features = get_log_likelihood_features_in_model(individual1, individual2)
    return mimicry_features

def get_mimicry_features(individual1, individual2, model_type="sq-distance"):
    # Indv1 Lagged Indv2
    mimicry_features = get_asymmetric_mimicry(individual1, individual2, model_type)
    # Indv2 Lagged Indv1
    mimicry_features.extend(get_asymmetric_mimicry(individual2, individual1, model_type))
    return mimicry_features