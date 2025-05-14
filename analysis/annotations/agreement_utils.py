import numpy as np
import pandas as pd
import krippendorff
import pingouin as pg
from itertools import combinations
from scipy.stats import kendalltau, pearsonr
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

from utilities import data as data_utils

# Main Function to be used to interact from other py files
def calc_agreement_for(labels_df, group='4', session='1', emo_dim='arousal', mode='cohen', submode=None):
    
    """_summary_

    Args:
        labels_df (_type_): pandas table of annotations: columns are annotators and rows are samples
        group (str, optional): ID of group in MEMO. Defaults to '4'.
        session (str, optional): ID of session in MEMO. Defaults to '1'.
        emo_dim (str, optional): The emotion dimension to work on 'arousal' or 'valence'. Defaults to 'arousal'.
        mode (str, optional): _description_. Defaults to 'cohen'.
        submode (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    interactionID  = group+'_'+session
    interaction_df =  labels_df.loc[labels_df['interactionID'] == interactionID] 
    
    # print(" INTER ANNOTATOR AGREEMENT FOR ", emo_dim ," group ", group, " session ", session)
    # BUG in ANNOTATION for Valence in group 14_3
    if group == "14" and session == "3" and "Valence_006" in interaction_df.columns:
        interaction_df = interaction_df.drop("Valence_006", axis = 1)
    
    aggrement_score = 0
    all_pair_scores = {}
    mean_labels = None
    if mode == 'cohen':
        aggrement_score, all_pair_scores = calculate_cohen_kappa(interaction_df, emo_dim, np.arange(1, 10), mode=submode)
    elif mode == 'fleiss':
        aggrement_score = calculate_fleiss_kappa(interaction_df, emo_dim, np.arange(1, 10), mode=submode)
    elif mode == 'pearson':
        aggrement_score, all_pair_scores = calculate_correlation(interaction_df, emo_dim, np.arange(1, 10), mode=mode)
    elif mode == 'kendal_rank':
        aggrement_score, all_pair_scores = calculate_correlation(interaction_df, emo_dim, np.arange(1, 10), mode=mode)
    elif mode == 'cronbach':
        aggrement_score = calculate_cronbachs_alpha(interaction_df, emo_dim, np.arange(1, 10), mode=submode)
    elif mode == 'krippendorff':
        aggrement_score = calculate_krippendorff_alpha(interaction_df, emo_dim, np.arange(1, 10), mode=submode)
    elif mode == "mse":
        aggrement_score, all_pair_scores, mean_labels = calculate_samplewise_agreement(interaction_df, emo_dim, mode="mse")
    elif mode == "abs":
        aggrement_score, all_pair_scores, mean_labels = calculate_samplewise_agreement(interaction_df, emo_dim, mode="abs")
    
    if mean_labels is None:
        return aggrement_score, all_pair_scores
    else:
        return aggrement_score, all_pair_scores, mean_labels

def calculate_krippendorff_alpha(annot_table, emo_dim='arousal', classes=np.arange(1, 10), mode='krippendorff'):
    annot_table = drop_na_annotators(annot_table)
    annot_table = drop_unwanted_cols(annot_table, emo_dim)
    annot_table = annot_table.values.transpose()
    alpha  = krippendorff.alpha(reliability_data=annot_table)
    
    return alpha



def calculate_cronbachs_alpha(annot_table, emo_dim='arousal', classes=np.arange(1, 10), mode='cronbach'):
     
    annot_table = drop_na_annotators(annot_table)
    annot_table = drop_unwanted_cols(annot_table, emo_dim)
    alpha, ci  = pg.cronbach_alpha(data=annot_table)
    
    return alpha

def calculate_samplewise_agreement(annot_table, emo_dim='arousal', mode="mse"):
    """_summary_

    Args:
        annot_table (_type_): table of annotations: columns are annotators and rows are samples
        emo_dim (string): The emotion dimension to operate on 'arousal' or 'valence'. Defaults to 'arousal'.
        mode (_type_, optional): mse- mean squared error /

    Returns:
        np.array: mse values array of annotator pairs, and importantly samplewise (for each time window)
    """
    annot_table = drop_unwanted_cols(annot_table, emo_dim)
    annot_table = drop_na_annotators(annot_table)

    all_pair_scores = dict.fromkeys([get_pairs_tag(i) for i in combinations(annot_table.columns, 2)])
    mean_labels = np.mean(annot_table.values, -1)

    for pairs in combinations(annot_table.columns, 2):
        var1 = annot_table.loc[:, pairs[0]].values
        var2 = annot_table.loc[:, pairs[1]].values
        if mode == "mse":
            curr_score = np.square(np.subtract(var1, var2)) #cohen_kappa_score(var1, var2, labels=classes, weights=mode)
        elif mode == "abs":
            curr_score = np.abs(np.subtract(var1, var2))
        all_pair_scores[get_pairs_tag(pairs)] = curr_score

    all_pairs_np = data_utils.convert_dictvalues_to_array(all_pair_scores)
    return np.mean(all_pairs_np, -1), all_pairs_np, mean_labels


def calculate_goldstandard_gt(annot_table, emo_dim='arousal'):
    
    _, pair_correls = calculate_correlation(annot_table, emo_dim, mode='pearson')
    pair_correls = data_utils.remove_none_in_dict(pair_correls)
    
    annot_table = drop_unwanted_cols(annot_table, emo_dim)
    annot_table = drop_na_annotators(annot_table)
    
    annot_rho_dict = {}
    for annotator in annot_table.columns:
        annot_dict = {key: pair_correls[key] for key in pair_correls.keys() if annotator.split("_")[-1] in key}
        annot_rho = np.mean(list(annot_dict.values()))
        annot_rho_dict[annotator] = annot_rho

    sum_rho = np.sum(list(annot_rho_dict.values()))
    
    nonorm_ewe_gt = None
    for annotator in annot_table.columns:
        if nonorm_ewe_gt is None:
            nonorm_ewe_gt = annot_table[annotator].values * annot_rho_dict[annotator] 
        else:
            nonorm_ewe_gt = nonorm_ewe_gt + (annot_table[annotator].values * annot_rho_dict[annotator])
    
    norm_ewe_gt = np.round(nonorm_ewe_gt/sum_rho, 4)
    mean_labels = np.mean(annot_table.values, -1)  

    return pair_correls, annot_rho_dict, mean_labels, norm_ewe_gt


    
def calculate_cohen_kappa(annot_table, emo_dim='arousal', classes=np.arange(1, 10), mode=None):
    """_summary_

    Args:
        annot_table (_type_): table of annotations: columns are annotators and rows are samples
        emo_dim (string): The emotion dimension to operate on 'arousal' or 'valence'. Defaults to 'arousal'.
        classes (_type_, optional): Annotation scale classes. Defaults to np.arange(1, 10).
        mode (_type_, optional): Weighted or vanilla cohens, None: vanilla, linear: linear weighted, quadratic: quadratic weighted. Defaults to None.

    Returns:
        np.array: kappa_scores array of annotator pairs
        float: mean kappa_score
    """
    annot_table = drop_unwanted_cols(annot_table, emo_dim)
    all_pair_scores = dict.fromkeys([get_pairs_tag(i) for i in combinations(annot_table.columns, 2)])
    annot_table = drop_na_annotators(annot_table)

    # print("Operating Cols = ", list(annot_table.columns))
    kappa_scores = []
    for pairs in combinations(annot_table.columns, 2):
        # print("Agreement for ", pairs[0], " and ", pairs[1])
        var1 = annot_table.loc[:, pairs[0]].values
        var2 = annot_table.loc[:, pairs[1]].values
        curr_kappa_score = cohen_kappa_score(var1, var2, labels=classes, weights=mode)
        all_pair_scores[get_pairs_tag(pairs)] = curr_kappa_score
        kappa_scores.append(curr_kappa_score)
    
    # print(kappa_scores)
    return np.mean(np.array(kappa_scores)), all_pair_scores

def calculate_fleiss_kappa(annot_table, emo_dim='arousal', classes=np.arange(1, 10), mode='fleiss'):
    """_summary_

    Args:
        annot_table (_type_): table of annotations: columns are annotators and rows are samples
        emo_dim (string): The emotion dimension to operate on 'arousal' or 'valence'. Defaults to 'arousal'.
        classes (_type_, optional): Annotation scale classes. Defaults to np.arange(1, 10).
        mode (_type_, optional): Mode of fleiss kappa calculation, 
                                ‘fleiss’ returns Fleiss’ kappa which uses the sample margin to define the chance outcome. 
                                Method ‘randolph’ or ‘uniform’ (only first 4 letters are needed) 
                                returns Randolph’s (2005) multirater kappa which assumes a uniform distribution of 
                                the categories to define the chance outcome. Defaults to 'fleiss'.

    Returns:
        _type_: fleiss_score, the final fleiss kappa score
    """
    
    annot_table = drop_na_annotators(annot_table)
    annot_table = drop_unwanted_cols(annot_table, emo_dim)
    annot_table = annot_table.astype(int).values

    fleiss_score = 0
    agg_annot_table, categories = aggregate_raters(annot_table, n_cat=int(classes[-1]+1))
    
    print(agg_annot_table)
    
    fleiss_score = fleiss_kappa(agg_annot_table, method=mode)
    
    return fleiss_score

def calculate_correlation(annot_table, emo_dim='arousal', classes=np.arange(1, 10), mode='pearson'):
    """_summary_

    Args:
        annot_table (_type_): table of annotations: columns are annotators and rows are samples
        emo_dim (string): The emotion dimension to operate on 'arousal' or 'valence'. Defaults to 'arousal'.
        classes (_type_, optional): _description_. Defaults to np.arange(1, 10).
        type (_type_, optional): Type of Correlation. Defaults to pearson.
                                 Options: pearson, kendal_rank

    Returns:
        np.array: correl_scores array of annotator pairs
        float: mean correl_score
    """
    
    annot_table = drop_unwanted_cols(annot_table, emo_dim)
    all_pair_scores = dict.fromkeys([get_pairs_tag(i) for i in combinations(annot_table.columns, 2)])
    annot_table = drop_na_annotators(annot_table)
    
    correl_scores = []
    for pairs in combinations(annot_table.columns, 2):
        # print("Agreement for ", pairs[0], " and ", pairs[1])
        var1 = annot_table.loc[:, pairs[0]].values
        var2 = annot_table.loc[:, pairs[1]].values
        if mode == 'pearson':
            curr_correl_score, p_val = pearsonr(var1, var2)
        elif mode == 'kendal_rank':
            curr_correl_score, p_val = kendalltau(var1, var2)
        all_pair_scores[get_pairs_tag(pairs)] = curr_correl_score
        correl_scores.append(curr_correl_score)
                    
    return np.mean(np.array(correl_scores)), all_pair_scores

def drop_na_annotators(annot_table):
    na_cols = []
    for i in range(len(annot_table.columns)):
        if  pd.isna(annot_table.iloc[:, i].values).any():
            na_cols.append(i)
    req_annot_table = annot_table.drop(annot_table.columns[na_cols],axis = 1)
    return req_annot_table

def drop_unwanted_cols(annot_table, emo_dim):
    na_cols = []
    for i in range(len(annot_table.columns)):
        if emo_dim not in annot_table.columns[i].lower():
            na_cols.append(i)
    req_annot_table = annot_table.drop(annot_table.columns[na_cols],axis = 1)
    return req_annot_table

def get_pairs_tag(pairs):
    return pairs[0].split("_")[-1]+"_"+pairs[1].split("_")[-1]