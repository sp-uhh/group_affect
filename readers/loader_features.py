import numpy as np

import readers.memo_ga as MEMO
import utilities.calculation as calc

def load_individual_features(memoDB:MEMO, indiv_feats):
    for indivf in indiv_feats:
        memoDB.load_featureset(indivf)

def load_group_features(memoDB:MEMO, indiv_feats, sub_feats, agg_feats):
    # indiv_feats   = ["pitch", "pitch", "mfcc", "intensity", "aucs", "aucs", "facepose"]
    # sub_feats     = ["", "", "", "AU07", "AU20", "headroll"]
    # agg           = [np.mean, np.std, np.mean, np.mean, "synchrony_corrcoeff", "synchrony_maxcorr", "convergence_symmetric"]
    # Note: indiv_feats, sub_feats, agg should be ZIP-able
    for indivf, subf, agg in zip(indiv_feats, sub_feats, agg_feats):
        if "synchrony" in agg or "convergence" in agg:
            # e.g.,: 1) memoDB.load_synchrony_convergence_featset(ftype="convergence_symmetric", feature='aucs', sub_feature='AU07')
            # e.g.,: 2) memoDB.load_synchrony_convergence_featset(ftype="convergence_symmetric", feature='pitch', sub_feature='')
            memoDB.load_synchrony_convergence_featset(ftype=agg, feature=indivf, sub_feature=subf)
        else:
            # e.g.,: memoDB.load_aggregated_featset("pitch", agg_type=np.mean, normalize=False)
            agg_fn = get_agg_function(agg)
            memoDB.load_aggregated_featset(indivf, agg_type=agg_fn, normalize=False)

def get_agg_function(agg_type):
    if agg_type == "mean":
        return np.mean
    elif agg_type == "std":
        return np.std
    elif agg_type == "min":
        return np.amin
    elif agg_type == "max":
        return np.amax
    elif agg_type == "median":
        return np.median
    elif agg_type == "grad":
        return calc.calc_gradient