"""
.. moduleauthor:: Navin Raj Prabhu
"""

from scipy.spatial import distance
from scipy import signal
import numpy as np

from features.dyadic import mimicry_model
from features.utils.windowing import split_dataframe
from features.utils.preproc import standardize

class Convergence():
    """
    It computes the convergence over time (symmetric, asymmetric, or global) between two univariate signals x and y (in pandas DataFrame format).
    
    :param mode:
        Type of convergence to calculate: Either symmetric, asymmetric, or global
    :type mode: str
    
    :param plot:
        if True the plot of correlation function is returned. Default: False
    :type plot: bool
    
    :param standardization:
        if True the inputs are standardize to mean 0 and variance 1. Default: False
    :type standardization: bool
    
    """
    
    ''' Constructor '''
    def __init__(self, window_size, sr, mode='global', plot=False, standardization=True, agg=np.mean):
        super(Convergence, self).__init__()
        
        ' Raise error if parameters are not in the correct type '
        try :
            if not(isinstance(window_size, int))      : raise TypeError("Requires window_size to be an integer, in secs")
            if not(isinstance(sr, int))               : raise TypeError("Samplerate of input signals")
            if not(isinstance(mode, str))             : raise TypeError("Either symmetric, assymmetric, or global")
            if not(isinstance(plot, bool))            : raise TypeError("Requires plot to be a boolean")
            if not(isinstance(standardization, bool)) : raise TypeError("Requires standardization to be a boolean")
        except TypeError as err_msg:
            raise TypeError(err_msg)
            return
         
        self.sr = sr
        self.window_size = self.sr * window_size       
        self.mode=mode
        self.standardization=standardization
        self.plot = plot
        self.agg = agg

    def get_interdistance(self, x, y, metric='euc'):
        distance_i = 0
        if metric == 'euc':
            distance_i = distance.euclidean(x, y)
        elif metric == 'city':
            distance_i = distance.cityblock(x, y)
        elif metric == 'cosine':
            distance_i = distance.cosine(x, y)
        elif metric == 'correl':
            # If correl, the distance is reveresed in scale, more the better sim. So glb_conv is +ve if convergence
            distance_i = np.corrcoef(x, y)[0, 1]
            distance_i = -1*distance_i # scale reversed
        elif metric == 'lag-correl':
            # If correl, the distance is reveresed in scale, because intially if correl is higher then better similarity.
            # However, for interdistance(x, y) we want the distance value to be in scale of lower is better. 
            # So that glb_conv is +ve if convergence
            x = (x - np.mean(x)) / (np.std(x) * len(x))
            y = (y - np.mean(y)) / (np.std(y))
            correl_full = signal.correlate(x, y, mode='full')
            distance_i = -1*np.max(correl_full) # scale reversed
        else:
            #l1-norm , same as city ...
            distance_i = np.sum(np.abs(x - y))
        return distance_i

    def get_correl_with_time(self, x):
        '''
        Correlation returned is expected to be more negative for converging interactions (-vely correlated with Time),
        meaning that the participants tend to show similar behavior over time.
        
        Correl(distance, time) 
            * greater means bad convergence (distance increases with time)
            * smaller means good convergence (distance decreases with time)
        
        '''
        # * -1: Reverese scaled, for higher the better
        return np.corrcoef(x, np.arange(len(x)))[0, 1] * -1 

    def compute_symconv(self, x, y, metric='euc'):
        '''
        Correlation returned is expected to be more negative for converging interactions (-vely correlated with Time),
        meaning that the participants tend to show similar behavior over time
        '''
        squared_distance = (x-y)**2
        # Correlation between Tfeatureime and Evolving Distance ->
        sym_convergence = self.get_correl_with_time(squared_distance)
        return sym_convergence


    def compute_asymconv(self, x, y, learning_period = 2/3, metric='euc'):
        '''
            Correlation returned is expected to be more negative for 
            converging interactions (-vely correlated with Time),
            meaning that the participants tend to show similar behavior over time

            Currently - Mix Gauss Implemented
        '''

        # Split Samples based on learning_period
        splitter = int(len(x)*learning_period)
        distance_array = mimicry_model.get_log_likelihood_features_in_model(individual1=x[:splitter,], individual2=y[splitter:])
        asym_convergence = self.get_correl_with_time(distance_array)
        
        return asym_convergence

    def compute_glbconv(self, x, y):
        '''
        Similarity between both person’s first half’s features are computed using squared differences and saved as d0,
        and similarity between their second half’s features are computed and saved as d1. After that,
        the difference between these similarities is computed by subtraction as: c = d1 − d0.
        
        So: if c > 0, then good global convergence, c < 0 no convergence
        '''
        splitter = int(len(x)/2)
        met='lag-correl' 
        init_sim = self.get_interdistance(x[:splitter], y[:splitter], metric=met) # Higher => less Similarity
        latr_sim = self.get_interdistance(x[splitter:], y[splitter:], metric=met) # Higher => less Similarity
        glb_convergence = init_sim - latr_sim # (+1 - (-1) = 2 )
        return glb_convergence

    def compute(self, signals):
        x = signals[0]
        y = signals[1]

        if self.standardization==True:
            x=standardize(x)
            y=standardize(y)
            
        x = x.values[:,0]
        y = y.values[:,0]

        if self.mode == "symmetric":
            # print("Extracting Symmetric convergence .......")
            conv = self.compute_symconv(x, y)
        elif self.mode == "asymmetric":
            # print("Extracting Asymmetric convergence .......")
            # Assym-clmvergemce is assymetric - f(x, y) != f(y, x)
            conv = self.compute_asymconv(x, y)
        elif self.mode == "global":
            # print("Extracting Global convergence .......")
            conv = self.compute_glbconv(x, y)
        return conv
    
    # Main feature compute fn.,
    def compute_convergence(self, signals):
        if self.mode == 'global' or self.window_size == 0:
            return self.compute(signals=signals)
        
        x = signals[0]
        y = signals[1]
        x_split = split_dataframe(x, self.window_size)
        y_split = split_dataframe(y, self.window_size)

        conv_dynamics = []
        # print("Total Windows  = ", len(x_split))
        # Split
        for i, (x_seg, y_seg) in enumerate(zip(x_split, y_split)):
            # print("Extracting for ", i, "th window........")
            conv = self.compute([x_seg, y_seg])
            conv_dynamics.append(conv)
        return self.agg(conv_dynamics)