"""
.. moduleauthor:: Navin Raj Prabhu
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal

from features.utils.plots import plot_result_corr
from features.utils.preproc import standardize
from features.utils.windowing import split_dataframe


class Correlation():
    """
    It computes the linear correlation between two univariate signals x and y (in pandas DataFrame format) as a function of their delay tau.
    It computes autocorrelation when y coincides with x.
    
    :param window_size:
        The analysis window size. if window_size > 0, correlation is calculated within windows of size window_size and aggregated using average, 
        dynamics of this window_size-based correlation will also be saved 
    :type window_size: int
    
    :param tau_max:
        the maximum lag (in samples) at which correlation should be computed. It is in the range [0; (length(x)+length(y)-1)/2] 
    :type tau_max: int
    
    :param plot:
        if True the plot of correlation function is returned. Default: False
    :type plot: bool
    
    :param standardization:
        if True the inputs are standardize to mean 0 and variance 1. Default: False
    :type standardization: bool
    
    :param corr_tau_max:
        if True the maximum of correlation and its lag are returned. Default: False
    :type corr_tau_max: bool
    
    :param corr_coeff:
        if True the correlation coefficient (Pearson's version) is computed. It is enabled only if the parameter standardize is True. Default: False
    :type corr_coeff: bool
    
    :param scale:
        if True the correlation function is scaled in the range [-1;1]
    :type scale: bool
    """
    
    ''' Constructor '''
    def __init__(self, window_size, sr, tau_max, plot=False, standardization=False, corr_tau_max=False, corr_coeff=False, scale=False, agg=np.mean):
        super(Correlation, self).__init__()
        
        ' Raise error if parameters are not in the correct type '
        try :
            if not(isinstance(window_size, int))      : raise TypeError("Requires window_size to be an integer, in secs")
            if not(isinstance(sr, int))               : raise TypeError("Samplerate of input signals")
            if not(isinstance(tau_max, int))          : raise TypeError("Requires tau_max to be an integer")
            if not(isinstance(plot, bool))            : raise TypeError("Requires plot to be a boolean")
            if not(isinstance(standardization, bool)) : raise TypeError("Requires standardization to be a boolean")
            if not(isinstance(corr_tau_max, bool))    : raise TypeError("Requires corr_tau_max to be a boolean")
            if not(isinstance(corr_coeff, bool))      : raise TypeError("Requires corr_coeff to be a boolean")
            if not(isinstance(scale, bool))           : raise TypeError("Requires scale to be a boolean")
        except TypeError as err_msg:
            raise TypeError(err_msg)
            return
        
        ' Raise error if parameters do not respect input rules '
        try : 
            if tau_max < 0 : raise ValueError("Requires tau_max to be a strictly positive scalar")
            if window_size < 0 : raise ValueError("Requires window_size to be a strictly positive scalar")
        except ValueError as err_msg:
            raise ValueError(err_msg)
            return
        
        self.sr = sr
        self.window_size = self.sr * window_size
        self.tau_max=tau_max
        self.standardization=standardization
        self.corr_tau_max=corr_tau_max
        self.corr_coeff=corr_coeff
        self.scale=scale
        self.plot = plot
        self.agg = agg
    
    
    def compute_tau_range(self, lx, ly):
        """
        Computes the range of tau values the correlation function is returned for.
        
        :param self.lx:
            length of the first input signal
        :type self.lx: int
        
        :param self.ly:
            length of the second input signal
        :type self.ly: int
        
        :returns: numpy.array 
          -- the range of tau values the correlation function is returned for
        """
        ll=max(-self.tau_max,(-(ly - 1)))
        ul=min(self.tau_max,(lx - 1))+1

        tau_array=np.arange(ll,ul,1)
        # print("tau_array - ", tau_array)
        start = tau_array[0]+(ly-1)
        stop = tau_array[tau_array.size - 1] + (ly-1)
        
        return tau_array, start, stop
    
    
    def compute_coeff(self, corr_f, lmin, ly):
        """
        It computes the Pearson's correlation coefficient.
        
        :param corr_f:
            correlation function
        :type corr_f: numpy.array
        
        :param lmin:
            the length of the shortest input 
        :type limn: int
        
        :param ly:
            length of the second input 
        :type ly: int
        
        :returns: numpy.array 
          -- time/Pearson's correlation coefficient
        """
        
        corr_coeff=corr_f[ly-1]/(lmin-1)
        
        return corr_coeff
         
    def compute(self, signals):
        """
        It computes the correlation function between x and y
        
        :param signals:
            array containing the 2 signals as pd.DataFrame
        :type signals: list

        :returns: dict 
                -- correlation function,
                    maximum of correlation and 
                    lag/Pearson's coefficient
        """
        x, y = self._validate_args(signals)
        
        lmax=max(self.lx,self.ly)
        lmin=min(self.lx,self.ly)
        
        self.tau_array, start, stop =self.compute_tau_range(self.lx,self.ly)
            
        if self.standardization==False:
            self.corr_f_full=signal.correlate(x.iloc[:,0],y.iloc[:,0], mode='full')
            self.corr_f=self.corr_f_full[start:stop+1]
        else:
            x_std=standardize(x)
            y_std=standardize(y)
                              
            self.corr_f_full=signal.correlate(x_std.iloc[:,0], y_std.iloc[:,0], mode='full')
            self.corr_f=self.corr_f_full[start:stop+1]
        
        if self.scale==True:
            nx=np.linalg.norm(x.values,2)
            ny=np.linalg.norm(y.values,2)
            self.corr_f=self.corr_f_full[start:stop+1]/(nx*ny)

        res_corr={}
        res_corr['corr_funct']=self.corr_f # Correlation  for all possible "lag" tau 
        
        if self.corr_tau_max : 
            max_corr=np.amax(self.corr_f)
            t_max=np.argmax(self.corr_f)
            t_max=self.tau_array[t_max]
            corr_coeff = self.compute_coeff(self.corr_f_full, lmin, self.ly)
            res_corr['max_corr']=max_corr # Max correlation
            res_corr['t_max']=t_max # Its lag tau (of max correlation)

        if self.corr_coeff : 
            corr_coeff = self.compute_coeff(self.corr_f_full, lmin, self.ly)
            res_corr['corr_coeff']=corr_coeff # Pearson's coefficient
        
        res_corr['tau_array']=self.tau_array # Lag Tau range used

        self.res = res_corr
        # TODO: Plot utils and save w.r.t windows
        # if self.plot:
        #     corr_fig = plot_result_corr(res_corr)

        return res_corr
    

    def windowed_compute(self, signals):
        """
        It computes the correlation function between x and y, in an windowed fashioN.
        i.e. signal is split into segments of self.window_size
        Calcualte PCC, max_tau, and max_corrcoeff w.r.t the segments, and aggregate using self.agg
        
        :param signals:
            array containing the 2 signals as pd.DataFrame
        :type signals: list

        :returns: dict 
            contains (for all window segments),
                -- list of maximum of correlation 
                -- list of its lag (for max correl)
                -- list of Pearson's coefficient
        """
        if self.window_size == 0:
            return self.compute(signals)

        x, y = self._validate_args(signals) 
        
        max_corr = []
        t_max = []
        corr_coeff = []

        x_split = split_dataframe(x, self.window_size)
        y_split = split_dataframe(y, self.window_size)

        print("Total Windows  = ", len(x_split))
        # Split
        for i, (x_seg, y_seg) in tqdm(enumerate(zip(x_split, y_split))):
            # print("Extracting for ", i, "th window........")
            lx_seg = x_seg.shape[0]
            ly_seg = y_seg.shape[0]

            lmax_seg=max(lx_seg, ly_seg)
            lmin_seg=min(lx_seg, ly_seg)
            
            tau_array_seg, start_seg, stop_seg  = self.compute_tau_range(lx_seg, ly_seg)
            
            if self.standardization==True:
                x_seg=standardize(x_seg)
                y_seg=standardize(y_seg)  
            
            x_seg = x_seg.iloc[:,0]
            y_seg = y_seg.iloc[:,0]
            
            # x_seg = (x_seg - np.mean(x_seg)) / (np.std(x_seg) * len(x_seg))
            # y_seg = (y_seg - np.mean(y_seg)) / (np.std(y_seg))
            
            corr_f_full_seg = signal.correlate(x_seg ,y_seg, mode='full')
            corr_f_seg = corr_f_full_seg[start_seg:stop_seg+1] # Correlation  for all possible "lag" tau - corr_f_seg

            if self.scale:
                nx = np.linalg.norm(x_seg.values,2)
                ny = np.linalg.norm(y_seg.values,2)
                corr_f_seg = corr_f_full_seg[start_seg:stop_seg+1]/(nx*ny)
                    
            # Max lagged-correlation
            max_corr_seg = np.amax(corr_f_seg) 
            # Its lag tau (of max correlation)
            t_max_seg = signal.correlation_lags(x.size, y.size, mode="full")[np.argmax(corr_f_seg)]
            # t_max_seg = tau_array_seg[np.argmax(corr_f_seg)] 
            print(" tau_array_seg -= ", tau_array_seg)
            print(" np.argmax(corr_f_seg) -= ", np.argmax(corr_f_seg))
            print(" tau_array_seg[np.argmax(corr_f_seg)] -= ", tau_array_seg[np.argmax(corr_f_seg)])
            
            corr_coeff_seg = self.compute_coeff(corr_f_full_seg, lmin_seg, ly_seg) # Pearson's coefficient

            max_corr.append(max_corr_seg)
            t_max.append(t_max_seg)
            print("t_max = ", t_max) 
            corr_coeff.append(corr_coeff_seg)

        self.windowed_res = {
            "dynamics"      : {
                "max_corr"      : max_corr,
                "t_max"         : t_max,
                "corr_coeff"    : corr_coeff,
            },
            "aggregated"    : {
                "max_corr"      : self.agg(max_corr),
                "t_max"         : self.agg(t_max),
                "corr_coeff"    : self.agg(corr_coeff),
            }
        }

        return self.windowed_res


    # Validate Utils
    def _validate_args(self, signals):
        try:
            if not (isinstance(signals, list)): raise TypeError("Requires signals be an array")
            if len(signals) != 2: raise TypeError("Requires signals be an array of two elements")
        except TypeError as err_msg:
            raise TypeError(err_msg)

        x = signals[0]
        y = signals[1]

        self.lx=x.shape[0]
        self.ly=y.shape[0]
        
        ' Raise error if parameters are not in the correct type '
        try :
            if not(isinstance(x, pd.DataFrame)) : raise TypeError("Requires x to be a pd.DataFrame")
            if not(isinstance(y, pd.DataFrame)) : raise TypeError("Requires y to be a pd.DataFrame")
        except TypeError as err_msg:
            raise TypeError(err_msg)
            return
        
        ' Raise error if parameters do not respect input rules '
        try : 
            if self.tau_max < 0 or self.tau_max >(self.lx-1) : 
                # raise ValueError("Requires tau_max to be in the range [0,length x -1]")         
                self.tau_max = self.lx-1
        except ValueError as err_msg:
            raise ValueError(err_msg)
            return
        
        ' Raise warnings '
        try : 
            if self.standardization==False and self.corr_coeff==True :
                raise Warning("Warning! The computation of the correlation coefficient is enabled only when the time series are standardized")
            if self.scale==True and (x.shape[0]!=y.shape[0]) :
                raise Warning("Warning! The computation of scaled correlation function is enabled only when the time series have the same length")
            if self.tau_max > self.ly :
                raise Warning("the value -(length y -1) will be used as -tau_max")       
        except Warning as war_msg:
            raise Warning(war_msg)

        return x, y