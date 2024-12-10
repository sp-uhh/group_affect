import numpy as np
import pandas as pd

# Licence acknowledge form - https://github.com/syncpy/SyncPy/blob/master/src/Methods/utils/Detrend.py CeCILL-B license

def perform_detrend(signal,det_type):
    """
    It removes constant or linear trending in a monoviarate/multivariate signal (in pandas DataFrame format).
    In case of multivariate signal, detrending is carried out on each column of the DataFrame
    
    :param signal:
        input signal
    :type signal: pd.DataFrame
    
    :param det_type:
        {'mean' 'linear'} 
    :type det_type: str
    
    :returns: pd.DataFrame 
            -- detrended signal
    """
    
    ' Raise error if parameters are not in the correct type '
    try :
        if not(isinstance(signal, pd.DataFrame)) : raise TypeError("Requires signal to be a pd.DataFrame")
        if not(isinstance(det_type, str))      : raise TypeError("Requires det_type to be a string")
    except TypeError as err_msg:
        raise TypeError(err_msg)
        return
    
    
    ' Raise error if parameters do not respect input rules '
    try : 
        if det_type!='mean' and det_type!='linear' : raise ValueError("Requires det_type to be 'mean' or 'linear'")
    except ValueError as err_msg:
        raise ValueError(err_msg)
        return
    

    if det_type=='mean':
        signal_det=signal-signal.mean(axis=0)
    
    elif det_type=='linear':
        signal=signal.reset_index().drop('index',1)
        signal_det_tot=pd.DataFrame()
        
        x_signal=pd.Series(np.arange(0,signal.shape[0]))
        
    
        for k in range(0,signal.shape[1]):
            beta=x_signal.cov(signal.iloc[:,k])/x_signal.var(axis=0)
            alpha=signal.iloc[:,k].mean(axis=0)-beta*(x_signal.mean(axis=0))          
            
            signal_det=signal.iloc[:,k]-(alpha + beta*x_signal)
            signal_det_tot.loc[:,k]=signal_det
            signal_det=signal_det_tot
        
    return (signal_det)




