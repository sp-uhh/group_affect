"""
.. moduleauthor:: Navin Raj Prabhu
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_result_corr(self):
        """
        It plots the correlation function in the range specified.

        :returns: plt.figure 
         -- figure plot
        """

        result = self.res

        ' Raise error if parameters are not in the correct type '
        try :
            if not(isinstance(result, dict)) : raise TypeError("Requires result to be a dictionary")
        except TypeError as err_msg:
            raise TypeError(err_msg)
            return
        
        ' Raise error if not the good dictionary '
        try : 
            if not 'corr_funct' in result : raise ValueError("Requires dictionary to be the output of compute() method")
            if not 'tau_array' in result : raise ValueError("Requires dictionary to be the output of compute() method")
        except ValueError as err_msg:
            raise ValueError(err_msg)
            return
        
        figure = plt.figure() # Define a plot figure 
        ax = figure.add_subplot(111) # Add axis on the figure
        
        ax.set_ylabel('Value')
        ax.set_xlabel('Lag')
        ax.set_title('Correlation Function')
        #ax.set_xlim(max(-self.tau_max, (- (self.ly - 1))),min(self.tau_max, (self.lx - 1)))
        ax.set_ylim(np.min(result['corr_funct']),np.max(result['corr_funct']))
                
        ax.plot(result['tau_array'], result['corr_funct'])
        
        return figure
    
    