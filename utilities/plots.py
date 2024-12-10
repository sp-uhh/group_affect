

def set_plot_params(plt, figsize=(30, 30)):
    plt.rcParams.update({
        'figure.figsize' : figsize,
        'font.family' : 'Times New Roman',
        'font.serif' : ['Times New Roman'],
        'font.weight' : 'medium',
        'mathtext.fontset' : 'cm',
        'xtick.labelsize' : 25,
        'ytick.labelsize' : 25,
        'axes.labelsize' : 40,
        'axes.titlesize': 40,
        'lines.linewidth' : 1.5,
        'lines.markersize' : 12,
        'axes.xmargin' : 0.01,
        'legend.fontsize': 25,
        'legend.title_fontsize': 25,
        'legend.handlelength': 2
    })
    return plt