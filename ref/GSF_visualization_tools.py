import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import modules.ongoing_mod as ongm
import modules.utils as utls
# 
def plot_data(data, var, tit, st_anlss=None, supt_fonts = 20):
    '''
    Visualise the data over time and a histogram. 
    '''
    f, (ax1, ax2) = plt.subplots(1,2,  figsize=(12, 8),facecolor='.85') #12, 8
    
    ax1.scatter(data['DATESTAMP'], data[var], s = 2, label = var)
    num_ticks = 5
    tick_positions = np.linspace(0, len(data) - 1, num_ticks, dtype=int)
    xtick_labels = data['DATESTAMP'].iloc[tick_positions].dt.strftime('%m-%d %H:%M') #'%Y-%m-%d %H:%M'
    ax1.set_xticks(data['DATESTAMP'].iloc[tick_positions])
    ax1.set_xticklabels(xtick_labels, rotation=20, fontsize=10)
    ax1.set_ylabel(var)
    ax1.set_xlabel('Date - Time')
    ax1.legend(loc = 1,fontsize=12)
    
    sns.histplot(data[var], bins=30, kde=False, ax=ax2, color="lightgray", edgecolor="gray", alpha=0.7, stat="density")
    if data[var].var() != 0 and not np.isnan(data[var].var()):
        sns.kdeplot(data[var], color="blue", ax=ax2)
    ax2.set_xlabel(var)
    ax2.set_ylabel("Frequency / Density")
    if st_anlss:
        props = dict(boxstyle='round', facecolor='blue', alpha=0.1)
        ax2.text(0.6, 0.95, st_anlss, transform=ax2.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
    f.suptitle(f'{var}\n {tit}',fontsize=supt_fonts, x=0.5,y=.98, weight='semibold')
    plt.tight_layout()
    
    return ax1, ax2, f

def plot_vars(data, var1, var2=None, tit_n = None, ax1_ylim=None, ax2_ylim=None, axvspan_inds=None):
    '''
    Visualise one or two variables over time.
    '''   
    f, ax1 = plt.subplots(figsize=(6,6),facecolor='.85')
    ax1.scatter(data['DATESTAMP'], data[var1], s = 10, color='b')
    
    num_ticks = 5
    tick_positions = np.linspace(0, len(data) - 1, num_ticks, dtype=int)
    xtick_labels = data['DATESTAMP'].iloc[tick_positions].dt.strftime('%m-%d %H:%M') #'%Y-%m-%d %H:%M'
    ax1.set_xticks(data['DATESTAMP'].iloc[tick_positions])
    ax1.set_xticklabels(xtick_labels, rotation=20, fontsize=10)
    
    if var2:
        ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
        ax2.scatter(data['DATESTAMP'], data[var2], s = 1, color='r')
        ax2.tick_params(axis='y', labelcolor='r', colors='r')
        ax2.set_ylabel(var2, color='r')
        
    if axvspan_inds:
        for ind in axvspan_inds:
            ax1.axvspan(data['DATESTAMP'][ind[0]], data['DATESTAMP'][ind[1]], alpha=.3, color='lightgrey')
    if ax1_ylim: ax1.set_ylim(ax1_ylim)
    if ax2_ylim: ax2.set_ylim(ax2_ylim)  
    
    ax1.set_xlabel('Date - Time')
    ax1.tick_params(axis='y', labelcolor='b', colors='b')
    ax1.set_ylabel(var1, color='b')
    plt.tight_layout()
    if tit_n: plt.title(tit_n, fontsize=15)
    return ax1

def plot_two_vars(data, var1, var2, coef = None, intercept = None):
    '''
    Visualise var1 in x and var2 in y.
    '''   
    f, ax = plt.subplots(figsize=(8,8),facecolor='.85')
    ax.scatter(data[var1], data[var2], s = 1)
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    if coef and intercept:
        min_v = data[var1].min()
        max_v = data[var1].max()
        x_lin_m = np.linspace(min_v-abs(max_v-min_v)*.01,max_v+abs(max_v-min_v)*.01,2)
        y_lin_m = coef*x_lin_m+intercept
        ax.plot(x_lin_m, y_lin_m, label='Lin. regress. mod.', color='red', linestyle='--')
        ax.legend()
    return ax

# =============================================================================
# 
# =============================================================================

def variable_hists(data, var_name, tit = None, plot_mean = True, xlims= None):
    '''
    Visualise a set of historms.
    ----------
    data: [series dict / np arrays dict / list dict / df]
    var_name: [str] | descrption | Def. arg.: default arg
    tit: [str] | title name | Def. arg.: None
    plot_mean: [bool] | Def. arg.: None
    xlims: [list of lims] | Def. arg.: None
    '''
    if isinstance(data, pd.DataFrame): data = data.to_dict(orient = 'series')
    if isinstance(next(iter(data.values()), None), list):
        data = {key: np.array(value) for key, value in data.items()}
    data_copy = data.copy()
    f, ax = plt.subplots(figsize=(8,8),facecolor='.85')
    for case_n , value in data_copy.items():
        if value.var() != 0:
            sns.kdeplot(data=value, fill=True, label=case_n)
        else:
            data.pop(case_n)
            print(f"Warning: '{case_n}' data has 0 variance")
    
    if plot_mean: sns.kdeplot(data=np.concatenate(list(data.values())), label='mean', color='k')
    ax.set_xlabel(var_name)
    ax.legend()
    if xlims: ax.set_xlim(xlims)
    plt.title(tit if tit else f'Kernel Density Estimation (KDE) | variable: {var_name}')
    return ax, f

   
def variable_hist(data, var_name, tit = None, bins = 30, kdeplot = False, st_anlss=None):
    '''
    This function generates a historgram given a set of data.
    E.g.(1): variable_hist(data, 'Temperature', kdeplot = True, bins=10)
    ----------
    data: [df[column_name] / np array]
    var_name: [str] | variable name
    tit: [str] | title name
    bins: [int] | number bins | Def. arg.: 30
    kdeplot: [bool]  | don't visualize / visualize kdeplot | Def. arg.: False
    st_anlss: [str]  | display the statistical variables | Def. arg.: None
    '''
    f, ax = plt.subplots(figsize=(8,8),facecolor='.85')
    sns.histplot(data, bins=bins, kde=False, ax=ax, color="lightgray", edgecolor="gray", alpha=0.7, stat="density")
    if kdeplot and data.var() != 0 : sns.kdeplot(data, color="blue", ax=ax)
    if st_anlss:
        props = dict(boxstyle='round', facecolor='blue', alpha=0.1)
        ax.text(0.7, 0.95, st_anlss, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
    ax.set_xlabel(var_name)
    plt.title(tit if tit else f'Data distribution | variable: {var_name}')


def vis_percent_of_data_outside_lims(data, var, lims, tit, nom_val = None, ylims=None, supt_fonts = 20):
    if not isinstance(lims[0], list): lims = [lims]
    ax1, ax2, f = plot_data(data, var, tit, supt_fonts=supt_fonts)
    for lim in lims:
        ax1.axhspan(lim[0], lim[1], alpha=.15, color='royalblue')
        ax2.axvspan(lim[0], lim[1], alpha=.15, color='royalblue')
    if nom_val:
        ax1.axhline(nom_val, color='k', linestyle='--')
        ax2.axvline(nom_val, color='k', linestyle='--',label=f'Nom. val.: {nom_val}')
        ax2.legend(fontsize=12)
    if ylims: ax1.set_ylim(ylims)
    return f

def get_percent_of_data_outside_lims(data, var, lims):
    percent = 0
    if not isinstance(lims[0], list): lims = [lims]
    data = data[var].dropna()
    for lim in lims:
        percent += len(data[(data >= lim[0]) & (data <= lim[1])]) / len(data)
    return 100-percent* 100


