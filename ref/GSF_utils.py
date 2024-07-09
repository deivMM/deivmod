import re
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression

# =============================================================================
# modelulo de Garay
# =============================================================================

def get_data(f_name):
    '''
    Function to open a csv file with the manufacturing parameters.
    E.g.: get_data('../data/w48_2023_data')
    ----------
    f_name: [str]
    '''    
    with open(f'{f_name}.csv', 'r') as file:
        first_line = file.readline()
        delimiter = ',' if ',' in first_line else ';'
    data = pd.read_csv(f'{f_name}.csv', sep = delimiter)
    data['DATESTAMP'] =data['DATESTAMP'].apply(lambda date_string: datetime.fromisoformat(date_string[:19]))
    data['frecuencia'] = data['DATESTAMP'].shift(-1)-data['DATESTAMP']
    data['frecuencia'] = data['frecuencia'].apply(lambda x: x.total_seconds())
    data['week'] = data['DATESTAMP'].dt.isocalendar().week
    data['dayofweek'] = data['DATESTAMP'].dt.dayofweek
    w = data['week'].value_counts().idxmax()
    return data, w

# =============================================================================
# filters
# =============================================================================

def filt_df_btwn_days(data, day1, day2):
    '''
    Filter to obtain data between two days
    E.g.: filt_df_btwn_days(data, 0, 4)
    ----------
    data: [df] | df must contain a dayofweek column
    ''' 
    return data[data['dayofweek'].between(day1,day2)]


def filt_remove_holidays(data, holidays):
    '''
    Filter to remove holidays from data
    E.g.: filt_remove_holidays(data, ['2023-07-16', '2023-07-18', '2023-07-20'])
    ----------
    data: [df]
    holidays: [list]
    ''' 
    holidays = pd.to_datetime(holidays)
    return data[~pd.to_datetime(data['DATESTAMP'].dt.date).isin(holidays)]

def filter_btwn_lw_up_by_hours(data, lb=None, ub=None):
    '''
    Filter to  obtain data between two hours / above or below a certain hour
    E.g.: filter_btwn_lw_up_by_hours(data, 8, 17)
    ----------
    data: [df]
    lb: [int] | Def. arg.: None
    ub: [int] | Def. arg.: None
    ''' 
    if lb is not None and ub is not None:
        return data[(data['DATESTAMP'].dt.hour >= lb) & (data['DATESTAMP'].dt.hour <= ub)]
    elif lb:
        return data[data['DATESTAMP'].dt.hour >= lb]
    elif ub:
        return data[data['DATESTAMP'].dt.hour <= ub]

def filter_data(*filters):
    '''
    Function to apply multiple filters
    E.g.: filter_data((filt_df_btwn_days(data, 0, 4), (filter_remove_holidays, data, ['2023-07-19', '2023-07-18']))
    ----------
    data: [df]
    lb: [int] | Def. arg.: None
    ub: [int] | Def. arg.: None
    ''' 
    for filter_func, *args in filters:
        filtered_df = filter_func(args[0], *args[1:])
    return filtered_df

# =============================================================================
# ----
# =============================================================================

def get_cp(data, LL, UL):
    """
    Calculates the capability index for a process.
    ----------
    data: [np array / pd series]
    LL: [float]
    UL: [float]
    ----------
    cp: [float]
    """
    std = data.std()
    if std == 0:
        return 'std=0'
    
    cp = (UL-LL) / (6 * std)
    return round(cp, 2 )


def get_st_anlss(data, list_of_params, var = None):
    '''
    Function to get a dictionary with some statistical indicators.
    E.g.:get_st_anlss(data, ['std', 'max', 'min'])
    ----------
    data: [df/np array]
    list_of_params: [list]
    var: [str] | df variable name
    ----------
    st_anlss: [dict]
    '''     
    st_anlss = {}
    if isinstance(data, list): data = np.array(data)
    if var: data = data[var]
    for param in list_of_params:
        if param == 'mean':
            st_anlss[f'{param.capitalize()}'] = data.mean()
        elif param == 'std':
            st_anlss[f'{param.capitalize()}'] = data.std()
        elif param == 'min':
            st_anlss[f'{param.capitalize()}'] = data.min()
        elif param == 'max':
            st_anlss[f'{param.capitalize()}'] = data.max()
    return st_anlss

def get_str_given_params_val_dict(params_val_dict, rnd  = 2):
    '''
    Function to get a str given a parameter-value dict
    E.g.:get_str_given_params_val_dict(params_val_dict)
    ----------
    params_val_dict: [dict]
    rnd: [int] | Def. arg.: 2
    ----------
    [str]
    '''
    return '\n'.join([f'{key}: {round(value,rnd)}' for key, value in params_val_dict.items()])

def get_random_cont_sample(data, n):
    '''
    Obtain a random n-sample in a data frame
    E.g.:get_random_cont_sample(data, n)
    ----------
    data: [df]
    n: [int]
    ----------
    sample: [df]
    '''
    start_index = np.random.randint(0, len(data)-n)
    sample = data.iloc[start_index:start_index + n]
    return sample

def fil_l_given_regex(l, regex):
    '''
    Filter a list given a regex pattern 
    E.g.: fil_l_given_regex(l, regex)
    ----------
    l: [list]
    regex: [regex pattern ]
    ----------
    [list]
    '''
    return [item for item in l if re.search(regex, item)]

def lin_model(data, var1, var2):
    d = data.dropna()
    model = LinearRegression()
    model.fit(d[var1].values.reshape(-1, 1), d[var2])
    
    y_pred = model.predict(d[var1].values.reshape(-1, 1))

    R_2 = model.score(d[var2].values.reshape(-1, 1), y_pred)
    return model.coef_[0], model.intercept_, R_2

# =============================================================================
# Others
# =============================================================================

def remove_outliers(data, var):
    # IQR (Interquartile Range) Method
    Q1 = data[var].quantile(0.25)
    Q3 = data[var].quantile(0.75)
    IQR = Q3 - Q1
    l_bound = Q1 - 1.5 * IQR
    u_bound = Q3 + 1.5 * IQR
    return data[data[var].between(l_bound, u_bound)], l_bound, u_bound

def remove_outliers_IQR_meth(data):
    if isinstance(data, list): data = np.array(data)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    l_bound = Q1 - 1.5 * IQR
    u_bound = Q3 + 1.5 * IQR
    
    return data[(data >= l_bound) & (data <= u_bound)]

def remove_outliers_zscore(data, threshold=3):
    if isinstance(data, list): data = np.array(data)
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    return data[z_scores < threshold]

def find_consecutive_true_indexes(series):
    groups = (series != series.shift()).cumsum() # Compare the first value with the second value and sum the True values cumulatively. 
    result = []

    for group_id, group_data in series.groupby(groups):
        if group_data.iloc[0]:  # Check if it's a group of True values
            start_index = group_data.index[0]
            end_index = group_data.index[-1]
            result.append((start_index, end_index))

    return result

def get_indxs_btwn_upper_lower(data, var, cond, lb=None, ub=None):
    if cond == 'b_filt' and lb is not None and ub is not None:
        df = data[var].between(lb, ub)
    elif cond == 'l_filt' and lb is not None:
        df = data[var]<lb
    elif cond == 'u_filt' and ub is not None:
        df = data[var]>ub
    return find_consecutive_true_indexes(df)

def get_time_btwn_index(data, indx1=None, indx2=None):
    if not indx1 and not indx2:
        indx1 = -1
        indx2 = 0
    return (data.iloc[indx1] - data.iloc[indx2]).total_seconds()/60

def get_porcetaje_t_apagado(data, indxs):
    total_t = get_time_btwn_index(data['DATESTAMP'])
    total_t_apagado = 0
    for indx in indxs:
        d = data.loc[indx[0]:indx[1]]
        total_t_apagado += get_time_btwn_index(d['DATESTAMP'])
    porcetaje_t_apagado = total_t_apagado/total_t
    print(f'Porcentaje de t apagado: {porcetaje_t_apagado:.2%}')
    return porcetaje_t_apagado