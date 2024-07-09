import json
import pandas as pd
from datetime import datetime
import re
import modules.ongoing_mod as ongm
import modules.visualization_tools as vist
import modules.utils as utls
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def get_OFinfo_op_data_dict(json_f):
    settings_by_OF = {}
    op_data_dict = {}
    with open(f'{json_f}.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)[0]
    ### get OFinfo
    for order_d in json_data['orders']:
        OF = order_d['id']
        if isinstance(list(order_d['settings'].values())[0], dict):
            setting_dict = {}
            setting_dict['start_timestamp'] = datetime.strptime(order_d['start_timestamp'], '%d/%m/%Y %H:%M:%S')
            setting_dict['end_timestamp'] = datetime.strptime(order_d['end_timestamp'], '%d/%m/%Y %H:%M:%S')
            setting_dict['duration'] = setting_dict['end_timestamp'] - setting_dict['start_timestamp']
            setting_dict['colada'] = [diferentes_colada['id'] for diferentes_colada in order_d['inputs']]

            for setting in order_d['settings']:
                value = order_d['settings'][setting]['value']
                if value is not None and not re.search('^[0]{3}|[a-zA-Z]', str(value)):
                    value = float(value)
                setting_dict[setting] = value
            settings_by_OF[OF] = setting_dict
        else:
            print(f"OF sin diccionario dentro: {OF}")

        OFinfo = pd.DataFrame.from_dict(settings_by_OF, orient='index')
        OFinfo.sort_values(by='start_timestamp', inplace=True)
        OFinfo = OFinfo.rename(columns={'Plnbez': 'TSS'})
        OFinfo['tube_name'] = OFinfo['Wrkst'] + ' ' + OFinfo['Diametro'].astype(str)+ '*' +\
                            OFinfo['Espesor'].astype(str) + ' ' + OFinfo['Longitud'].astype(str)
        ############################################################
        ### get op_data_dict
        
        data = pd.DataFrame(order_d['operational_data'])

        data['DATESTAMP'] =data['DATESTAMP'].apply(lambda date_string: datetime.fromisoformat(date_string[:19]))
        data['frecuencia'] = data['DATESTAMP'].shift(-1)-data['DATESTAMP']
        data['frecuencia'] = data['frecuencia'].apply(lambda x: x.total_seconds())
        data['week'] = data['DATESTAMP'].dt.isocalendar().week
        data['dayofweek'] = data['DATESTAMP'].dt.dayofweek
        data['DATESTAMP'] = data['DATESTAMP'] + pd.Timedelta(hours=1)  # Adelanto de tiempo
        data['ZOE5__potencia'] = data['ZOE5__potencia'].apply(lambda x: 0.77 * x - 0.57) # correción digital de la potencia
        op_data_dict[OF] = data
    ############################################################
    ### get emails
    emails_list = json_data['params']['distribution_emails']
    ############################################################

    return OFinfo, op_data_dict, emails_list

def get_OFinfo_given_json(json_data, print_info=False):
    with open(f'{json_data}.json', 'r', encoding='utf-8') as file:
        data_d = json.load(file)
    settings_by_OF = {}
    Ofs_sin_settings = 0
    Ofs_sin_diccionarios_dentro = 0
    
    inputs_list = data_d['inputs']
    dict_order_id = defaultdict(list) # Crear un defaultdict con listas como valores predeterminados
    for elemento in inputs_list:
        dict_order_id[elemento['order_id']].append(elemento['id'])
    dict_order_id = dict(dict_order_id)
    for order in data_d['orders']:
        if order['settings'] != None:
            if isinstance(list(order['settings'].values())[0], dict):
                OF = order['id']
                setting_dict = {}
                setting_dict['start_timestamp'] = datetime.strptime(order['start_timestamp'], '%d/%m/%Y %H:%M:%S')
                setting_dict['end_timestamp'] = datetime.strptime(order['end_timestamp'], '%d/%m/%Y %H:%M:%S')
                setting_dict['duration'] = setting_dict['end_timestamp'] - setting_dict['start_timestamp']
                setting_dict['colada'] = dict_order_id.get(OF)
                
                for setting in order['settings']:
                    value = order['settings'][setting]['value']
                    if value is not None and not re.search('^[0]{3}|[a-zA-Z]', str(value)):
                        value = float(value)
                    setting_dict[setting] = value
                settings_by_OF[OF] = setting_dict
            else:
                Ofs_sin_diccionarios_dentro += 1
                if print_info:
                    print(f"OF sin diccionario dentro: {order['id']}")
        else:
            Ofs_sin_settings += 1
            if print_info: print(f"OF settings: {order['id']}")
    if print_info:
        print(10*'---')
        print(f'Numero de OFs sin settings: {Ofs_sin_settings}')
        print(f'Numero de OFs sin diccionario dentro: {Ofs_sin_diccionarios_dentro}')
    print(f'OF analizas: {len(settings_by_OF)}')
    df = pd.DataFrame.from_dict(settings_by_OF, orient='index')
    df.sort_values(by='start_timestamp', inplace=True)
    df = df.rename(columns={'Plnbez': 'TSS'})
    df['tube_name'] = df['Wrkst'] + ' ' + df['Diametro'].astype(str)+ '*' +\
                        df['Espesor'].astype(str) + ' ' + df['Longitud'].astype(str)
    
    return df

def sep_dict_with_given_str(inp_dict, sep = ' | '):
    '''
    Separate a dictionary with a given str
    ----------
    inp_dict: [dict]
    sep: [str] | Def. arg.: ' | '
    ----------
    [str]
    '''
    return sep.join(f'{key}: {value}' for key, value in inp_dict.items())

def introduce_line_breaks_in_str(string, max_chars=60):
    '''
    Introduce line breaks given str divided by '|'
    ----------
    string: [str]
    max_chars: [int] | Def. arg.: 60
    ----------
    [str]
    '''
    max_chars = max_chars
    lines = []
    current_line = ''
    for pair in string.split(' | '):
        if len(current_line) + len(pair) + 3 <= max_chars:  # 3 for ': ' and ' | '
            if current_line:
                current_line += ' | ' + pair
            else:
                current_line = pair
        else:
            lines.append(current_line)
            current_line = pair
    if current_line:
        lines.append(current_line)
    return  ' | \n'.join(lines)

def vis_stts_of_a_OF_var(data, OF, OFinfo, var, vrs_dict):
    results = {}
    results['OF'] = OF
    results['TSS'] = get_val_given_OF_and_var(OFinfo, OF, 'TSS')
    zoe_var = vrs_dict[var][0]
    if not re.search('^P_', var):
        tols = vrs_dict[var][1]
        Bnds = vrs_dict[var][2]
        results['nom_val'] = get_val_given_OF_and_var(OFinfo, OF, var)
        lims = [results['nom_val'] + tol for tol in tols]
    else:
        P1 = vrs_dict[var][1]
        P2 = vrs_dict[var][2]
        l1 = get_val_given_OF_and_var(OFinfo, OF, P1)
        l2 = get_val_given_OF_and_var(OFinfo, OF, P2)
        results['nom_val'] = (l2-l1)*0.5+l1
        lims = [l1, l2]
        Bnds = vrs_dict[var][3]
        tols = [results['nom_val']-l1, l2-results['nom_val']]
        
    results['tols'] = str(tols)
    results['lims'] = str(lims)
    OF_start_timestamp = get_val_given_OF_and_var(OFinfo, OF, 'start_timestamp')
    OF_end_timestamp = get_val_given_OF_and_var(OFinfo, OF, 'end_timestamp')
    mod_d = ongm.filt_data_by_start_end_time(data, OF_start_timestamp, OF_end_timestamp)
    mod_d =  ongm.get_data_filter_between_vals(mod_d, zoe_var, Bnds) 
    if not mod_d.empty and not mod_d[zoe_var].isna().all(): 
        results['%_fuera'] = round(vist.get_percent_of_data_outside_lims(mod_d, zoe_var, lims),2)
        if not re.search('^P_', var):
            results['dist'] = round((mod_d[zoe_var].mean()-results['nom_val'])/tols[1],2)
            cp = utls.get_cp(mod_d[zoe_var], results['nom_val']+tols[0], results['nom_val']+tols[1])
            results['cp'] = cp   
        else:        
            results['dist'] = round((mod_d[zoe_var].mean()-results['nom_val'])/((l2-l1)*0.5),2)
            cp = utls.get_cp(mod_d[zoe_var], l1, l2)
            results['cp'] = cp 
            
        results['std'] = round(mod_d[zoe_var].std(),2)
        results_str = sep_dict_with_given_str(results)
        tit = introduce_line_breaks_in_str(results_str, max_chars=80)
        f = vist.vis_percent_of_data_outside_lims(mod_d, zoe_var, lims, tit, nom_val = results['nom_val'], supt_fonts = 14)
        plt.show()
        return f, results
    else: return None, None










def vis_var_of_OFs_asigned_to_TSS(TSS, var, OFinfo, data, vrs_dict):
    OFs = OFinfo[OFinfo['TSS'] == TSS].index
    results = {}
    zoe_var = vrs_dict[var][0]
    d = {}
    if not re.search('^P_', var):
        tols = vrs_dict[var][1]
        Bnds = vrs_dict[var][2]
        results['nom_val'] = get_val_given_OF_and_var(OFinfo, OFs[0], var)
        lims = [results['nom_val'] + tol for tol in tols]
    else:
        P1 = vrs_dict[var][1]
        P2 = vrs_dict[var][2]
        l1 = get_val_given_OF_and_var(OFinfo, OFs[0], P1)
        l2 = get_val_given_OF_and_var(OFinfo, OFs[0], P2)
        results['nom_val'] = (l2-l1)*0.5+l1
        lims = [l1, l2]
        Bnds = vrs_dict[var][3]
        tols = [results['nom_val']-l1, l2-results['nom_val']]

    for OF in OFs:
        OF_start_timestamp = get_val_given_OF_and_var(OFinfo, OF, 'start_timestamp')
        OF_end_timestamp = get_val_given_OF_and_var(OFinfo, OF, 'end_timestamp')
        TSS = get_val_given_OF_and_var(OFinfo, OF, 'TSS')
        mod_d = ongm.filt_data_by_start_end_time(data, OF_start_timestamp, OF_end_timestamp)
        mod_d =  ongm.get_data_filter_between_vals(mod_d, zoe_var, Bnds)  
        if not mod_d.empty and not mod_d[zoe_var].isna().all():  
            OF_st_name = get_OF_st_name(OF, OFinfo)
            d[OF_st_name] = np.array(mod_d[zoe_var].dropna())
            
    ax, f = vist.variable_hists(d, f"Temp: {results['nom_val']} | TSS: {TSS}")
    ax.axvspan(lims[0], lims[1], alpha=.15, color='lightgrey')
    ax.axvline(x=lims[0], color='black', linewidth=.5)
    ax.axvline(x=lims[1], color='black', linewidth=.5)

    ax.axvline(results['nom_val'], color='k', linestyle='--', linewidth=.8)
    plt.show()
    return ax, f

def get_unique_df_given_csvs(csvs_list):
    data_l = []
    for csv in csvs_list:
        d, _ = utls.get_data(f'../data/{csv}_data')
        data_l.append(d)
    return pd.concat(data_l, ignore_index=True)

def get_val_given_OF_and_var(OFinfo, OF, var):
    OF = int(OF) if isinstance(OF, str) else OF
    return OFinfo.loc[f'{OF:012}', var]

def get_OFs_of_cmmn_TSS(data):
    OF_TSS_dict = {}
    OFs_of_cmmn_TSS = {}
    for key, df in data.items():
        OF_TSS_dict[key] = df['Plnbez'].values[0]

    for OF, TSS in OF_TSS_dict.items():
        if TSS not in OFs_of_cmmn_TSS:
            OFs_of_cmmn_TSS[TSS] = [OF]
        else:
            OFs_of_cmmn_TSS[TSS].append(OF)
    return OFs_of_cmmn_TSS

def get_OF_st_name(OF, OFinfo):
    'Get OF457729_231212_1124 given OF and OFinfo'
    stime = get_val_given_OF_and_var(OFinfo, OF, 'start_timestamp')
    stime_str = stime.strftime('%y%m%d_%H%M')
    return f'OF{int(OF):d}_{stime_str}'

def get_n_most_common_TSSs(OFinfo, n=3):
    most_common_TSSs = OFinfo['TSS'].value_counts().head(n)
    most_common_TSSs.name = 'n_times'
    df = pd.DataFrame(most_common_TSSs)
    df['Frequency_[%]'] = (df['n_times']*100 / len(OFinfo['TSS'].unique())).round(2)   
    return df
    
def get_consecutive_OFs(data):
    time_diff = data['start_timestamp'].shift(-1) - data['end_timestamp']
    time_diff = time_diff <= pd.Timedelta(minutes=1)
    grupos = []
    grupo_actual = []
    for k, v in time_diff.items():
        if not v:
            if grupo_actual:
                grupos.append(grupo_actual + [k])
                grupo_actual = []
            else:
                grupos.append([k])
        else:
            grupo_actual.append(k)

    return grupos