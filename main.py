import os
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from tqdm import tqdm
from difflib import SequenceMatcher
import jellyfish
from pprint import pprint

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def convert_number(number, number_type):
    output = math.nan
    try:
        output = number_type(number)
    except ValueError:
        if number == '':
            output = math.nan
        else:
            raise
    return output
    
def find_files(path, ext='.csv'):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if ext in file:
                files.append(os.path.join(r, file))
    return files

def read_bvd_files(files):
    orbis = []
    for file in files:
        print(file)
        with open(file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                data = line.split(';')
                if len(data) >= 26:
                    orbis.append(data)
    return orbis

def read_Manu_file(file):
    print(file)
    data_out = []
    with open(file, 'r', encoding='utf-8') as f:
        i = 0
        for line in tqdm(f):
            if i < 1:
                i += 1
                continue
            else:
                data = line.split(';')
                data = [entry.strip() for entry in data]
                while True:
                    quote_fix = False
                    for i in range(len(data)-1):
                        for j in range(i+1,len(data)):
                            if data[i] and data[j]:
                                if data[i][0] == '"' and data[j][-1] == '"':
                                    quote_fix = True
                                    data[i] = '; '.join(data[i:j+1])
                                    data[i] = data[i][1:-1]
                                    for r in data[i+1:j+1]:
                                        data.remove(r)
                                    break
                    if not quote_fix:
                        break
                data_out.append(data)
    return data_out

def read_rda_file(file, country):
    print(file)
    data_out = []
    if country == 'hr':
        with open(file, 'r', encoding='utf-8') as f:
            i = 0
            for line in tqdm(f):
                if i < 1:
                    i += 1
                    continue
                else:
                    data = line.split(';')
                    data = [entry.strip() for entry in data]
                    data_out.append(data)
    elif country == 'bg':
        with open(file, 'r', encoding='utf-8') as f:
            i = 0
            for line in tqdm(f):
                if i < 1:
                    i += 1
                    continue
                else:
                    data = line.split(';')
                    data = [entry.strip() for entry in data]
                    while True:
                        quote_fix = False
                        for i in range(len(data)-1):
                            for j in range(i+1,len(data)):
                                if data[i] and data[j]:
                                    if data[i][0] == '"' and data[j][-1] == '"':
                                        quote_fix = True
                                        data[i] = '; '.join(data[i:j+1])
                                        data[i] = data[i][1:-1]
                                        for r in data[i+1:j+1]:
                                            data.remove(r)
                                        break
                        if not quote_fix:
                            break
                    data_out.append(data)
    return data_out

def parse_orbis(table_in):
    orbis_dicts = []
    for entry in tqdm(table_in):
        entry_dict = {
            'name': entry[1], 
            'ID': entry[2], 
            'country': entry[3], 
            'NACE': entry[4], 
            'turnover': [],
            'employment' :[],
            'years_t': [],
            'years_e': []
        }

        for e in entry[16:26]:
            e = ''.join(e.split())
            e = '.'.join(e.split(','))
            try:
                float(e)
            except ValueError:
                entry_dict['turnover'].append(math.nan)
            else:
                break

        had_value = False
        for e in entry[6:16]:
            e = ''.join(e.split())
            e = '.'.join(e.split(','))
            try:
                e = int(float(e)*1000)
            except ValueError:
                if had_value:
                    entry_dict['turnover'].append(math.nan)
            else:
                had_value = True
                entry_dict['turnover'].append(e)

        if len(entry) > 26:
            for e in entry[36:46]:
                e = ''.join(e.split())
                try:
                    int(e)
                except ValueError:
                    entry_dict['employment'].append(math.nan)
                else:
                    break

            had_value = False
            for e in entry[26:36]:
                e = ''.join(e.split())
                try:
                    e = int(e)
                except ValueError:
                    if had_value:
                        entry_dict['employment'].append(math.nan)
                else:
                    had_value = True
                    entry_dict['employment'].append(e)

        all_turnover_nan = True
        for t in entry_dict['turnover']:
            if not math.isnan(t):
                all_turnover_nan = False
                break
                    
        all_employment_nan = True
        for e in entry_dict['employment']:
            if not math.isnan(e):
                all_employment_nan = False
                break

        if all_turnover_nan and all_employment_nan:
            continue

        for entry_check in orbis_dicts:
            if entry_dict['ID'] == entry_check['ID']:
                if not entry_check['NACE']:
                    entry_check['NACE'] = entry_dict['NACE']
                if not entry_check['turnover']:
                    entry_check['turnover'] = entry_dict['turnover']
                if not entry_check['employment']:
                    entry_check['employment'] = entry_dict['employment']
                break
        else:
            orbis_dicts.append(entry_dict)
        
        for entry_year_check in orbis_dicts:
            entry_year_check['years_t'] = list(range(2019, 2019-len(entry_year_check['turnover']), -1))
            entry_year_check['years_e'] = list(range(2019, 2019-len(entry_year_check['employment']), -1))
    
    return orbis_dicts

def parse_Manu(table_in, country, set_type):
    Manu_dicts = []
    for entry in tqdm(table_in):
        if set_type == 'lkn':
            entry_dict = {
                'name': entry[0].strip(), 
                'news': convert_number(entry[1],int), 
                'conference': convert_number(entry[2],int), 
                'publication': convert_number(entry[3],int), 
                'pubmed': convert_number(entry[4],int),
                'patent': convert_number(entry[5],int),
                'trademark': convert_number(entry[6],int),
                'grant': convert_number(entry[7],int),
                'web': convert_number(entry[8],int)
            }
        else:
            if country == 'bg':
                entry_dict = {
                    'name': entry[0].strip(), 
                    'website': entry[1].strip(), 
                    'country': entry[2].strip(), 
                    'city': entry[3].strip(), 
                    'generic NACE': entry[4].strip(),
                    'specific NACE': entry[5].strip(),
                    'turnover': [],
                    'employment': [],
                    'years': list(range(2011,2017)),
                    'grants': []
                }
                for i, e in enumerate(entry[6:16]):
                    if i%2 == 0:
                        entry_dict['turnover'].append(convert_number(e,float))
                    else:
                        entry_dict['employment'].append(convert_number(e,int))

                for i, e in enumerate(entry[16:]):
                    if i%3 == 0:
                        entry_dict['grants'].append([convert_number(e,int)])
                    elif i%3 > 0:
                        entry_dict['grants'][-1].append(e)
            elif country == 'de':
                try:
                    entry_dict = {
                        'name': entry[0].strip(), 
                        'website': entry[1].strip(), 
                        'country': entry[2].strip(), 
                        'city': entry[3].strip(),
                        'turnover': [],
                        'employment': [],
                        'years': list(range(2015,2018)),
                        'grants': []
                    }
                except IndexError:
                    continue

                for i, e in enumerate(entry[4:10]):
                    if i%2 == 0:
                        entry_dict['turnover'].append(convert_number(e,float))
                    else:
                        entry_dict['employment'].append(convert_number(e,int))
                
                for i, e in enumerate(entry[10:]):
                    if i%3 == 0:
                        entry_dict['grants'].append([convert_number(e,int)])
                    elif i%3 > 0:
                        entry_dict['grants'][-1].append(e)
            elif country == 'es':
                entry_dict = {
                    'name': entry[0].strip(), 
                    'website': entry[1].strip(), 
                    'country': entry[2].strip(), 
                    'city': entry[3].strip(),
                    'turnover': [],
                    'employment': [],
                    'years': list(range(2015,2018)),
                    'grants': []
                }

                for i, e in enumerate(entry[4:10]):
                    if i%2 == 0:
                        entry_dict['turnover'].append(convert_number(e,float))
                    else:
                        entry_dict['employment'].append(convert_number(e,int))
                        
                for i, e in enumerate(entry[10:]):
                    if i%3 == 0:
                        entry_dict['grants'].append([convert_number(e,int)])
                    elif i%3 > 0:
                        entry_dict['grants'][-1].append(e)
            elif country == 'hr':
                entry_dict = {
                    'name': entry[0].strip(), 
                    'website': entry[1].strip(), 
                    'country': entry[2].strip(), 
                    'city': entry[3].strip(),
                    'turnover': [],
                    'employment': [],
                    'years': list(range(2013,2019)),
                    'grants': []
                }
                
                for i, e in enumerate(entry[4:]):
                    if i%2 == 0:
                        entry_dict['turnover'].append(convert_number(e,float))
                    else:
                        entry_dict['employment'].append(convert_number(e,int))
            elif country == 'tr_bebka':
                entry_dict = {
                    'name': entry[0].strip(), 
                    'website': entry[1].strip(), 
                    'country': entry[2].strip(), 
                    'city': entry[3].strip(),
                    'turnover': [],
                    'employment': [],
                    'years': [],
                    'grants': []
                }
                
                for i, e in enumerate(entry[6:]):
                    if i%3 == 0:
                        entry_dict['grants'].append([convert_number(e,int)])
                    elif i%3 > 0:
                        entry_dict['grants'][-1].append(e)
            elif country == 'tr_marka':
                entry_dict = {
                    'name': entry[0].strip(), 
                    'website': entry[1].strip(), 
                    'country': entry[2].strip(), 
                    'city': entry[3].strip(),
                    'turnover': [],
                    'employment': [],
                    'years': list(range(2011,2019)),
                    'grants': []
                }

                for i, e in enumerate(entry[6:20]):
                    if i%2 == 0:
                        entry_dict['turnover'].append(convert_number(e,float))
                    else:
                        entry_dict['employment'].append(convert_number(e,int))
                        
                for i, e in enumerate(entry[20:]):
                    if i%3 == 0:
                        entry_dict['grants'].append([convert_number(e,int)])
                    elif i%3 > 0:
                        entry_dict['grants'][-1].append(e)

            elif country == 'it':
                entry_dict = {
                    'name': '', 
                    'website': '', 
                    'country': '', 
                    'city': '', 
                    'grants': []
                }

        Manu_dicts.append(entry_dict)
    
    return Manu_dicts


def parse_rda(table_in, country):
    table_out = []
    
    if country == 'hr':
        for year in table_in:
            for entry in tqdm(table_in[year]):
                if len(entry)>1 and entry[1] != '':
                    if table_out:
                        for company in table_out:
                            if company['name'] == entry[1]:
                                company['help_years'].append(convert_number(year,int))
                                break
                        else:
                            table_out.append({'name': entry[1], 'help_years': [convert_number(year,int)]})
                    else:
                        table_out.append({'name': entry[1], 'help_years': [convert_number(year,int)]})
    elif country == 'bg':
        for year in table_in:
            for entry in tqdm(table_in[year]):
                try:
                    employment = bg_employment_translation[entry[13]]
                except KeyError:
                    print(year)
                    raise
                turnover = bg_turnover_translation[entry[15]]*bg_eur[str(year)]
                pl = bg_pl_translation[entry[17]]*bg_eur[str(year)]
                assets = bg_assets_translation[entry[19]]*bg_eur[str(year)]
                #capital = bg_capital_translation[entry[21]]*bg_eur[str(year)]
                
                if table_out:
                    for company in table_out:
                        if company['id']  == entry[0]:
                            company['employment'].append(employment)
                            company['turnover'].append(turnover)
                            company['profit/loss'].append(pl)
                            company['tangible assets'].append(assets)
                            #company['capital'].append(capital)
                            company['help_years'].append(year)
                            break
                    else:
                        table_out.append({'id': entry[0],
                                          'name': entry[1],
                                          'employment': [employment],
                                          'turnover':[turnover],
                                          'profit/loss':[pl],
                                          'tangible assets':[assets],
                                          #'capital':[capital],
                                          'help_years':[year]})
                else:
                    table_out.append({'id': entry[0],
                                      'name': entry[1],
                                      'employment': [employment],
                                      'turnover':[turnover],
                                      'profit/loss':[pl],
                                      'tangible assets':[assets],
                                      #'capital':[capital],
                                      'help_years':[year]})
    return table_out

def correlate_names(table_in, country):
    names_nodups = []
    names_table = []
    
    country_big = country.upper()
    
    for orbis_entry in all_data['orbis_dicts']:
        if orbis_entry['country'] == country_big:
            for in_entry in table_in:
                score_L = jellyfish.levenshtein_distance(orbis_entry['name'], in_entry['name'].upper())
                if score_L <= 10:
                    names_table.append([score_L, orbis_entry, in_entry])
    
    names_found = []
    for names in names_table:
        score_SM = SequenceMatcher(None, names[1]['name'], names[2]['name'].upper()).ratio()
        if (1-score_SM)*names[0] <= 3:
            names_found.append([(1-score_SM)*names[0], names[1], names[2]])
            
    names_found.sort(key=lambda name: name[0])
    for name in names_found:
        if not names_nodups:
            names_nodups.append(name)
        else:
            for name_nodup in names_nodups:
                if name[1]['name'] == name_nodup[1]['name']:
                    break
            else:
                names_nodups.append(name)
    return names_nodups
    
def save_namelist(table_in, filename):
    with open(filename,'w',encoding='utf-8') as f:
        table_in.sort(key=lambda name: name[0])
        for name in tqdm(table_in):
            line = '\t'.join([str(name[0]), name[1]['name'], name[2]['name']])
            line = line+'\n'
            f.write(line)
    return 0

def read_names_correlation(file_in, table_out):
    with open(file_in, 'r', encoding='utf-8') as f:
        for line in f:
            line_split = line.split('\t')
            table_out.append([line_split[1].strip(), line_split[2].strip()])
    return 0

def connect_datasources(table_in, country):
    manu_found = []
    rda_found = []
    
    for entry in tqdm(table_in):
        entry['rda_help'] = []
        entry['grants'] = []
        entry['lkn'] = dict()
        names_found = []
        
        if country in all_data['names_correlation']['Manu'].keys():
            for name in all_data['names_correlation']['Manu'][country]:
                if name[0] == entry['name']:
                    names_found = name
                    manu_found.append(name[1])
                    break
            if names_found:
                for manu in all_data['Manu_dicts']['grants_'+country]:
                    if manu['name'] == names_found[1]:
                        for grant in manu['grants']:
                            if grant[0]:
                                entry['grants'].append(grant[0])
                        for year in manu['years']:
                            if year not in entry['years_t']:
                                pass #TODO
                            if year not in entry['years_e']:
                                pass #TODO
                        break
                for manu_lkn in all_data['Manu_dicts']['lkn_'+country]:
                    if manu_lkn['name'] == names_found[1]:
                        entry['lkn'] = manu_lkn
                    
        if country in all_data['names_correlation']['rda'].keys():
            for name in all_data['names_correlation']['rda'][country]:
                if name[0] == entry['name']:
                    names_found = name
                    rda_found.append(name[1])
                    break
            if names_found:
                for rda in all_data['rda_dicts'][country]:
                    if rda['name'] == names_found[1]:
                        for year in rda['help_years']:
                            if year:
                                entry['rda_help'].append(int(year))
                        break

    manu_entries = []
    if 'grants_'+country in all_data['Manu_dicts'].keys():
        for manu_entry in all_data['Manu_dicts']['grants_'+country]:
            if manu_entry['name'] not in manu_found:
                dict_entry = {
                    'name': manu_entry['name'],
                    'ID': '',
                    'country': country.upper(),
                    'NACE': '',
                    'turnover': manu_entry['turnover'],
                    'employment': manu_entry['employment'],
                    'years_t': manu_entry['years'],
                    'years_e': manu_entry['years'],
                    'grants': []
                }
                for grant in manu_entry['grants']:
                    if grant[0]:
                        dict_entry['grants'].append(grant[0])
                manu_entries.append(dict_entry)

    if 'lkn_'+country in all_data['Manu_dicts'].keys():
        for lkn_entry in all_data['Manu_dicts']['lkn_'+country]:
            if lkn_entry['name'] not in manu_found:
                for manu_entry in manu_entries:
                    if manu_entry['name'] == lkn_entry['name']:
                        manu_entry['lkn'] = lkn_entry
                        break
                else:
                    dict_entry = {
                        'name': lkn_entry['name'],
                        'ID': '',
                        'country': country.upper(),
                        'NACE': '',
                        'turnover': [],
                        'employment': [],
                        'years_t': [],
                        'years_e': [],
                        'grants': [],
                        'lkn': lkn_entry
                    }
                    manu_entries.append(dict_entry)
    table_in += manu_entries
    
    rda_entries = []
    if country in all_data['rda_dicts'].keys():
        for rda_entry in all_data['rda_dicts'][country]:
            if rda_entry['name'] not in rda_found:
                dict_entry = {
                    'name': rda_entry['name'],
                    'ID': '',
                    'country': country.upper(),
                    'NACE': '',
                    'turnover': [],
                    'employment': [],
                    'years_t': [],
                    'years_e': [],
                    'grants': [],
                    'rda_help': []
                }
                for year in rda_entry['help_years']:
                    if year:
                        dict_entry['rda_help'].append(year)
                rda_entries.append(dict_entry)
    table_in += rda_entries
    return 0

def get_smes(dictionary_in):
    dictionary_out = []
    for entry in dictionary_in:
        sme = True
        for t in entry['turnover']:
            if t >= 50000000:
                sme = False
                break
        else:
            for e in entry['employment']:
                if e >= 250:
                    sme = False
                    break
        
        if sme:
            dictionary_out.append(entry)
    return dictionary_out

def calc_productivity(dictionary_in):
    for entry in tqdm(dictionary_in):
        len_t = len(entry['turnover'])
        len_e = len(entry['employment'])
        productivity = []
        if len_t >= len_e:
            for i in range(len_e):
                if entry['employment'][i] == 0:
                    productivity.append(math.nan)
                else:
                    productivity.append(entry['turnover'][i]/entry['employment'][i])
            for i in range(len_e, len_t):
                productivity.append(math.nan)
        else:
            for i in range(len_t):
                if entry['employment'][i] == 0:
                    productivity.append(math.nan)
                else:
                    productivity.append(entry['turnover'][i]/entry['employment'][i])
            for i in range(len_t, len_e):
                productivity.append(math.nan)
        entry['productivity'] = productivity
    return 0

def calc_fit(dictionary_in, dictionary_out):
    for entry in tqdm(dictionary_in):

        all_values_nan = True
        negative_value_exists = True
        for value in entry['turnover']:
            if not math.isnan(value):
                all_values_nan = False
                if value == 0:
                    value = math.nan
                if not value < 0:
                    negative_value_exists = False

        if (not all_values_nan) and (not negative_value_exists):
            y = np.array(entry['turnover'])

            t0 = np.array(range(2019,2019-len(y),-1))
            t = sm.add_constant(t0, prepend=False)

            model = sm.OLS(y,t,missing='drop')
            n_observations_t = len(model.exog)
            if n_observations_t >= 5:
                result = model.fit()

                params_t = result.params #x, const
                errors_t = result.bse #x, const
                rsquared_t = result.rsquared
                errors_relative_t = [0,0]

                if params_t[0] != 0.0:
                    errors_relative_t[0] = abs(errors_t[0]/params_t[0]*100)
                if params_t[1] != 0.0:
                    errors_relative_t[1] = abs(errors_t[1]/params_t[1]*100)
                    
                t2 = np.column_stack((t0**2, t0, np.ones(len(t0))))
                model2 = sm.OLS(y,t2,missing='drop')
                result2 = model2.fit()
                
                params_t2 = result2.params #x, const
                errors_t2 = result2.bse #x, const
                rsquared_t2 = result2.rsquared
                errors_relative_t2 = [0,0,0]

                if params_t2[0] != 0.0:
                    errors_relative_t2[0] = abs(errors_t2[0]/params_t2[0]*100)
                if params_t2[1] != 0.0:
                    errors_relative_t2[1] = abs(errors_t2[1]/params_t2[1]*100)
                if params_t2[2] != 0.0:
                    errors_relative_t2[2] = abs(errors_t2[2]/params_t2[2]*100)
            else:
                params_t = [math.nan, math.nan]
                errors_t = [math.nan, math.nan]
                rsquared_t = math.nan
                errors_relative_t = [math.nan, math.nan]
                n_observations_t = 0
                params_t2 = [math.nan, math.nan, math.nan]
                errors_t2 = [math.nan, math.nan, math.nan]
                rsquared_t2 = math.nan
                errors_relative_t2 = [math.nan, math.nan, math.nan]
        else:
            params_t = [math.nan, math.nan]
            errors_t = [math.nan, math.nan]
            rsquared_t = math.nan
            errors_relative_t = [math.nan, math.nan]
            n_observations_t = 0
            params_t2 = [math.nan, math.nan, math.nan]
            errors_t2 = [math.nan, math.nan, math.nan]
            rsquared_t2 = math.nan
            errors_relative_t2 = [math.nan, math.nan, math.nan]


        all_values_nan = True
        negative_value_exists = True
        for value in entry['employment']:
            if not math.isnan(value):
                all_values_nan = False
                if value == 0:
                    value = math.nan
                if not value < 0:
                    negative_value_exists = False

        if (not all_values_nan) and (not negative_value_exists):
            y = np.array(entry['employment'])

            t0 = np.array(range(2019,2019-len(y),-1))
            t = sm.add_constant(t0, prepend=False)

            model = sm.OLS(y,t,missing='drop')
            n_observations_e = len(model.exog)
            if n_observations_e >= 5:
                result = model.fit()

                params_e = result.params #x, const
                errors_e = result.bse #x, const
                rsquared_e = result.rsquared
                errors_relative_e = [0,0]

                if params_e[0] != 0.0:
                    errors_relative_e[0] = abs(errors_e[0]/params_e[0]*100)
                if params_e[1] != 0.0:
                    errors_relative_e[1] = abs(errors_e[1]/params_e[1]*100)
                
                t2 = np.column_stack((t0**2, t0, np.ones(len(t0))))
                model2 = sm.OLS(y,t2,missing='drop')
                result2 = model2.fit()
                
                params_e2 = result2.params #x, const
                errors_e2 = result2.bse #x, const
                rsquared_e2 = result2.rsquared
                errors_relative_e2 = [0,0,0]

                if params_e2[0] != 0.0:
                    errors_relative_e2[0] = abs(errors_e2[0]/params_e2[0]*100)
                if params_e2[1] != 0.0:
                    errors_relative_e2[1] = abs(errors_e2[1]/params_e2[1]*100)
                if params_e2[2] != 0.0:
                    errors_relative_e2[2] = abs(errors_e2[2]/params_e2[2]*100)
            else:
                params_e = [math.nan, math.nan]
                errors_e = [math.nan, math.nan]
                rsquared_e = math.nan
                errors_relative_e = [math.nan, math.nan]
                n_observations_e = 0
                params_e2 = [math.nan, math.nan, math.nan]
                errors_e2 = [math.nan, math.nan, math.nan]
                rsquared_e2 = math.nan
                errors_relative_e2 = [math.nan, math.nan, math.nan]
        else:
            params_e = [math.nan, math.nan]
            errors_e = [math.nan, math.nan]
            rsquared_e = math.nan
            errors_relative_e = [math.nan, math.nan]
            n_observations_e = 0
            params_e2 = [math.nan, math.nan, math.nan]
            errors_e2 = [math.nan, math.nan, math.nan]
            rsquared_e2 = math.nan
            errors_relative_e2 = [math.nan, math.nan, math.nan]
        
        
        all_values_nan = True
        all_values_zeroes = True
        for value in entry['productivity']:
            if not math.isnan(value):
                all_values_nan = False
                if not value == 0:
                    all_values_zeroes = False

        if (not all_values_nan) and (not all_values_zeroes):
            y = np.array(entry['productivity'])

            t0 = np.array(range(2019,2019-len(y),-1))
            t = sm.add_constant(t0, prepend=False)

            model = sm.OLS(y,t,missing='drop')
            n_observations_p = len(model.exog)
            if n_observations_p >= 5:
                result = model.fit()

                params_p = result.params #x, const
                errors_p = result.bse #x, const
                rsquared_p = result.rsquared
                errors_relative_p = [0,0]

                if params_p[0] != 0.0:
                    errors_relative_p[0] = abs(errors_p[0]/params_p[0]*100)
                if params_p[1] != 0.0:
                    errors_relative_p[1] = abs(errors_p[1]/params_p[1]*100)
                
                t2 = np.column_stack((t0**2, t0, np.ones(len(t0))))
                model2 = sm.OLS(y,t2,missing='drop')
                result2 = model2.fit()
                
                params_p2 = result2.params #x, const
                errors_p2 = result2.bse #x, const
                rsquared_p2 = result2.rsquared
                errors_relative_p2 = [0,0,0]

                if params_p2[0] != 0.0:
                    errors_relative_p2[0] = abs(errors_p2[0]/params_p2[0]*100)
                if params_p2[1] != 0.0:
                    errors_relative_p2[1] = abs(errors_p2[1]/params_p2[1]*100)
                if params_p2[2] != 0.0:
                    errors_relative_p2[2] = abs(errors_p2[2]/params_p2[2]*100)
            else:
                params_p = [math.nan, math.nan]
                errors_p = [math.nan, math.nan]
                rsquared_p = math.nan
                errors_relative_p = [math.nan, math.nan]
                n_observations_p = 0
                params_p2 = [math.nan, math.nan, math.nan]
                errors_p2 = [math.nan, math.nan, math.nan]
                rsquared_p2 = math.nan
                errors_relative_p2 = [math.nan, math.nan, math.nan]
        else:
            params_p = [math.nan, math.nan]
            errors_p = [math.nan, math.nan]
            rsquared_p = math.nan
            errors_relative_p = [math.nan, math.nan]
            n_observations_p = 0
            params_p2 = [math.nan, math.nan, math.nan]
            errors_p2 = [math.nan, math.nan, math.nan]
            rsquared_p2 = math.nan
            errors_relative_p2 = [math.nan, math.nan, math.nan]

        if n_observations_t == 0 and n_observations_e == 0:
            continue
        else:
            dictionary_out.append(entry)
            dictionary_out[-1]['turnover_fit'] = {
                'params': params_t,
                'errors': errors_t,
                'rsq': rsquared_t,
                'errors_relative': errors_relative_t,
                'n': n_observations_t,
                'params2': params_t2,
                'errors2': errors_t2,
                'rsq2': rsquared_t2,
                'errors_relative2': errors_relative_t2,
            }
            dictionary_out[-1]['employment_fit'] = {
                'params': params_e,
                'errors': errors_e,
                'rsq': rsquared_e,
                'errors_relative': errors_relative_e,
                'n': n_observations_e,
                'params2': params_e2,
                'errors2': errors_e2,
                'rsq2': rsquared_e2,
                'errors_relative2': errors_relative_e2
            }
            dictionary_out[-1]['productivity_fit'] = {
                'params': params_p,
                'errors': errors_p,
                'rsq': rsquared_p,
                'errors_relative': errors_relative_p,
                'n': n_observations_p,
                'params2': params_p2,
                'errors2': errors_p2,
                'rsq2': rsquared_p2,
                'errors_relative2': errors_relative_p2
            }
    return 0

def clean_recent(dictionary_in, dictionary_out, limit_year):
    for entry in dictionary_in:
        year_t = math.nan
        year_e = math.nan
        if entry['turnover_fit']['n'] > 0:
            year_t = 2019
            for t in entry['turnover']:
                if math.isnan(t):
                    year_t = year_t - 1
                else:
                    break

        if entry['employment_fit']['n'] > 0:
            year_e = 2019
            for t in entry['employment']:
                if math.isnan(t):
                    year_e = year_e - 1
                else:
                    break

        if year_t >= limit_year or year_e >= limit_year:
            dictionary_out.append(entry)
    return 0

def categorise_fit(dictionary):
    for entry in tqdm(dictionary):
        entry['class'] = {}

        x = entry['turnover_fit']['params'][0]
        err = entry['turnover_fit']['errors'][0]

        if math.isnan(x):
            entry['class']['direction_t'] = 'unspecified'
        elif abs(x)-err <= 0:
            entry['class']['direction_t'] = 'steady'
        elif x > 0:
            entry['class']['direction_t'] = 'rising'
        else:
            entry['class']['direction_t'] = 'falling'
        
        x = entry['turnover_fit']['params2'][0]
        err = entry['turnover_fit']['errors2'][0]

        if math.isnan(x):
            entry['class']['direction_t2'] = 'unspecified'
        elif abs(x)-err <= 0:
            entry['class']['direction_t2'] = 'steady'
        elif x > 0:
            entry['class']['direction_t2'] = 'rising'
        else:
            entry['class']['direction_t2'] = 'falling'

        x = entry['employment_fit']['params'][0]
        err = entry['employment_fit']['errors'][0]

        if math.isnan(x):
            entry['class']['direction_e'] = 'unspecified'
        elif abs(x)-err <= 0:
            entry['class']['direction_e'] = 'steady'
        elif x > 0:
            entry['class']['direction_e'] = 'rising'
        else:
            entry['class']['direction_e'] = 'falling'
        
        x = entry['employment_fit']['params2'][0]
        err = entry['employment_fit']['errors2'][0]

        if math.isnan(x):
            entry['class']['direction_e2'] = 'unspecified'
        elif abs(x)-err <= 0:
            entry['class']['direction_e2'] = 'steady'
        elif x > 0:
            entry['class']['direction_e2'] = 'rising'
        else:
            entry['class']['direction_e2'] = 'falling'
            
        x = entry['productivity_fit']['params'][0]
        err = entry['productivity_fit']['errors'][0]

        if math.isnan(x):
            entry['class']['direction_p'] = 'unspecified'
        elif abs(x)-err <= 0:
            entry['class']['direction_p'] = 'steady'
        elif x > 0:
            entry['class']['direction_p'] = 'rising'
        else:
            entry['class']['direction_p'] = 'falling'
        
        x = entry['productivity_fit']['params2'][0]
        err = entry['productivity_fit']['errors2'][0]

        if math.isnan(x):
            entry['class']['direction_p2'] = 'unspecified'
        elif abs(x)-err <= 0:
            entry['class']['direction_p2'] = 'steady'
        elif x > 0:
            entry['class']['direction_p2'] = 'rising'
        else:
            entry['class']['direction_p2'] = 'falling'
    return 0

def calc_mean(dictionary):
    for entry in tqdm(dictionary):
        entry['turnover_change'] = {
            'mean': math.nan,
            'error': math.nan,
            'error_relative': math.nan,
            'difference': math.nan,
            'difference_p': math.nan,
            'difference_normalized': math.nan,
            'difference_p_normalized': math.nan
        }
        entry['employment_change'] = {
            'mean': math.nan,
            'error': math.nan,
            'error_relative': math.nan,
            'difference': math.nan,
            'difference_p': math.nan,
            'difference_normalized': math.nan,
            'difference_p_normalized': math.nan
        }
        entry['productivity_change'] = {
            'mean': math.nan,
            'error': math.nan,
            'error_relative': math.nan,
            'difference': math.nan,
            'difference_p': math.nan,
            'difference_normalized': math.nan,
            'difference_p_normalized': math.nan
        }
        if entry['turnover_fit']['n'] > 0:
            entry['turnover_change']['mean'] = np.nanmean(entry['turnover'])
            entry['turnover_change']['error'] = np.nanstd(entry['turnover'])
            entry['turnover_change']['error_relative'] = abs(entry['turnover_change']['error']/entry['turnover_change']['mean']*100)
            newest_t = math.nan
            oldest_t = math.nan
            n_difference = 0
            for t in entry['turnover']:
                if not math.isnan(t) and not t == 0:
                    if math.isnan(newest_t):
                        n_difference = 1
                        newest_t = t
                        oldest_t = t
                    else:
                        n_difference += 1
                        oldest_t = t
            entry['turnover_change']['difference'] = newest_t - oldest_t
            entry['turnover_change']['difference_p'] = ((newest_t - oldest_t)/oldest_t)*100
            if not n_difference == 0:
                entry['turnover_change']['difference_normalized'] = entry['turnover_change']['difference']/n_difference
                entry['turnover_change']['difference_p_normalized'] = entry['turnover_change']['difference_p']/n_difference
            
        if entry['employment_fit']['n'] > 0:
            entry['employment_change']['mean'] = np.nanmean(entry['employment'])
            entry['employment_change']['error'] = np.nanstd(entry['employment'])
            entry['employment_change']['error_relative'] = abs(entry['employment_change']['error']/entry['employment_change']['mean']*100)
            newest_e = math.nan
            oldest_e = math.nan
            n_difference = 0
            for e in entry['employment']:
                if not math.isnan(e) and not e == 0:
                    if math.isnan(newest_e):
                        n_difference = 1
                        newest_e = e
                        oldest_e = e
                    else:
                        n_difference += 1
                        oldest_e = e
            entry['employment_change']['difference'] = newest_e - oldest_e
            entry['employment_change']['difference_p'] = ((newest_e - oldest_e)/oldest_e)*100
            if not n_difference == 0:
                entry['employment_change']['difference_normalized'] = entry['employment_change']['difference']/n_difference
                entry['employment_change']['difference_p_normalized'] = entry['employment_change']['difference_p']/n_difference

        if entry['productivity_fit']['n'] > 0:
            entry['productivity_change']['mean'] = np.nanmean(entry['productivity'])
            entry['productivity_change']['error'] = np.nanstd(entry['productivity'])
            entry['productivity_change']['error_relative'] = abs(entry['productivity_change']['error']/entry['productivity_change']['mean']*100)
            newest_p = math.nan
            oldest_p = math.nan
            n_difference = 0
            for p in entry['productivity']:
                if not math.isnan(p) and not p == 0:
                    if math.isnan(newest_p):
                        n_difference = 1
                        newest_p = p
                        oldest_p = p
                    else:
                        n_difference += 1
                        oldest_p = p
            entry['productivity_change']['difference'] = newest_p - oldest_p
            entry['productivity_change']['difference_p'] = ((newest_p - oldest_p)/oldest_p)*100
            if not n_difference == 0:
                entry['productivity_change']['difference_normalized'] = entry['productivity_change']['difference']/n_difference
                entry['productivity_change']['difference_p_normalized'] = entry['productivity_change']['difference_p']/n_difference

    return 0

def get_country(dictionary_in, country_code):
    dictionary_out = []
    for entry in dictionary_in:
        if entry['country'] == country_code:
            dictionary_out.append(entry)
    return dictionary_out

def get_best(dictionary_in, dictionary_best, dictionary_rest, parameter='turnover'):
    if parameter == 'turnover':
        direction = 't'
    elif parameter == 'employment':
        direction = 'e'
    elif parameter == 'productivity':
        direction = 'p'
    else:
        raise AssertionError
        
    for entry in tqdm(dictionary_in):
        if entry['class']['direction_'+direction] == 'rising' and entry[parameter+'_fit']['rsq'] >= 0.5:
            dictionary_best.append(entry)
        else:
            dictionary_rest.append(entry)
    return 0

def get_best2(dictionary_in, dictionary_best, dictionary_rest, parameter='turnover'):
    if parameter == 'turnover':
        direction = 't'
    elif parameter == 'employment':
        direction = 'e'
    elif parameter == 'productivity':
        direction = 'p'
    else:
        raise AssertionError
        
    parameter_changes = []
    for entry in tqdm(dictionary_in):
        parameter_changes.append(entry[parameter+'_change']['difference_p_normalized'])
    mean_parameter_change = np.nanmean(parameter_changes)
    for entry in tqdm(dictionary_in):
        if (entry['class']['direction_'+direction] == 'rising' and 
            entry[parameter+'_fit']['rsq'] >= 0.5 and 
            entry[parameter+'_change']['difference_p_normalized'] >= mean_parameter_change
           ):
            dictionary_best.append(entry)
        else:
            dictionary_rest.append(entry)
    return 0

def get_turnarounds(dictionary_in, dictionary_best, dictionary_rest, parameter='turnover'):
    if parameter == 'turnover':
        direction = 't'
    elif parameter == 'employment':
        direction = 'e'
    elif parameter == 'productivity':
        direction = 'p'
    else:
        raise AssertionError
        
    for entry in tqdm(dictionary_in):
        if (entry[parameter+'_fit']['rsq'] < 0.5 and 
            entry['class']['direction_'+direction+'2'] == 'rising' and 
            entry[parameter+'_fit']['rsq2'] >= 0.7
           ):
            dictionary_best.append(entry)
        else:
            dictionary_rest.append(entry)
    return 0


all_data = dict()

countries_all = ['bg', 'it', 'hr', 'es', 'de']
turkey_all = ['bebka', 'marka']

bg_employment_translation = {
    '1': (9-1)/2,
    '2': (50-10)/2,
    '3': (100-51)/2,
    '4': (250-101)/2,
    '5': math.inf
}
bg_turnover_translation = {
    '1': (50000-10000)/2,
    '2': (100000-51000)/2,
    '3': (500000-101000)/2,
    '4': (1000000-501000)/2,
    '5': (20000000-1001000)/2,
    '6': math.inf
}
bg_eur = {
    '2011': np.mean([0.511272,0.511278,0.511330,0.511910,0.511762,0.511261,0.511488,0.511298,0.512770,0.512133,0.511286,0.511193]),
    '2012': np.mean([0.511276,0.511251,0.511286,0.511725,0.511297,0.511451,0.511209,0.511289,0.511460,0.511485,0.511073,0.511079]),
    '2013': np.mean([0.511037,0.510701,0.511052,0.510221,0.510553,0.510934,0.511190,0.511316,0.511228,0.511367,0.511345,0.511305]),
    '2014': np.mean([0.511354,0.511389,0.511438,0.511373,0.511161,0.511185,0.511246,0.511496,0.511314,0.511388,0.511408,0.511238]),
    '2015': np.mean([0.511293,0.511454,0.511287,0.511582,0.511364,0.511277,0.511289,0.511427,0.511386,0.511235,0.511146,0.511238]),
    '2016': np.mean([0.511361,0.511331,0.511395,0.511334,0.511330,0.511326,0.511240,0.511378,0.511315,0.511406,0.511119,0.510547]),
    '2017': np.mean([0.511131,0.511202,0.531675,0.511076,0.510883,0.510821,0.510808,0.510905,0.511365,0.511197,0.511318,0.511313]),
    '2018': 0.511292
}
bg_pl_translation = {
    '0': 0,
    '1': (10000-1000)/2,
    '2': (50000-11000)/2,
    '3': (100000-51000)/2,
    '4': (500000-101000)/2,
    '5': (1000000-501000)/2,
    '6': math.inf,
    '7': (-10000-(-1000))/2,
    '8': (-50000-(-11000))/2,
    '9': (-100000-(-51000))/2,
    '10': (-500000-(-101000))/2,
    '11': -math.inf
}
bg_assets_translation = {
    '1': (100000)/2,
    '2': (500000-101000)/2,
    '3': (1000000-501000)/2,
    '4': (10000000-1001000)/2,
    '5': math.inf
}
bg_capital_translation = {
    '1': (100000)/2,
    '2': (500000-101000)/2,
    '3': (1000000-501000)/2,
    '4': (10000000-1001000)/2,
    '8': math.inf,
    '9': (-100000-(-1000))/2,
    '10': (-500000-(-101000))/2,
    '11': (-1000000-(-501000))/2,
    '12': -math.inf
}


path = 'data/bvd_rdas'
all_data['files_rdas'] = find_files(path)

path = 'data/bvd_background'
all_data['files_background'] = find_files(path)

path = 'data/Manu'
all_data['files_Manu'] = find_files(path)

path = 'data/rda'
all_data['files_rda'] = find_files(path)


all_data['orbis_rdas'] = read_bvd_files(all_data['files_rdas'])
all_data['orbis_background'] = read_bvd_files(all_data['files_background'])

for file in all_data['files_Manu']:
    if 'weser_ems' in file:
        if 'lkn' in file:
            all_data['Manu_lkn_de'] = read_Manu_file(file)
        else:
            all_data['Manu_grants_de'] = read_Manu_file(file)
    elif 'aragon' in file:
        if 'lkn' in file:
            all_data['Manu_lkn_es'] = read_Manu_file(file)
        else:
            all_data['Manu_grants_es'] = read_Manu_file(file)
    elif 'simora' in file:
        if 'lkn' in file:
            all_data['Manu_lkn_hr'] = read_Manu_file(file)
        else:
            all_data['Manu_grants_hr'] = read_Manu_file(file)
    elif 'gabrovo' in file:
        if 'lkn' in file:
            all_data['Manu_lkn_bg'] = read_Manu_file(file)
        else:
            all_data['Manu_grants_bg'] = read_Manu_file(file)
    elif 'bebka' in file:
        if 'lkn' in file:
            all_data['Manu_lkn_tr_bebka'] = read_Manu_file(file)
        else:
            all_data['Manu_grants_tr_bebka'] = read_Manu_file(file)
    elif 'marka' in file:
        if 'lkn' in file:
            all_data['Manu_lkn_tr_marka'] = read_Manu_file(file)
        else:
            all_data['Manu_grants_tr_marka'] = read_Manu_file(file)
all_data['Manu_lkn_it'] = []
all_data['Manu_grants_it'] = []

for country in countries_all:
    all_data['rda_'+country] = dict()
    for file in all_data['files_rda']:
        if country in file:
            year = file.split('_')
            year = year[-1]
            year = year.split('.')
            year = int(year[0])
            all_data['rda_'+country][year] = read_rda_file(file, country)


all_data['orbis_dicts'] = parse_orbis(all_data['orbis_rdas'])
all_data['background_dicts_first'] = parse_orbis(all_data['orbis_background'])
all_data['Manu_dicts'] = dict()
for country in countries_all:
    all_data['Manu_dicts']['lkn_'+country] = parse_Manu(all_data['Manu_lkn_'+country], country, 'lkn')
    all_data['Manu_dicts']['grants_'+country] = parse_Manu(all_data['Manu_grants_'+country], country, 'grants')
for turkey in turkey_all:
    all_data['Manu_dicts']['lkn_tr_'+turkey] = parse_Manu(all_data['Manu_lkn_tr_'+turkey], 'tr_'+turkey, 'lkn')
    all_data['Manu_dicts']['grants_tr_'+turkey] = parse_Manu(all_data['Manu_grants_tr_'+turkey], 'tr_'+turkey, 'grants')

all_data['rda_dicts'] = dict()
all_data['rda_dicts']['hr'] = parse_rda(all_data['rda_hr'], 'hr')
all_data['rda_dicts']['bg'] = parse_rda(all_data['rda_bg'], 'bg')


duplicates = []
for rda in tqdm(all_data['orbis_dicts']):
    for entry in all_data['background_dicts_first']:
        if rda['ID'] == entry['ID']:
            duplicates.append(rda.copy())
            break
            
background_dicts = []
for entry in tqdm(all_data['background_dicts_first']):
    for dup in duplicates:
        if entry['ID'] == dup['ID']:
            break
    else:
        background_dicts.append(entry)

all_data['duplicates'] = duplicates
all_data['background_dicts'] = background_dicts


for country in tqdm(countries_all):
    names_table = correlate_names(all_data['Manu_dicts']['grants_'+country], country)
    save_namelist(names_table, 'names_correlation/names_'+country+'_Manu.txt')
for country in ['hr','bg']:
    names_table = correlate_names(all_data['rda_dicts'][country], country)
    save_namelist(names_table, 'names_correlation/names_'+country+'_rda.txt')

#--------------------------------------

all_data['names_correlation'] = dict()

all_data['names_correlation']['Manu'] = dict()
for country in tqdm(countries_all):
    all_data['names_correlation']['Manu'][country] = []
    read_names_correlation('names_correlation/names_'+country+'_Manu_checked.txt', all_data['names_correlation']['Manu'][country])

all_data['names_correlation']['rda'] = dict()
for country in ['hr', 'bg']:
    all_data['names_correlation']['rda'][country] = []
    read_names_correlation('names_correlation/names_'+country+'_rda_checked.txt', all_data['names_correlation']['rda'][country])


for country in countries_all:
    connect_datasources(all_data['orbis_dicts'], country)
    connect_datasources(all_data['background_dicts'], country)
for turkey in turkey_all:
    connect_datasources(all_data['orbis_dicts'], 'tr_'+turkey)
    connect_datasources(all_data['background_dicts'], 'tr_'+turkey)
    

all_data['background_dicts_first'] = all_data['background_dicts']
duplicates = []
for rda in tqdm(all_data['orbis_dicts']):
    for entry in all_data['background_dicts_first']:
        if rda['ID'] == entry['ID']:
            duplicates.append(rda.copy())
            break
            
background_dicts = []
for entry in tqdm(all_data['background_dicts_first']):
    for dup in duplicates:
        if entry['ID'] == dup['ID']:
            break
    else:
        background_dicts.append(entry)

all_data['duplicates'] = duplicates
all_data['background_dicts'] = background_dicts


all_data['orbis_dicts'] = get_smes(all_data['orbis_dicts'])
all_data['background_dicts_first'] = get_smes(all_data['background_dicts_first'])


calc_productivity(all_data['orbis_dicts'])
calc_productivity(all_data['background_dicts'])


all_data['orbis_dicts_rda'] = []
calc_fit(all_data['orbis_dicts'], all_data['orbis_dicts_rda'])
            
all_data['orbis_dicts_background'] = []
calc_fit(all_data['background_dicts'], all_data['orbis_dicts_background'])


limit_year = 2017

all_data['orbis_dicts_rda_recent'] = []
clean_recent(all_data['orbis_dicts_rda'], all_data['orbis_dicts_rda_recent'], limit_year)

all_data['orbis_dicts_background_recent'] = []
clean_recent(all_data['orbis_dicts_background'], all_data['orbis_dicts_background_recent'], limit_year)


categorise_fit(all_data['orbis_dicts_background_recent'])
categorise_fit(all_data['orbis_dicts_rda_recent'])


calc_mean(all_data['orbis_dicts_background_recent'])
calc_mean(all_data['orbis_dicts_rda_recent'])


countries_turkey = countries_all + ['tr_'+t for t in turkey_all]
countries = np.column_stack([countries_turkey, [country.upper() for country in countries_turkey]])
for country in countries:
    all_data[country[0]+'_background'] = get_country(all_data['orbis_dicts_background_recent'], country[1])
    all_data[country[0]+'_rda'] = get_country(all_data['orbis_dicts_rda_recent'], country[1])


for country in countries_all:
    all_data[country+'_turnover_rda_best'] = []
    all_data[country+'_turnover_rda_rest'] = []
    all_data[country+'_turnover_rda_turnarounds'] = []
    all_data[country+'_turnover_rda_rest_rest'] = []
    
    all_data[country+'_turnover_background_best'] = []
    all_data[country+'_turnover_background_rest'] = []
    all_data[country+'_turnover_background_turnarounds'] = []
    all_data[country+'_turnover_background_rest_rest'] = []

    get_best(all_data[country+'_rda'],all_data[country+'_turnover_rda_best'],all_data[country+'_turnover_rda_rest'])
    get_turnarounds(all_data[country+'_turnover_rda_rest'],all_data[country+'_turnover_rda_turnarounds'],all_data[country+'_turnover_rda_rest_rest'])
    get_best(all_data[country+'_background'],all_data[country+'_turnover_background_best'],all_data[country+'_turnover_background_rest'])
    get_turnarounds(all_data[country+'_turnover_background_rest'],all_data[country+'_turnover_background_turnarounds'],all_data[country+'_turnover_background_rest_rest'])

    all_data[country+'_turnover_rda_best2'] = []
    all_data[country+'_turnover_rda_rest2'] = []
    all_data[country+'_turnover_rda_turnarounds2'] = []
    all_data[country+'_turnover_rda_rest_rest2'] = []
    
    all_data[country+'_turnover_background_best2'] = []
    all_data[country+'_turnover_background_rest2'] = []
    all_data[country+'_turnover_background_turnarounds2'] = []
    all_data[country+'_turnover_background_rest_rest2'] = []

    get_best2(all_data[country+'_rda'],all_data[country+'_turnover_rda_best2'],all_data[country+'_turnover_rda_rest2'])
    get_turnarounds(all_data[country+'_turnover_rda_rest2'],all_data[country+'_turnover_rda_turnarounds2'],all_data[country+'_turnover_rda_rest_rest2'])
    get_best2(all_data[country+'_background'],all_data[country+'_turnover_background_best2'],all_data[country+'_turnover_background_rest2'])
    get_turnarounds(all_data[country+'_turnover_background_rest2'],all_data[country+'_turnover_background_turnarounds2'],all_data[country+'_turnover_background_rest_rest2'])
    
    
    all_data[country+'_productivity_rda_best'] = []
    all_data[country+'_productivity_rda_rest'] = []
    all_data[country+'_productivity_rda_turnarounds'] = []
    all_data[country+'_productivity_rda_rest_rest'] = []
    
    all_data[country+'_productivity_background_best'] = []
    all_data[country+'_productivity_background_rest'] = []
    all_data[country+'_productivity_background_turnarounds'] = []
    all_data[country+'_productivity_background_rest_rest'] = []

    get_best(all_data[country+'_rda'],all_data[country+'_productivity_rda_best'],all_data[country+'_productivity_rda_rest'], parameter='productivity')
    get_turnarounds(all_data[country+'_productivity_rda_rest'],all_data[country+'_productivity_rda_turnarounds'],all_data[country+'_productivity_rda_rest_rest'], parameter='productivity')
    get_best(all_data[country+'_background'],all_data[country+'_productivity_background_best'],all_data[country+'_productivity_background_rest'], parameter='productivity')
    get_turnarounds(all_data[country+'_productivity_background_rest'],all_data[country+'_productivity_background_turnarounds'],all_data[country+'_productivity_background_rest_rest'], parameter='productivity')

    all_data[country+'_productivity_rda_best2'] = []
    all_data[country+'_productivity_rda_rest2'] = []
    all_data[country+'_productivity_rda_turnarounds2'] = []
    all_data[country+'_productivity_rda_rest_rest2'] = []
    
    all_data[country+'_productivity_background_best2'] = []
    all_data[country+'_productivity_background_rest2'] = []
    all_data[country+'_productivity_background_turnarounds2'] = []
    all_data[country+'_productivity_background_rest_rest2'] = []

    get_best2(all_data[country+'_rda'],all_data[country+'_productivity_rda_best2'],all_data[country+'_productivity_rda_rest2'], parameter='productivity')
    get_turnarounds(all_data[country+'_productivity_rda_rest2'],all_data[country+'_productivity_rda_turnarounds2'],all_data[country+'_productivity_rda_rest_rest2'], parameter='productivity')
    get_best2(all_data[country+'_background'],all_data[country+'_productivity_background_best2'],all_data[country+'_productivity_background_rest2'], parameter='productivity')
    get_turnarounds(all_data[country+'_productivity_background_rest2'],all_data[country+'_productivity_background_turnarounds2'],all_data[country+'_productivity_background_rest_rest2'], parameter='productivity')