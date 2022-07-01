# -*- coding: utf-8 -*-
"""
@authors: Lorenzo Rinaldi,  Department of Energy, Politecnico di Milano
          Matteo V. Rocco,  Department of Energy, Politecnico di Milano
          Nicolò Golinucci, Department of Energy, Politecnico di Milano
"""

#%% Importing dependences
import mario  # https://mario-suite.readthedocs.io/en/latest/
import pandas as pd
import numpy as np
from copy import deepcopy
from Functions import summation_matrix, production_by_region, decompositions_by_region, \
                      decompositions_by_sector, calc_domestic, calc_imported, calc_exported, calc_net_excl_dom, calc_net_incl_dom, calc_net_over_dom, metabolism_dynamics, \
                      units_parser, units_converter, transactions_total_demand, get_carbon_tax_excel, tax_filter_generation, transactions_matrix_filtered,\
                      calc_ghosh, calc_price, competition_among_imports, plot_competition_heatmap, subplot_grid, calc_tax_revenues, plot_metabolism, generate_carbon_taxes_simulations, \
                      get_emissions

#%% Setting up directories and basic params
paths = {"exiobase": r"C:\Users\loren\OneDrive - Politecnico di Milano\Politecnico di Milano\Research\Database\Exiobase\3.8.2\economic\IOT\ixi\IOT_2019_ixi.zip",
         "aggregation": r"C:\Users\loren\OneDrive - Politecnico di Milano\Politecnico di Milano\Research\Database\Exiobase\aggregations\EXIOBASE_3.8_IOT_ixi_agg.xlsx",
         "database": r"C:\Users\loren\Politecnico di Milano\Matteo Vincenzo Rocco - 2021_EY_Energy transition\comunication\Davos teaser\model\Excel",
         "saved": r"C:\Users\loren\Politecnico di Milano\Matteo Vincenzo Rocco - 2021_EY_Energy transition\comunication\Davos teaser\model\Excel\flows",
         "emissions": r"C:\Users\loren\Politecnico di Milano\Matteo Vincenzo Rocco - 2021_EY_Energy transition\comunication\Davos teaser\model\Excel\Emissions.xlsx",
         "results": r"C:\Users\loren\Politecnico di Milano\Matteo Vincenzo Rocco - 2021_EY_Energy transition\comunication\Davos teaser\model\Excel/disaggregated/Export_disaggregated_new_cases_last.xlsx"}

accounts = ["CO2", "Value Added"]
categories = ['Domestic','Imported','Exported','Net (excluding domestic)','Net (including domestic)', "Net/Domestic"]

#%% Parse, aggregate and save exiobase (use if aggregated database is not ready)
# world = mario.parse_exiobase_3(path=paths["exiobase"],)
# world = world.aggregate(paths["aggregation"], inplace=False, drop='unused')
# world.to_txt(paths["database"])

#%% Parse aggregated database (use in case an aggregated version of the database is ready)
world = mario.parse_from_txt(paths["saved"], table="IOT")

#%% Algebra
summ_matrix = summation_matrix(world)  # summation matrix
X_mat = production_by_region(world)    # production matrix
Z_totdem, Y_totdem, X_totdem = transactions_total_demand(world.Z, world.Y, world) # transactions matrix total demand

#%% Calculating decompositions of accounts exchanged in trades
decompositions_by_reg = decompositions_by_region(world, accounts, summ_matrix, X_mat)
decompositions_by_sec = decompositions_by_sector(world, accounts, X_mat)

for account in accounts:
    decompositions_by_reg[account] = calc_domestic(decompositions_by_reg[account])
    decompositions_by_reg[account] = calc_imported(decompositions_by_reg[account])
    decompositions_by_reg[account] = calc_exported(decompositions_by_reg[account])
    decompositions_by_reg[account] = calc_net_excl_dom(decompositions_by_reg[account])
    decompositions_by_reg[account] = calc_net_incl_dom(decompositions_by_reg[account])
    decompositions_by_reg[account] = calc_net_over_dom(decompositions_by_reg[account])
    
    for sector in world.get_index('Sector'):
        decompositions_by_sec[account][sector] = calc_domestic(decompositions_by_sec[account][sector])
        decompositions_by_sec[account][sector] = calc_imported(decompositions_by_sec[account][sector])
        decompositions_by_sec[account][sector] = calc_exported(decompositions_by_sec[account][sector])
        decompositions_by_sec[account][sector] = calc_net_incl_dom(decompositions_by_sec[account][sector])
        decompositions_by_sec[account][sector] = calc_net_excl_dom(decompositions_by_sec[account][sector])
        decompositions_by_sec[account][sector] = calc_net_over_dom(decompositions_by_sec[account][sector])

#%% Defining metabolism by account, region and sector
metabolism = metabolism_dynamics(decompositions_by_sec, decompositions_by_reg)

#%% Getting units of measure
units = units_parser(world, accounts)

#%% Converting unit of measures for plots
new_units = ['Million tonnes', "Million Euros"]
conversion_factors = [1e-9, 1]
decompositions_by_sec, decompositions_by_reg, units = units_converter(units, new_units, conversion_factors, accounts, decompositions_by_sec, decompositions_by_reg)

#%% Implementing carbon taxes 
taxed_regions = world.get_index('Region')+['Global tax']
tax_mechanisms = ['PBA','CBA','CBAM']
carbon_price = 90 #€/ton

carbon_taxes = generate_carbon_taxes_simulations(taxed_regions,tax_mechanisms,carbon_price,world)

#%% Export emissions
emissions = get_emissions(world)
emissions.to_excel(paths["emissions"])

#%%
new_cases = {}
new_cases['PBA'] = {}
new_cases['CBAM'] = {}
new_cases['CBA'] = {}

new_cases['PBA']['PBA-EU-PBA-CN'] = deepcopy(carbon_taxes['PBA']['EU28'])
new_cases['PBA']['PBA-EU-PBA-CN'].loc['PBA',('EU28',slice(None))] = 70
new_cases['PBA']['PBA-EU-PBA-CN'].loc['PBA',('CHN',slice(None))] = 70

new_cases['CBAM']['CBAM-EU-CBAM-CN'] = deepcopy(new_cases['PBA']['PBA-EU-PBA-CN'])
new_cases['CBAM']['CBAM-EU-CBAM-CN'].loc['CBA',('EU28',slice(None))] = 70
new_cases['CBAM']['CBAM-EU-CBAM-CN'].loc['CBA',('CHN',slice(None))] = 70

new_cases['CBA']['CBA-EU-CBA-CN'] = deepcopy(new_cases['PBA']['PBA-EU-PBA-CN'])
new_cases['CBA']['CBA-EU-CBA-CN'].loc['CBA',('EU28',slice(None))] = 70
new_cases['CBA']['CBA-EU-CBA-CN'].loc['CBA',('CHN',slice(None))] = 70
new_cases['CBA']['CBA-EU-CBA-CN'].loc['PBA',:] = 0

#%% getting simulation list
simulations = ['{}_{}'.format(i,j) for j in new_cases[list(new_cases.keys())[0]] for i in new_cases]

simulations = ['PBA_PBA-EU-PBA-CN','CBAM_CBAM-EU-CBAM-CN','CBA_CBA-EU-CBA-CN']

#%%
export = pd.ExcelWriter(paths['results'])

for simulation in simulations:
 
    tax = simulation.split('_')[0]
    region = simulation.split('_')[1]
    ctax = deepcopy(new_cases[tax][region])
    ctax /= 1e9
        
    # tax filter
    tax_filter = tax_filter_generation(ctax)
    
    # filtering transaction matrix and calculating ghosh
    Z_filtered = transactions_matrix_filtered(world, tax_filter)
    Z_totdem_new, Y_totdem_new, X_totdem_new = transactions_total_demand(Z_filtered, world.matrices['baseline']['Y'], world)
    
    demand_share_matrix = calc_ghosh(X_totdem,Z_totdem_new)
    
    #
    z_filtered = pd.DataFrame(Z_filtered.values @ np.linalg.inv(np.diagflat(world.X.values)),
                              index=Z_filtered.index,
                              columns=Z_filtered.columns)
    
    
    # calculating price indices
    price_indices, total_price_index, total_price_index_by_reg = calc_price(ctax, z_filtered, world)
    
    # calculating tax revenues
    tax_revenues = calc_tax_revenues(price_indices, world, X_totdem, ctax, z_filtered)
    
    # competition among imports
    price_competition = competition_among_imports(price_indices, ctax, world, emissions)
    
    # export to excel
    toexcel = deepcopy(ctax)#*1e9)
    toexcel.columns = price_indices.columns
    toexcel = toexcel.append(pd.DataFrame(["" for i in range(toexcel.shape[1])], index=toexcel.columns, columns=[""]).T)
    toexcel = toexcel.append(pd.DataFrame(["" for i in range(toexcel.shape[1])], index=toexcel.columns, columns=["Price indices"]).T)

    toexcel = toexcel.append(price_indices)
    toexcel = toexcel.append(pd.DataFrame(["" for i in range(toexcel.shape[1])], index=toexcel.columns, columns=[""]).T)
    toexcel = toexcel.append(pd.DataFrame(["" for i in range(toexcel.shape[1])], index=toexcel.columns, columns=["Competition with imports"]).T)
    toexcel = toexcel.append(pd.DataFrame(["" for i in range(toexcel.shape[1])], index=toexcel.columns, columns=["Values"]).T)

    toexcel = toexcel.append(price_competition["Values"])
    toexcel = toexcel.append(pd.DataFrame(["" for i in range(toexcel.shape[1])], index=toexcel.columns, columns=[""]).T)
    toexcel = toexcel.append(pd.DataFrame(["" for i in range(toexcel.shape[1])], index=toexcel.columns, columns=["Differences"]).T)
    
    toexcel = toexcel.append(price_competition["Differences"])
    toexcel = toexcel.append(pd.DataFrame(["" for i in range(toexcel.shape[1])], index=toexcel.columns, columns=[""]).T)
    toexcel = toexcel.append(pd.DataFrame(["" for i in range(toexcel.shape[1])], index=toexcel.columns, columns=["Tax revenues"]).T)

    toexcel = toexcel.append(tax_revenues)
    
    toexcel.columns = pd.MultiIndex.from_arrays([toexcel.columns.get_level_values(0), toexcel.columns.get_level_values(-1)], 
                                                names=["","Carbon tax"])
    
    toexcel.to_excel(export, sheet_name=simulation)

export.save()
export.close()    
