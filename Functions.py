# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 13:47:55 2021

@author: loren
"""


import pandas as pd
import numpy as np
import copy
import math
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#%%
def summation_matrix(database):
    
    summ_matrix = database.Y*0
    regions = database.get_index('Region')
    
    for region in regions:
        summ_matrix.loc[(region,slice(None),slice(None)),(region,slice(None),slice(None))] += 1
    
    return summ_matrix

#%%
def production_by_region(database):
    
    X_by_reg = database.w @ database.Y
    X_by_reg.columns = pd.MultiIndex.from_arrays([database.Y.columns.get_level_values(0), database.Y.columns.get_level_values(1), ["Production" for i in range(X_by_reg.shape[1])]])
    
    return X_by_reg

#%%
def transactions_total_demand(Z,Y,database):
    
    Z_totdem = database.Z*0
    Y_totdem = database.Y*0
        
    for region in database.get_index('Region'):
        # other_regions = database.get_index('Region')
        # other_regions.remove(region)
        
        for other_region in database.get_index('Region'):
            Z_no_trades = Z.loc[(other_region,slice(None),slice(None)),(region,slice(None),slice(None))]
            Y_no_trades = Y.loc[(other_region,slice(None),slice(None)),(region,slice(None),slice(None))]
              
            Z_totdem.loc[(region,slice(None),slice(None)),(region,slice(None),slice(None))] += Z_no_trades.values
            Y_totdem.loc[(region,slice(None),slice(None)),(region,slice(None),slice(None))] += Y_no_trades.values
    
    X_totdem = pd.DataFrame(Z_totdem.sum(1).to_frame().values + Y_totdem.sum(1).to_frame().values,
                            index = database.X.index,
                            columns = database.X.columns)
            
    return Z_totdem, Y_totdem, X_totdem

#%%
def calc_ghosh(X,Z):
    
    g = pd.DataFrame(
                     np.linalg.inv(np.diagflat(X.values)) @ Z.values,
                     index = Z.index,
                     columns = Z.columns,
                    )

    return g

#%%
def decompositions_by_region(database, accounts:list, summ_matrix, X_by_reg):
    
    decompositions_by_reg = {}

    for account in accounts:       
        
        for item in database.units:
            if database.search(item, account):
                if item == 'Satellite account':
                    matrix = database.e
                if item == 'Factor of production':
                    matrix = database.v      
        
                decompositions_by_reg[account] = pd.DataFrame( 
                                                              data    = summ_matrix.values.T @ (np.diagflat(matrix.values) @ X_by_reg.values),
                                                              index   = list(dict.fromkeys(summ_matrix.index.get_level_values(0))),
                                                              columns = list(dict.fromkeys(summ_matrix.index.get_level_values(0))),
                                                             )
    
    return decompositions_by_reg
    
#%%
def decompositions_by_sector(database, accounts:list, X_by_reg):
    
    decompositions_by_sec = {}
    
    for account in accounts:       
        
        decompositions_by_sec[account] = {}
        
        for item in database.units:
            if database.search(item, account):
                if item == 'Satellite account':
                    matrix = copy.deepcopy(database.e)
                if item == 'Factor of production':
                    matrix = copy.deepcopy(database.v)
                        
                decomposition = pd.DataFrame( 
                                             data    = np.diagflat(matrix.values) @ X_by_reg.values,
                                             index   = X_by_reg.index,
                                             columns = list(dict.fromkeys(X_by_reg.columns.get_level_values(0))),
                                            )
                
                sectors = list(dict.fromkeys(matrix.columns.get_level_values(-1)))        
                for sector in sectors:
                    decompositions_by_sec[account][sector] = decomposition.loc[(slice(None),slice(None),sector), :]
                    decompositions_by_sec[account][sector].index = decompositions_by_sec[account][sector].index.get_level_values(0)

    return decompositions_by_sec

#%% 
def calc_domestic(decomposition):
    
    regions = copy.deepcopy(list(decomposition.columns))
    
    for region in regions:
        decomposition.loc["Domestic",region] = decomposition.loc[region,region].sum(0)

    return decomposition           
    
#%%
def calc_imported(decomposition):
    
    regions = copy.deepcopy(list(decomposition.columns))
    
    for region in regions:
        other_regions = copy.deepcopy(regions)
        other_regions.remove(region)
        decomposition.loc["Imported",region] = -decomposition.loc[other_regions,region].sum(0)

    return decomposition           

#%%
def calc_exported(decomposition):
    
    regions = copy.deepcopy(list(decomposition.columns))
    
    for region in regions:
        other_regions = copy.deepcopy(regions)
        other_regions.remove(region)
        decomposition.loc["Exported",region] = decomposition.loc[region,other_regions].sum().sum()

    return decomposition           

#%%
def calc_net_excl_dom(decomposition):
    
    regions = copy.deepcopy(list(decomposition.columns))
    
    for region in regions:
        other_regions = copy.deepcopy(regions)
        other_regions.remove(region)
        decomposition.loc["Net (excluding domestic)",region] = decomposition.loc["Imported", region] +\
                                                               decomposition.loc["Exported", region]

    return decomposition           

#%%
def calc_net_incl_dom(decomposition):
    
    regions = copy.deepcopy(list(decomposition.columns))
    
    for region in regions:
        other_regions = copy.deepcopy(regions)
        other_regions.remove(region)
        decomposition.loc["Net (including domestic)",region] = decomposition.loc["Domestic", region] +\
                                                decomposition.loc["Imported", region] +\
                                                decomposition.loc["Exported", region]

    return decomposition           

#%%
def calc_net_over_dom(decomposition):
    
    regions = copy.deepcopy(list(decomposition.columns))
    
    for region in regions:
        other_regions = copy.deepcopy(regions)
        other_regions.remove(region)
        decomposition.loc["Net/Domestic",region] = np.abs(decomposition.loc["Net (excluding domestic)", region] / decomposition.loc["Domestic", region])

    return decomposition 
  
#%%
def metabolism_dynamics(decompositions_by_sec, decompositions_by_reg):
    
    metabol = {}
    regions = decompositions_by_sec[list(decompositions_by_sec.keys())[0]][list(decompositions_by_sec[list(decompositions_by_sec.keys())[0]].keys())[0]].columns
    
    for region in regions:
        metabol[region] = {}
        metabol[region]['_Overall'] = {}
        
        for sector in list(decompositions_by_sec[list(decompositions_by_sec.keys())[0]].keys()):
            metabol[region][sector] = {}

            for account in decompositions_by_sec:
                if decompositions_by_sec[account][sector].loc['Net (excluding domestic)', region] < 0:
                    metabol[region][sector][account] = 'Net importer'
                elif decompositions_by_sec[account][sector].loc['Net (excluding domestic)', region] > 0:
                    metabol[region][sector][account] = 'Net exporter'
                else:
                    metabol[region][sector][account] = 'Neutral'
        
        for account in decompositions_by_reg:
            if decompositions_by_reg[account].loc['Net (excluding domestic)', region] < 0:
                metabol[region]['_Overall'][account] = 'Net importer'
            elif decompositions_by_reg[account].loc['Net (excluding domestic)', region] > 0:
                metabol[region]['_Overall'][account] = 'Net exporter'
            else:
                metabol[region]['_Overall'][account] = 'Neutral'
    
    return metabol
    

#%%
def units_parser(database, accounts):
    
    units = {}
    
    for account in accounts:
        
        for item in database.units:
            try:
                units[account] = database.units[item].loc[account,"unit"]
            except:
                pass
        
    return units

#%%
def units_converter(units, new_units, conversion_factors, accounts, decompositions_by_sec, decompositions_by_reg):
    
    for account in accounts:
        
        units[account] = new_units[accounts.index(account)]
        decompositions_by_reg[account].iloc[:-1,:] *= conversion_factors[accounts.index(account)]
        decompositions_by_reg[account].iloc[-1,:] *= 100
        
        for sector in decompositions_by_sec[account]:
            decompositions_by_sec[account][sector].iloc[:-1,:] *= conversion_factors[accounts.index(account)]
            decompositions_by_sec[account][sector].iloc[-1,:] *= 100
        
    return decompositions_by_sec, decompositions_by_reg, units

#%%
def get_carbon_tax_excel(path, database, tax_types):
    
    carbon_taxes = pd.DataFrame(
                                np.zeros((len(tax_types), database.Z.shape[1])),
                                index = pd.MultiIndex.from_arrays([tax_types, ['Euro/ton' for i in range(len(tax_types))]]),
                                columns = pd.MultiIndex.from_arrays([database.Z.columns.get_level_values(0), database.Z.columns.get_level_values(-1)])
                               )
    
    carbon_taxes.to_excel(path)

#%%
def tax_filter_generation(ctax):
    
    tax_filter = pd.DataFrame(
                              np.ones((ctax.shape[1], ctax.shape[1])),
                              index = ctax.columns,
                              columns = ctax.columns,
                             )
    
    tax_types = list(ctax.index.get_level_values(0))
    
    for pos1,tax1 in enumerate(ctax.loc[tax_types[0], :].values.tolist()):
        for pos2,tax2 in enumerate(ctax.loc[tax_types[1], :].values.tolist()):
            
            if tax1!=0 and tax2!=0:
                tax_filter.iloc[pos1,pos2] = tax2-tax1
    
    return tax_filter
                
#%%
def transactions_matrix_filtered(database, tax_filter):       
        
    Z_filtered = pd.DataFrame(
                              np.multiply(database.Z.values, tax_filter.values),
                              index = database.Z.index,
                              columns = database.Z.columns
                             )
    
    return Z_filtered

#%%
def get_emissions(database):
    
    emissions = database.e.loc["CO2",:].to_frame().T
    emissions = emissions.append(database.f.loc["CO2",:].to_frame().T)
    emissions = emissions.append(pd.DataFrame((np.divide(database.F.values, database.X.values.T)), index=[''], columns=emissions.columns))
    
    emissions.index = ['Direct CO2 emissions', 'Embeddied CO2 emissions (Y)', 'Embeddied CO2 emissions (X)']
    emissions.columns.names = ["Region","Level","Sector"]
    
    emissions = emissions.groupby(level=[0,2], axis=1, sort=True).sum()
    
    return emissions

#%%
def calc_price(ctax, z_filtered, database):
    
    price_indices = pd.DataFrame(
                                 database.p.values.T,
                                 index = ['Initial'],
                                 columns = database.e.columns
                                )
    
    for tax in ctax.index.get_level_values(0):
        if tax == "PBA":
            p = pd.DataFrame(
                             (database.e.values @ np.diagflat(ctax.loc[tax,:].values)) @ database.w.values,
                             index = [tax],
                             columns = database.e.columns
                            )
            
        elif tax == "CBA":
            p = pd.DataFrame(
                             (np.multiply((database.e.values @ z_filtered.values), ctax.loc[tax,:].values) @ database.w.values),
                             index = [tax],
                             columns = database.e.columns
                            )
        price_indices = pd.concat([price_indices, p], axis=0)

        
    price_indices = pd.concat([price_indices, pd.DataFrame(price_indices.sum(0).to_frame().T.values,
                                                           index = ['Price index base'],
                                                           columns = price_indices.columns
                                                           )], axis=0)
    price_indices = pd.concat([price_indices, pd.DataFrame(
                                                           np.multiply(price_indices.loc["Price index base",:].to_frame().T.values, database.X.T.values) / database.X.sum().sum(),
                                                           index = ['Price index base - weighted total'],
                                                           columns = price_indices.columns
                                                           )], axis=0) 
    p_weight_by_region = pd.DataFrame(
                                      np.zeros((1,database.e.shape[1])),
                                      index = ['Price index base - weighted by region'],
                                      columns = database.e.columns
                                     )
    total_price_index_by_reg = pd.DataFrame(
                                          np.zeros((1,len(database.get_index('Region')))),
                                          index = ['Total price index by region'],
                                          columns = database.get_index('Region')
                                         )
    
    for region in database.get_index('Region'):
        p_weight_by_region.loc['Price index base - weighted by region', (region,slice(None),slice(None))] = \
            np.multiply(price_indices.loc["Price index base",(region,slice(None),slice(None))].to_frame().T.values[0], database.X.loc[(region,slice(None),slice(None)),:].T.values[0]) / database.X.loc[(region,slice(None),slice(None)),:].sum().sum()  
    
        total_price_index_by_reg.loc['Total price index by region',region] = p_weight_by_region.loc['Price index base - weighted by region', (region,slice(None),slice(None))].sum().sum()
    
    price_indices = pd.concat([price_indices, p_weight_by_region], axis=0)

    total_price_index = price_indices.loc['Price index base - weighted total',:].sum().sum()
    
        
    return price_indices, total_price_index, total_price_index_by_reg
 
#%%
def calc_tax_revenues(price_indices, database, X_totdem, ctax, z_filtered):
    
    tax_revenues = pd.DataFrame()
    
    for tax in ["PBA", "CBA"]:

        if tax == "PBA":
            p = pd.DataFrame(
                             database.e.values @ np.diagflat(ctax.loc[tax,:].values),
                             index = [tax],
                             columns = database.e.columns
                            )
            
        elif tax == "CBA":
            p = pd.DataFrame(
                             np.multiply((database.e.values @ z_filtered.values), ctax.loc[tax,:].values),
                             index = [tax],
                             columns = database.e.columns
                            )
        
        tax_revenues = tax_revenues.append(pd.DataFrame((p.values @ np.diagflat(database.X.values)),
                                                        index=p.index,
                                                        columns=p.columns))
        
    return tax_revenues
       
#%%
def competition_among_imports(price_indices, ctax, database, emissions):

    price_competition = {}
    
    price_competition['Values'] = pd.DataFrame(
                                               np.zeros((len(database.get_index('Region')),database.e.shape[1])),
                                               index = database.get_index('Region'),
                                               columns = database.e.columns,
                                              )
    price_competition['Differences'] = copy.deepcopy(price_competition['Values'])
    
    
    for region1 in database.get_index('Region'):
        for region2 in database.get_index('Region'):
            
            if region1 == region2:
                price_competition['Values'].loc[region1,(region2,slice(None),slice(None))] = price_indices.loc['Price index base',(region1,slice(None),slice(None))].values
            else:
                price_competition['Values'].loc[region1,(region2,slice(None),slice(None))] = (ctax.loc["CBA",(region2,slice(None))].to_frame().T.values * database.e.loc["CO2",(region1,slice(None),slice(None))].to_frame().T.values + 1)[0]
                
                
                
                # price_indices.loc['Price index base',(region1,slice(None),slice(None))].values + \
                #                                                                               np.multiply(emissions.loc["Embeddied CO2 emissions (Y)",(region1,slice(None),slice(None))].values,
                #                                                                                           ctax.loc["CBA",(region2,slice(None))].values[0])

        
    for region1 in database.get_index('Region'):
        for region2 in database.get_index('Region'):

            if region1 != region2:
                price_competition['Differences'].loc[region1,(region2,slice(None),slice(None))] = np.divide((price_competition['Values'].loc[region1,(region2,slice(None),slice(None))].values - \
                                                                                                             price_competition['Values'].loc[region2,(region2,slice(None),slice(None))].values),
                                                                                                            price_competition['Values'].loc[region2,(region2,slice(None),slice(None))].values)
    
    price_competition['Differences'].sort_index(axis=0, inplace=True)    
    price_competition['Differences'].sort_index(axis=1,level=0, inplace=True)    
    price_competition['Values'].sort_index(axis=0, inplace=True)    
    price_competition['Values'].sort_index(axis=1,level=0, inplace=True)    
    
    return price_competition    

#%%
def subplot_grid(subplot_number, orientation="v"):

    if orientation == "v":
        j = 0
        n_cols = []
        for i in reversed(range(subplot_number + 1)):
            if int(math.sqrt(i) + 0.5) ** 2 == i:
                n_cols += [int(math.sqrt(i))]
            j += 1
        n_cols = n_cols[0]

        if int(math.sqrt(subplot_number) + 0.5) ** 2 == subplot_number:
            n_rows = n_cols
        else:
            n_rows = n_cols + int(math.ceil((subplot_number - n_cols ** 2) / n_cols))

    elif orientation == "h":
        j = 0
        n_rows = []
        for i in reversed(range(subplot_number + 1)):
            if int(math.sqrt(i) + 0.5) ** 2 == i:
                n_rows += [int(math.sqrt(i))]
            j += 1
        n_rows = n_rows[0]

        if int(math.sqrt(subplot_number) + 0.5) ** 2 == subplot_number:
            n_cols = n_rows
        else:
            n_cols = n_rows + int(math.ceil((subplot_number - n_rows ** 2) / n_rows))

    grid = [(row + 1, col + 1) for row in range(n_rows) for col in range(n_cols)]

    return (n_rows, n_cols, grid)
    
#%%
def plot_competition_heatmap(path, price_competition, simulation, template, key="Differences", orientation='v'):

    n_rows, n_cols, grid = subplot_grid(len(price_competition[key].index), orientation=orientation)
    fig = make_subplots(rows=n_rows,
                        cols=n_cols,
                        subplot_titles=["To {}".format(i) for i in list(price_competition[key].index)],
                        shared_xaxes = 'all',
                        shared_yaxes = 'all')
    
    counter = 0
    for region in list(price_competition[key].index):
        data = copy.deepcopy(price_competition[key]).loc[:,(region,slice(None),slice(None))]
        data.columns = list(data.columns.get_level_values(-1))
    
        fig.add_trace(go.Heatmap(x = list(data.columns), 
                                 y = list(data.index),
                                 z = data.values,
                                 colorscale="Viridis",),
                      row=grid[counter][0],
                      col=grid[counter][1])
        counter += 1

    
    fig.update_layout(title = "Prices of imported goods from all regions towards specic regions <br>Tax type: {}".format(simulation),
                      template = template,  
                      )
    fig.write_html(path, auto_open=False)

#%%
def plot_metabolism(decompositions_by_reg, accounts, categories, palette, template):
    
    sorted_data = copy.deepcopy(decompositions_by_reg)

    for account in accounts:
        fig = make_subplots(rows=1, 
                            cols=2, 
                            subplot_titles=["(a)","(b)"], 
                            specs=[[{"secondary_y": False}, {"secondary_y": True}]],
                            horizontal_spacing = 0.05,
                            shared_yaxes = True,)
    
            
        for cat in categories:        
    
            if cat in ["Imported","Exported"]:
                sorted_data[account].sort_values(by=('Net (excluding domestic)'), axis=1, ascending=True, inplace=True)    
    
                fig.add_trace(go.Bar(x = list(sorted_data[account].columns), 
                                     y = sorted_data[account].loc[cat,:].values,
                                     name = cat,
                                     legendgroup = cat,
                                     marker_color = palette[categories.index(cat)],),
                              row = 1,
                              col = 1)
            if cat == 'Net (excluding domestic)':
                sorted_data[account].sort_values(by=('Net (excluding domestic)'), axis=1, ascending=True, inplace=True)    
    
                fig.add_trace(go.Scatter(x = list(sorted_data[account].columns), 
                                         y = sorted_data[account].loc[cat,:].values,
                                         name = cat,
                                         legendgroup = cat,                                     
                                         marker_color = palette[categories.index(cat)],
                                         mode='markers',
                                         marker_size = 13),
                              row = 1,
                              col = 1)
                
            if cat == "Domestic":
                sorted_data[account].sort_values(by=('Net/Domestic'), axis=1, ascending=True, inplace=True)    
    
                fig.add_trace(go.Bar(x = list(sorted_data[account].columns), 
                                     y = sorted_data[account].loc[cat,:].values,
                                     name = cat,
                                     legendgroup = cat,
                                     marker_color = palette[categories.index(cat)],),
                              row = 1,
                              col = 2)
            if cat == 'Net/Domestic':
                sorted_data[account].sort_values(by=('Net/Domestic'), axis=1, ascending=True, inplace=True)    
    
                fig.add_trace(go.Scatter(x = list(sorted_data[account].columns), 
                                         y = sorted_data[account].loc[cat,:].values,
                                         name = cat,
                                         legendgroup = cat,                                     
                                         marker_color = palette[categories.index(cat)],
                                         mode='markers',
                                         marker_size = 13),
                              secondary_y=True,
                              row = 1,
                              col = 2)
    
                              
        fig.update_layout(barmode='relative',
                          template = template,
                          )
        fig.write_html(r"Plots\Trades\By region\{}.html".format(account), auto_open=False,)

#%%
def generate_carbon_taxes_simulations(taxed_regions,tax_mechanisms,carbon_price,database):
    
    carbon_taxes = {}
    for tax in tax_mechanisms:
        carbon_taxes[tax] = {} 
        for region in taxed_regions:
            
            if region != "Global tax":
                carbon_taxes[tax][region] = pd.DataFrame(
                                                         np.zeros((2, database.e.shape[1])),
                                                         index = ['PBA','CBA'],
                                                         columns = pd.MultiIndex.from_arrays([database.e.columns.get_level_values(0), database.e.columns.get_level_values(-1)])
                                                        )
                
                if tax == 'PBA' or tax=='CBA':
                    carbon_taxes[tax][region].loc[tax,(region,slice(None),slice(None))] += carbon_price
                elif tax == 'CBAM':
                    carbon_taxes[tax][region].loc[:,(region,slice(None),slice(None))] += carbon_price
            
            else:
                carbon_taxes[tax][region] = pd.DataFrame(
                                                         np.zeros((2, database.e.shape[1])),
                                                         index = ['PBA','CBA'],
                                                         columns = pd.MultiIndex.from_arrays([database.e.columns.get_level_values(0), database.e.columns.get_level_values(-1)])
                                                        )
                
                if tax == 'PBA' or tax=='CBA':
                    carbon_taxes[tax][region].loc[tax,:] += carbon_price
                elif tax == 'CBAM':
                    carbon_taxes[tax][region].loc[:,:] += carbon_price
                
    
    return carbon_taxes

