import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter 

def most_frequent(List): 
    occurence_count = Counter(List) 
    return occurence_count.most_common(1)[0][0]

def clean_list(List,to_keep):
    out = List.copy()
    for i in List:
        if i not in to_keep:
            out.remove(i)
    return out

def create_quantile_feature_matrix(dataframe, feature, K, robust = False, local = None):
    '''
    Create a matrix with K row, made by all the columns dataframe mean values, except for the "feature", which is replaced by K quantiles of the feature empiracal distribution
    '''
    if local is None:
        x_mean = dataframe.mean()
    else:
        x_mean = dataframe.loc[local]

    if robust:
        min_q = 0.05
        max_q = 0.95
    else:
        min_q = 0
        max_q = 1
    
    if K > 2:
        quantile = np.linspace(min_q,max_q,K-1)
    if K == 2:
        quantile = [ min_q, max_q ]
    
    quantile = np.sort(list(quantile) + [stats.percentileofscore(dataframe[feature],x_mean[feature])/100])

    x_j = dataframe[feature].values
    x_j_k = np.quantile(x_j,quantile)
    
    Z = pd.DataFrame( x_mean ).T
    Z = pd.concat([Z]*len(x_j_k), ignore_index=True)
    Z[feature] = x_j_k
    Z['quantile'] = quantile
   
    return Z

def create_level_variable_matrix(dataframe, feature, local=None):
    
    x_j_k = np.unique(dataframe[feature].to_list())
    if local is None:
        x_most_freq = dataframe.apply(lambda x: most_frequent(x),axis=0)
    else:
        x_most_freq = dataframe.loc[local]
    
    Z = pd.DataFrame( x_most_freq ).T
    Z = pd.concat([Z]*len(x_j_k), ignore_index=True)
    Z[feature] = x_j_k
    Z['quantile'] = np.linspace(0,1,len(x_j_k))
    
    return Z

def calculate_quantile_position(dataframe,column,local):

    from statsmodels.distributions.empirical_distribution import ECDF

    ecdf = ECDF(dataframe[column])
    return ecdf(dataframe.loc[local,column])    

def nearest_quantile(dataframe, local_value, cat_features):

    original_list = dataframe['original'].unique()
    if cat_features:
        quantile = dataframe.loc[ dataframe.original == local_value,  'quantile']
    else:
        quantile = dataframe.loc[ dataframe.original == original_list[np.argmin(np.abs(original_list - local_value))],  'quantile']
    
    return quantile.values[0]

def plot_express(plot_df, meta):
    import plotly.express as px
    
    x = meta['x']
    
    if meta['local']:
        label_x = 'prediction'
        color_scale = ['royalblue','red']
        title = 'Local AcME: observation ID ' + str(meta['index']) + '. Predicted: ' + str(round(x,3))
        if 'label_class' in meta.keys():
            title = title + ' ( label_class : ' + str(meta['label_class']) + ' )'
        fig = px.scatter(plot_df, x="effect", y='feature', color="quantile", size = 'size', hover_data=['original'],
                        color_continuous_scale = color_scale,labels = {'effect':label_x,'feature':'Feature'}, title = title)
    else:
        label_x = 'standardized effect'
        color_scale = ['royalblue','red']
        title = 'AcME Global Importance'
        if meta['task'] == 'r' or meta['task'] == 'reg' or meta['task'] == 'regression':
            title = title + ' : regression ' 
        else:
            title = title + ' : classification. Label_class : ' + str(meta['label_class']) 
        fig = px.scatter(plot_df, x="effect", y='feature', color="quantile", hover_data=['original'],
                       color_continuous_scale = color_scale,
                       labels = {'effect':label_x,'feature':'Feature'}, title = title)

    y_bottom = meta['y_bottom']
    y_top = meta['y_top']
    
    if meta['local']:
        if x > meta['base_line']:
            color_local = 'red'
        else:
            color_local = 'blue' 
        fig.update_layout( shapes = [dict( type="line", x0 = x, y0 = y_bottom, x1 = x, y1 = y_top, line = dict(color = color_local , width = 2 ,dash = "dash" ) ),
                                    #dict( type="line", x0 = meta['base_line'], y0 = y_bottom, x1 = meta['base_line'], y1 = y_top, line = dict(color="black", width=2 ,dash="dash" ) ) 
                                    ] )
    else:
        fig.update_layout(shapes=[dict( type="line", x0=x, y0=y_bottom, x1=x, y1=y_top, line = dict(color="black", width=2 ,dash="dash" ) )] )
    
    return fig