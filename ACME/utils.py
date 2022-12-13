import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter 
import plotly.express as px

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
    Create a matrix with K row, made by all the columns dataframe mean values, except for the "feature", which is replaced by K quantiles of the feature empiracal distribution.

    Params:
    ------
    - dataframe : pd.DataFrame
        input dataframe
    - feature : str
        name of the feature
    - K : int
        number of quantile to use
    - robust : bool
        if True then use only quantile from 0.05 to 0.95, if false use quantile from 0 to 1
    - local : int,str (default = None)
        if valorized the use a single observation instead of the mean as baseline

    Returns:
    -------
    - Z : pd.DataFrame
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
    '''
    Params:
    -------
    - dataframe : pd.DataFrame
        input dataframe
    - feaure : str
        feature name
    - local : int,str
        index of the local observation

    Returns:
    -------
    - Z : pd.DataFrame
    '''
    
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

def nearest_quantile(dataframe, local_value, cat_features):
    '''
    Find the local values nearest quantile in the importance table.

    Params:
    ------
    - dataframe : pd.DataFrame
        input dataframe
    - local_value : int,str
        local value
    - cat_features : bool
        True if the feature is a categorical features
    
    Returns:
    -------
    - quantile values : float
    '''

    original_list = dataframe['original'].unique()
    if cat_features:
        quantile = dataframe.loc[ dataframe.original == local_value,  'quantile']
    else:
        quantile = dataframe.loc[ dataframe.original == original_list[np.argmin(np.abs(original_list - local_value))],  'quantile']
    
    return quantile.values[0]

def plot_express(plot_df, meta):
    '''
    Function generating the plot

    Params:
    -------
    - plot_df : pd.DataFrame
        plot dataframe
    - meta : dict
        metadata with information required for the plot 
        * x : 
        * local : bool
            if local or global
        * task : str
            ACME task
        * base_line : 
        * y_bottom : str
            last feature's name of the plot
        * y_top : sr
            first feature's name of the plot

    Returns:
    -------
    '''
    
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
        if meta['task'] in ['r','reg','regression']:
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
        fig.update_layout( shapes = [dict(
                                          type="line", x0 = x, y0 = y_bottom, x1 = x, y1 = y_top, 
                                          line = dict(color = color_local , width = 2 ,dash = "dash" ) 
                                         )
                                    ])
    else:
        fig.update_layout(shapes=[dict( type="line", x0=x, y0=y_bottom, x1=x, y1=y_top, line = dict(color="black", width=2 ,dash="dash" ) )] )
    
    return fig



#def calculate_quantile_position(dataframe,column,local):

#    from statsmodels.distributions.empirical_distribution import ECDF

#    ecdf = ECDF(dataframe[column])
#    return ecdf(dataframe.loc[local,column])    
