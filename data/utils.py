import pandas as pd
import numpy as np


def create_quantile_feature_matrix(dataframe, feature, target, K, local = None):
    '''
    Create a matrix with K row, made by all the columns dataframe mean values, except for the "feature", which is replaced by K quantiles of the feature empiracal distribution
    '''
    if K > 2:
        quantile = np.linspace(0,1,K)
    if K == 2:
        quantile = [ 0.01, 0.99 ]
    x_j = dataframe[feature].values
    x_j_k = np.quantile(x_j,quantile)
    if local is None:
        x_mean = dataframe.drop(columns = [target]).mean()
    else:
        x_mean = dataframe.drop(columns = [target]).loc[local]
    Z = pd.DataFrame( x_mean ).T
    Z = pd.concat([Z]*len(x_j_k), ignore_index=True)
    Z[feature] = x_j_k
    Z['quantile'] = quantile
    
    return Z

def plot_express(plot_df, meta):
    from data_science.plot.plotly_base import PlotlyBase
    import plotly.express as px
    
    x = meta['x']
    y_bottom = meta['y_bottom']
    y_top = meta['y_top']

    fig = px.scatter(plot_df, x="effect", y='feature', color="quantile", color_continuous_scale = ['royalblue','red'])
    fig.update_layout(shapes=[dict( type="line", x0=x, y0=y_bottom, x1=x, y1=y_top, line = dict(color="black", width=2 ,dash="dash" ) )] )
    
    #title = ' ACME'

    #fig.update_layout(title_text = title )
    
    return fig