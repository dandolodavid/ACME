import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def ACME_summary_plot(plot_df, meta):
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
        label_x = 'predict'
        color_scale = ['royalblue','red']
        title = 'Local AcME: observation ID ' + str(meta['index']) + '. Predicted: ' + str(round(x,3))
        if 'label_class' in meta.keys():
            title = title + ' ( label_class : ' + str(meta['label_class']) + ' )'
        fig = px.scatter(plot_df, x="effect", y='feature', color="quantile", size = 'size', hover_data=['original'],
                        color_continuous_scale = color_scale,labels = {'effect':label_x,'feature':'feature'}, title = title)
    else:
        label_x = 'standardized effect'
        color_scale = ['royalblue','red']
        title = 'AcME Global Importance'
        if meta['task'] in ['r','reg','regression'] or meta['task'] in ['ad','anomaly detection']:
            title = title + ' : regression ' 
        else:
            title = title + ' : classification. Label_class : ' + str(meta['label_class']) 
        fig = px.scatter(plot_df, x="effect", y='feature', color="quantile", hover_data=['original'],
                       color_continuous_scale = color_scale,
                       labels = {'effect':label_x,'feature':'feature'}, title = title)

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



def feature_exploration_plot(table, feature, task):
    '''
    Generate the anomaly detection explain plot

    Params:
    ------
    - table : pd.DataFrame
        ° regressiona nd classification : summary/local table 
        ° anomaly detection : importance table generated from the "build_anomaly_detection_feature_exploration_table" function for task 
    - feature : str
        name of the feature
    - task : str
        acme task

    Returns:
    -------
    - fig : plotly.figure
    '''

    # generate figure
    fig = go.Figure()

    # score and values of the baseline prediction (if local the baseline is the local observation)
    actual_score = table['baseline_prediction'].values[0]
    actual_values = table.loc[table['quantile'] == table['baseline_quantile'].values[0], 'original'].values[0]

    # set colors based on task
    if task in ['ad','anomaly detection']:
        color = 'red' if actual_score > 0 else 'blue'
    else: 
        color = 'black'

    # set the dictionary with the names to use in the plot
    plot_meta = {}
    if task in ['ad','anomaly detection']:
        plot_meta['lower_trace'] = {'name' : 'normal', 'value':'normal'}
        plot_meta['upper_trace'] = {'name' : 'anomalies', 'value':'anomalies'}
    else:
        plot_meta['lower_trace'] = {'name' : 'lower', 'value':'lower'}
        plot_meta['upper_trace'] = {'name' : 'upper', 'value':'upper'}

    # add effects that pushes the score to anomaly state
    fig.add_bar(x = table.loc[table.direction ==  plot_meta['lower_trace']['value'],'effect'], 
                y = table.loc[table.direction ==  plot_meta['lower_trace']['value'],'original'].values, 
                base = table['baseline_prediction'].values[0], 
                marker = dict(color = 'blue'), 
                name =  plot_meta['lower_trace']['name'], orientation='h')

    # add effects that pushes the score to normal state
    fig.add_bar(x = table.loc[table.direction == plot_meta['upper_trace']['value'],'effect'], 
                y = table.loc[table.direction == plot_meta['upper_trace']['value'],'original'].values, 
                base = table['baseline_prediction'].values[0],
                marker = dict(color = 'red'), 
                name =  plot_meta['upper_trace']['name'], orientation='h')

    # add a line that marks the actual state
    fig.add_scatter(y = [ table['original'].values[0]*0.9 ,table['original'].values[-1]*1.05 ],
                    x = [ actual_score,actual_score ], mode ='lines',
                    name = 'actual score', line = dict(color = color ,width=2,dash="dash") )

    # add a great point corrisponding to the the actual score
    fig.add_scatter( x = [actual_score],
                    y = [actual_values], mode='markers',
                    marker = dict(size=20,color=color),  name = 'current value')
    
    # add a line that marks the thresholds for state changing
    if task in ['ad','anomaly detection']:
        fig.add_scatter( y = [ table['original'].values[0]*0.9, table['original'].values[-1]*1.05 ],
                         x = [ 0,0 ], 
                         mode='lines',
                         line=dict(color="black",width=2),  name = 'change point')

    fig.update_layout(  title='Feature ' + str(feature), 
                        yaxis_title = 'feature values', 
                        xaxis_title = 'score', 
                        autosize=True )
    
    return fig


