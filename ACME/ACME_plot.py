import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def ACME_barplot_multicalss(importance_table, label_class):

    # generate container
    plot_df = pd.DataFrame()
    i = 0
    for label in label_class:
        tmp = pd.DataFrame(importance_table.iloc[:,i])
        tmp.columns = ['importance']
        tmp['class'] = str(label)
        plot_df = pd.concat([plot_df,tmp],axis=0)
        i+=1
    
    fig = px.bar(round(plot_df.iloc[::-1].reset_index().rename(columns={'index':'feature'}),3), 
                x='importance',y='feature', 
                color='class', 
                orientation='h', 
                hover_name="class",
                title='Overall Classification Importance')

    fig.update_traces(hovertemplate = 'Feature:<b>%{y}</b><br><br>Importance: %{x}<br>Class: %{hovertext}')

    return fig

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
        * baseline : 
        * y_bottom : str
            last feature's name of the plot
        * y_top : sr
            first feature's name of the plot

    Returns:
    -------
    '''
    color_scale =  ['midnightblue','lightskyblue','limegreen']
    label_x = 'Predict' if meta['local'] else 'Standardized effect'
    title = 'Local AcME: observation ID ' + str(meta['index']) + '. Predicted: ' + str(round(meta['x'],3)) if meta['local'] else 'AcME Global Importance'

    # set the bottom and the top of the plot
    y_bottom = meta['y_bottom']
    y_top = meta['y_top']

    # set title
    if meta['local']:
        if 'label_class' in meta.keys():
            title = title + ' ( label_class : ' + str(meta['label_class']) + ' )'
    else:
        if meta['task'] in ['r','reg','regression'] or meta['task'] in ['ad','anomaly detection']:
            title = title + ' : regression ' 
        else:
            title = title + ' : classification. Label_class : ' + str(meta['label_class'])       

    #set the color for the local score line
    if (meta['local'] is not None) and (meta['task'] in ['ad','anomaly detection']):
        color_local = 'red' if meta['x'] > 0 else 'blue' 
    else:
        color_local = 'black'
    
    # draw plot
    fig = px.scatter(round(plot_df,3), 
                        x = 'effect', y = 'feature', 
                        color = 'quantile', size = 'size' if meta['local'] else None, 
                        hover_data = ['original'],
                        color_continuous_scale = color_scale,
                        labels = {'effect':label_x.lower(),'feature':'feature'}, title = title)

    # add local score line
    shapes = [dict( 
                    type='line', 
                    x0 = meta['x'], y0 = y_bottom, 
                    x1 = meta['x'], y1 = y_top, 
                    line = dict(color = color_local, 
                                width = 2,
                                dash = 'dash')
                    )]
    # if anomaly detection add a line that marks the thresholds for state changing
    if meta['task'] in meta['task'] in ['ad','anomaly detection']:
        shapes = shapes + [dict( 
                                    type='line', 
                                    x0 = 0, y0 = y_bottom, 
                                    x1 = 0, y1 = y_top, 
                                    line = dict(color = 'black', 
                                                width = 2,
                                                dash = 'solid')
                                    )]

    fig.update_layout(shapes=shapes)
    
    # set hover template
    fig.update_traces(hovertemplate = 'Feature: <b>%{y}</b><br><br>' +  label_x + ': %{x}<br>Original value: %{customdata[0]}<br>Quantile: %{marker.color}')

    return fig.update_layout( title={ 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'} )


def feature_exploration_plot(table, feature, task):
    '''
    Generate the anomaly detection explain plot

    Params:
    ------
    - table : pd.DataFrame
        ° regression and classification : summary/local table 
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
    table = round(table,3)
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
                hovertemplate = 'Prediction: %{x}<br>Feature value: %{y}',
                name =  plot_meta['lower_trace']['name'], orientation='h')

    # add effects that pushes the score to normal state
    fig.add_bar(x = table.loc[table.direction == plot_meta['upper_trace']['value'],'effect'], 
                y = table.loc[table.direction == plot_meta['upper_trace']['value'],'original'].values, 
                base = table['baseline_prediction'].values[0],
                marker = dict(color = 'red'),
                hovertemplate = 'Prediction: %{x}<br>Feature value: %{y}',
                name =  plot_meta['upper_trace']['name'], orientation='h')

    # add a line that marks the actual state
    fig.add_scatter(y = [ table['original'].values[0]*0.9 ,table['original'].values[-1]*1.05 ],
                    x = [ actual_score,actual_score ], 
                    mode ='lines', 
                    hovertemplate = 'Actual '+ 'score' if task in ['ad','anomaly detection'] else 'prediction',
                    name = 'actual score', line = dict(color = color ,width=2,dash="dash") )

    # add a great point corrisponding to the the actual score
    fig.add_scatter( x = [actual_score],
                    y = [actual_values], mode='markers',
                    
                    hovertemplate = 'Actual feature value<br> '+ 'Prediction: %{x}<br>Value: %{y}',
                    marker = dict(size=20,color=color),  name = 'current value')
    
    # add a line that marks the thresholds for state changing
    if task in ['ad','anomaly detection']:
        fig.add_scatter( y = [ table['original'].values[0]*0.9, table['original'].values[-1]*1.05 ],
                         x = [ 0,0 ], 
                         mode='lines',
                         
                         hovertemplate = 'Changepoint',
                         line=dict(color="black",width=2),  name = 'change point')

    fig.update_layout(  title='Feature ' + str(feature), 
                        yaxis_title = 'feature values', 
                        xaxis_title = 'score', 
                        autosize=True )
    
    return fig


