import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
patterns = ['', '/', '|', 'x', '-', '\\', '+', '.']


def feature_importance_distribution_barplot(features_occurence_table : pd.DataFrame, title : str = None): 
    """
    Stacked barplot of the feature importance distribution organzed by position. 

    Args: 
        features_occurence_table (pd.DataFrame): table of dimension (n_features, n_features) where cell (i,j) represent the percentage of times feature j is ranked in position i. 
        title (str, optional): title of the plot. Defaults to None.
    
    Returns:
        fig (plotly.graph_objects.Figure): figure of the barplot plot
    """
    fig = go.Figure()

    for i, feature in enumerate(features_occurence_table.columns):
        fig.add_trace(go.Bar(
            x=features_occurence_table.index,
            y=features_occurence_table[feature],
            marker_color=colors[i % len(colors)],
            width=0.6,
            name=feature,
            marker_pattern_shape=patterns[i % len(patterns)],
            marker_pattern_size=6,
        ))

    fig.update_layout(
        barmode='stack',
        xaxis=dict(title='Ranking Position', tickmode='array', tickvals=features_occurence_table.index, tickfont=dict(size=18),  titlefont=dict(size=18)),
        yaxis=dict(title='Normalized Count', titlefont=dict(size=18), tickfont=dict(size=18)),
        title=dict(text=title, font=dict(size=18)) if title is not None else None,
        bargap=0.1,
        legend=dict(font=dict(size=18)), 
        legend_title='Feature',
    )
    
    return fig
  

def sub_scores_distributions(ratios, deltas, changes, distances_from_change, title=None):
    """
    Plot 3 boxplots for distributions of ratios, deltas, distances_from_change and a table with the number of changes for each feature.
    Args: 
        ratios: table of dimension (n_samples, n_features) where cell (i,j) represent the ratio of feature j for sample i
        deltas: table of dimension (n_samples, n_features) where cell (i,j) represent the delta of feature j for sample i
        changes: table of dimension (n_samples, n_features) where cell (i,j) represent the number of changes of feature j for sample i
        distances_from_change: table of dimension (n_samples, n_features) where cell (i,j) represent the distance from change of feature j for sample i
    
    Returns:
        fig (plotly.graph_objects.Figure): figure of the boxplots and the table
    """

    df_change = pd.DataFrame(changes.sum()).transpose()
    header_values = df_change.columns
    cell_values = df_change.values[0]

    fig = make_subplots(
        rows = 2, cols = 2, 
        specs = [[{"type": "box"}, {"type": "box"}], [{"type": "table"}, {"type": "box"}]],
        subplot_titles = ("Delta", "Ratio", "Number of state changes", "Distance to change"), 
        vertical_spacing = 0.15,
        horizontal_spacing = 0.05,
    )

    for i, feature in enumerate(deltas.columns):
        fig.add_trace(go.Box(
            y=deltas[feature],
            marker_color=colors[i % len(colors)],
            name=feature,
            showlegend=False,
        ), row=1, col=1)
    
    for i, feature in enumerate(ratios.columns):
        # Add a bar for each feature
        fig.add_trace(go.Box(
            y=ratios[feature],
            marker_color=colors[i % len(colors)],
            name=feature,
            showlegend = False,
        ), row=1, col=2)
    
    fig.add_trace(go.Table(
        # put each value in the cell of the corresponding color
        header=dict(values=header_values, height=30),
        cells = dict(values=cell_values, height=30, font = dict(color=colors)),
    ), row=2, col=1)


    for i, feature in enumerate(distances_from_change.columns):
        fig.add_trace(go.Box(
            y=distances_from_change[feature],
            marker_color=colors[i % len(colors)],
            name=feature,
        ), row=2, col=2)
    
    fig.update_layout(width=10*1.5*100, height = 6*1.5*100, font = dict(size=18))
    fig.update_annotations(font_size=18)

    return fig