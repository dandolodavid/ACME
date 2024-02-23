import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def plot_anomaly_dataset_2d(data, target, features=["x0", "x1"], title=None, marker_size=3): 
    """
    Plot the dataset with the target and the features in 2D.
    inliers are in tableau:blue of the balance color map, outliers are in tableau:orage.
    """
    if target not in data.columns:
        raise ValueError("target not in data.columns")
    
    if len(features) != 2:
        raise ValueError("len(features) != 2")
    
    inliers = data[data[target] == 0]
    outliers = data[data[target] == 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=inliers[features[0]], 
        y=inliers[features[1]],
        mode='markers',
        marker=dict(size=marker_size, color='#1170aa'), 
        name='inliers'))
    
    fig.add_trace(go.Scatter(
        x=outliers[features[0]], 
        y=outliers[features[1]],
        mode='markers',
        marker=dict(size=marker_size, color = "#fc7d0b"), 
        name='outliers'))
    
    # make quare image and equal axes
    fig.update_layout(width=800, height=800, yaxis_scaleanchor="x", yaxis_scaleratio=1, xaxis_scaleanchor="y", xaxis_scaleratio=1)

    # add labels to axis featurs
    fig.update_xaxes(title_text=features[0])
    fig.update_yaxes(title_text=features[1])


    if title is not None:
        fig.update_layout(title=title, xaxis_title=features[0], yaxis_title=features[1])
    
    return fig


def plot_anomaly_dataset_3d(data, target, features=["x0", "x1", "x2"], title=None, marker_size=3): 
    """
    Plot the dataset with the target and the features in 3D.
    """
    if target not in data.columns:
        raise ValueError("target not in data.columns")
    
    if len(features) != 3:
        raise ValueError("len(features) != 3")
    
    inliers = data[data[target] == 0]
    outliers = data[data[target] == 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=inliers[features[0]], 
        y=inliers[features[1]], 
        z=inliers[features[2]],
        mode='markers',
        marker=dict(size=3, color='#1170aa'), 
        name='inliers'))
    
    fig.add_trace(go.Scatter3d(
        x=outliers[features[0]], 
        y=outliers[features[1]], 
        z=outliers[features[2]],
        mode='markers',
        marker=dict(size=marker_size, color = "#fc7d0b" ), 
        name='outliers'))
    
    fig.update_layout(scene=dict(xaxis_title=features[0], yaxis_title=features[1], zaxis_title=features[2]))
    if title is not None:
        fig.update_layout(title=title)
    fig.show()





def plot_anomaly_score_2d(data, target, features=["x0", "x1"], title=None, cmap="balance", marker_size=3): 
    """
    Plot the anomaly scores with the target and the features in 2D.
    """
    if target not in data.columns:
        raise ValueError("target not in data.columns")
    
    if len(features) != 2:
        raise ValueError("len(features) != 2")
    
    # if anomaly score > 0.5 plot a x instead of a circle
    data["marker"] = np.where(data[target] > 0.5, "x", "circle")
    fig = px.scatter(data, x=features[0], y=features[1], color=target, title=title,
                     color_continuous_scale=cmap, color_continuous_midpoint=0.5, symbol="marker")
    fig.update_traces(marker=dict(size=marker_size))

    fig.update_layout(width=800, height=800, yaxis_scaleanchor="x", yaxis_scaleratio=1, xaxis_scaleanchor="y", xaxis_scaleratio=1)
    fig.for_each_trace(lambda t: t.update(name="Outlier" if t.name == "x" else "Inlier"))
    # Move the legend outside the plot area to the right
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    
    fig.show()



def plot_anomaly_score_3d(data, target, features=["x0","x1","x2"], title = None, cmap = "RdBu_r", marker_size=3): 
    """
    Plot the anomaly scores with the target and the features in 3D.
    """
    if target not in data.columns:
        raise ValueError("target not in data.columns")
    
    if len(features) != 3:
        raise ValueError("len(features) != 3")
    
    fig = px.scatter_3d(data, x=features[0], y=features[1], z=features[2], color=target, title=title, color_continuous_scale=cmap)
    fig.update_traces(marker=dict(size=marker_size))
    fig.show()
