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

#def calculate_quantile_position(dataframe,column,local):

#    from statsmodels.distributions.empirical_distribution import ECDF

#    ecdf = ECDF(dataframe[column])
#    return ecdf(dataframe.loc[local,column])    
