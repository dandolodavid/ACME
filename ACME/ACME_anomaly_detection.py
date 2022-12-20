import pandas as pd
import numpy as np
from ACME.ACME_plot import ACME_summary_plot, feature_exploration_plot

def build_anomaly_detection_feature_exploration_table(local_table, feature):
    '''
    Params:
    ---------
    - local_table : pd.DataFrame
        local table generate from computeACME
    - feature: str
        name of the feature to explore

    Returns:
    --------
    - feature_table : pd.DataFrame
    '''
    # calculate the effect 
    feature_table = local_table.loc[feature].copy()
    feature_table['direction'] = 'normal'
    feature_table.loc[feature_table['predict'] > 0,'direction'] = 'anomalies'
    feature_table['effect'] = np.abs(feature_table['baseline_prediction'] - feature_table['predict']) * np.sign(-2*(feature_table['baseline_prediction']>feature_table['predict']).astype(int)+1)

    return feature_table

def computeAnomalyDetectionImportance(local_table, weights={}):
    '''
    Compute the anomaly detection importance.

    Params:
    -------
    - local_table: pd.DataFrame
        local table generate from computeACME
    - weights : dict 
        Dictionary with the importance fo each element. Sum must be 1
        * ratio : float
            importance of local score position 
        * distance : float
            importance of interquanitle distance necessary to change 
        * change : float
            importance of the possibility to change prediction
        * delta : float
            importance of the score delta

    Returns:
    -------
    - importance_df: pd.DataFrame
    '''

    importance = {}

    if len(weights.keys())==4:
        if not sum(list(weights.values())) == 1:
            raise AttributeError('weights value must sum to 1')
    else:
        print('Using default weights for anomaly detection feature importance')
        weights = {'ratio':0.2,
            'distance':0.2,
            'change':0.3,
            'delta':0.3}
    
    # for each specific feaure
    for feature in local_table.index.unique():
        
        importance[feature] = {}

        tmp = local_table.loc[feature]

        # search when the sign changes
        tmp['sign_change'] = (np.sign(tmp['baseline_prediction']) != np.sign(tmp['predict'])).astype(int)
        tmp = tmp.reset_index(drop=True)
        # calculate quantile distance
        tmp['quantile_distance'] = np.abs(tmp['quantile']-tmp['baseline_quantile'])

        # local actual score
        local_score = tmp['baseline_prediction'].values[0]

        # min and max score possible altering the current feature values
        min_score = min(tmp['predict'].min(),local_score)
        max_score = max(tmp['predict'].max(),local_score)

        #delta of the score
        delta = np.abs( max_score - min_score )

        #ratio 
        ratio = (local_score - min_score)/delta
        
        # change
        if np.sign(min_score) !=  np.sign(max_score):
            change=1
        else:
            change=0
        
        # number of quantile required to change the state 
        if change == 1:
            distance = tmp.loc[tmp['sign_change']==1,'quantile_distance'].min() 
        else:
            distance = 1

        importance[feature]['ratio'] = ratio
        importance[feature]['delta'] = delta
        importance[feature]['change'] = change
        importance[feature]['distance_to_change'] = distance
        importance[feature]['max_score'] = max_score
        importance[feature]['min_score'] = min_score
        importance[feature]['local_score'] = local_score
    
    importance_df = pd.DataFrame.from_records(importance).T

    # weighted linear combination of each store
    importance_df['importance'] = importance_df['delta'] * weights['delta'] + importance_df['ratio'] * weights['ratio'] + importance_df['change'] * weights['change'] + (1-importance_df['distance_to_change']) * weights['distance']
    # sorting
    importance_df = importance_df.sort_values('importance',ascending=False)

    return importance_df
    


