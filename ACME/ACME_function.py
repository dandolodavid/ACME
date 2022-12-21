import pandas as pd
import numpy as np
from scipy import stats
from ACME.utils import most_frequent

def build_feature_exploration_table(table, feature):
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
    feature_table = table.loc[feature].copy()
    
    feature_table['direction'] = 'lower'
    feature_table.loc[feature_table['predict'] >= feature_table['baseline_prediction'],'direction'] = 'upper'

    feature_table['effect'] = np.abs(feature_table['baseline_prediction'] - feature_table['predict']) 
    feature_table['effect'] = feature_table['effect']*feature_table['direction'].map({'lower':-1,'upper':1})

    return feature_table

def create_quantile_feature_matrix(dataframe, feature, K, robust = False, local = None):
    '''
    Create a matrix with K row, made by all the columns dataframe mean values, except for the 'feature', which is replaced by K quantiles of the feature empiracal distribution.

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
        x_base = dataframe.mean()
    else:
        x_base = dataframe.loc[local]

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
    
    quantile = np.sort(list(quantile) + [stats.percentileofscore(dataframe[feature],x_base[feature])/100])

    x_j = dataframe[feature].values
    x_j_k = np.quantile(x_j,quantile)
    
    Z = pd.DataFrame( x_base ).T
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
        quantile = dataframe.loc[ dataframe['original'] == local_value,  'quantile']
    else:
        quantile = dataframe.loc[ dataframe['original'] == original_list[np.argmin(np.abs(original_list - local_value))],  'quantile']
    
    return quantile.values[0]

def computeACME(model, dataframe, features, numeric_df, cat_df, label, task, local, K, 
                robust = False, class_to_analyze = None, score_function = None ):
    '''
    Params:
    -------
    - model: object
        the object of the model
    - dataframe: pd.DataFrame
        the dataframe used to train the model or to predict
    - features: [str] 
        list of the features name in SAME ORDER as the the model input
    - numeric_df: pd.DataFrame
        splitted dataframe with only numerical (numeric) features
    - cat_df: pd.DataFrame
        splitted dataframe with only numerical (numeric) features
    - label: str
        name of the target_feature
    - task: str
        type of model task {'regression','reg','r', 'c','class','classification', 'ad','anomaly detection'}
    - local: int, str
        index of the local observation
    - K: int
        number of quantiles to use
    - robust : bool (default False)
        if uses the range [0.05, 0.95] for the quantiles instead fo [0,1]
    - class_to_analyze: int, str (default None)
        class to analyze in case of classification. If None the entire classification system is analyzed. Must be specified in case of local AcME
    - score_function: fun(model, x)
        a function that has in input the model and the input data to realize the prediction. It must return a numeric score

    Returns:
    -------
    - table : pd.DataFrame
        table with computation done on each feature
    - importance_table : pd.DataFrame
        table with importance score
    - baseline_pred : float
        predictions on the baseline
    - baseline : pd.DataFrame
        baseline feature values
    '''    

    # for all features containers 
    table = pd.DataFrame()
    importance_table = pd.DataFrame()

    # divide numeric and cat feature
    if type(numeric_df) != type(None): #chek if the dataframe is empty
        numeric_features = numeric_df.columns.to_list()
    else:
        numeric_features = []
    if type(cat_df) != type(None): #chek if the dataframe is empty
        cat_features = cat_df.columns.to_list()
    else:
        cat_features = []

    # for every feature, we compute the predictions based on the feature quantiles
    # create the variable-quantile matrix
    for feature in features:
        
        # containers 
        features_df = pd.DataFrame()
        features_local_df = pd.DataFrame()
       
        ## ------------------
        ## find the baseline
        ## ------------------

        if local is None:
            # if there are numeric features take the mean, else set to an empty df
            if len(numeric_features) > 0:
                numeric_baseline = pd.DataFrame(numeric_df.mean()).T
            else : 
                numeric_baseline = pd.DataFrame()
            # if there are categoric features take the mode, else set to an empty df
            if len(cat_features) > 0:
                cat_baseline = cat_df.mode()
            else : 
                cat_baseline = pd.DataFrame()
        else:
            # if there are numeric features take the mean, else set to an empty df
            if len(numeric_features) > 0:
                numeric_baseline = pd.DataFrame(numeric_df.loc[local]).T
            else : 
                numeric_baseline = pd.DataFrame()
            # if there are categoric features take the mode, else set to an empty df
            if len(cat_features) > 0:
                cat_baseline = pd.DataFrame(cat_df.loc[local]).T
            else : 
                cat_baseline = pd.DataFrame()

        # save the baseline
        baseline = pd.DataFrame(pd.concat([numeric_baseline, cat_baseline],axis=1))[features]

        ## --------------------------------------------------------------------------------------------------------------------------------
        ## for every numeric feature, we compute the predictions based on the feature quantiles and create the variable-quantile matrix
        ## for every cat feature, we compute the predictions based on the feature levels and create the variable-levels matrix         
        ## --------------------------------------------------------------------------------------------------------------------------------
        
        if feature in cat_features:
            Z_numeric = numeric_baseline
            if cat_features != []:
                Z_cat = create_level_variable_matrix( cat_df, feature,local = local) 
                Z = pd.concat( [ Z_numeric.loc[Z_numeric.index.repeat(len(Z_cat))].reset_index(drop=True) , Z_cat.reset_index(drop=True) ] , axis = 1 )
            else:
                Z = Z_numeric
        else:
            Z_numeric = create_quantile_feature_matrix( numeric_df, feature, K, local = local, robust = robust )
            if cat_features != []:
                Z_cat = cat_baseline
                Z = pd.concat( [ Z_cat.loc[Z_cat.index.repeat(len(Z_numeric))].reset_index(drop=True), Z_numeric.reset_index(drop=True) ] , axis = 1  )
            else:
                Z = Z_numeric
        
        ## ------------------------------------
        ##  Get baseline predictions for task
        ## ------------------------------------

        if score_function:
            # if the score function is available
            predictions = score_function( model, Z.drop(columns='quantile')[features] )
            baseline_pred = score_function( model, baseline)[0]
            
        elif task in ['r','reg','regression'] or task in ['ad','anomaly detection']:
            # baseline prediction
            baseline_pred = model.predict(baseline)[0]
            try:
                if len(baseline_pred) == 2:
                    baseline_pred = baseline_pred[0]
            except:
                pass
            
            # prediciton
            predictions = model.predict(Z.drop(columns='quantile')[features])
            try:
                if predictions.shape[1] == 2:
                    predictions = predictions[:,0]
            except:
                pass
            
        elif task in ['c','class','classification']:

            # mean prediction
            baseline_pred = model.predict_proba(baseline)[0][class_to_analyze]
            # prediciton
            try:
                predictions = model.predict_proba(Z.drop(columns='quantile')[features])[:,class_to_analyze]
            except:
                predictions = model.predict_proba(Z.drop(columns='quantile')[features])[class_to_analyze]

        ## ------------------------------------------------------------------------------------------------
        ## build the dataframe with the standardize_effect, the predictions and the original effects
        ## ------------------------------------------------------------------------------------------------

        if local:
            features_df['effect'] = predictions
        else:    
            effects = predictions - baseline_pred
            features_df['effect'] = ( effects - np.mean(effects) ) / np.sqrt( np.var(effects)+0.0001 ) * ( max(predictions) - min(predictions) )

        features_df['predict'] = predictions
        features_df['baseline_prediction'] = baseline_pred
        features_df['original'] = Z[feature].values
        features_df['quantile'] = Z['quantile'].values
    
        if task  in ['c','class','classification']:
            features_df['class'] = class_to_analyze

        if feature in cat_features:
            features_df['type_feature'] = 'categorical'
        else:
            features_df['type_feature'] = 'numeric'

        near_quantile = nearest_quantile(features_df, baseline[feature].values[0], feature in cat_features)
        features_df['baseline_quantile'] = near_quantile

        features_df['size'] = 0.1 if local else 0.05
        features_df.loc[features_df['quantile'] == near_quantile,'size'] = 0.5 if local else 0.05
       
        features_df.index = np.repeat(feature, len(predictions))
        features_df.index.name = 'feature'
        table = pd.concat([table, features_df])
    
    # calculate importance and merge with the table
    if local:
        importance_table = None
    else:
        importance_table = table[['effect']].abs().reset_index().groupby('feature')[['effect']].mean().sort_values('effect',ascending=False).rename(columns={'effect':'importance'})
        table = table.merge(importance_table, how = 'left', right_index=True, left_index=True)

    #### TBD : at the moment, importance given in local interpretability and saved in local table is not the same obtained by global interpretability
    return table, importance_table, baseline_pred, baseline

