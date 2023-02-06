import pandas as pd
import numpy as np
from scipy import stats
from functools import partial
from ACME.utils import most_frequent


def _percentileofscore(dataframe, feature):
    return partial( stats.percentileofscore, a = dataframe[feature] )

def get_exact_quantile(percentile_function, baseline_value):
    return percentile_function(score = baseline_value)/100

def compute_baseline_predictions(model, baseline, task, score_function, class_to_analyze):
    '''
    '''
    if score_function:
        # if the score function is available
        baseline_pred = score_function( model, baseline.values)[0]
        
    elif task in ['r','reg','regression'] or task in ['ad','anomaly detection']:
        # baseline prediction
        baseline_pred = model.predict(baseline.values)[0]
        
    elif task in ['c','class','classification']:
        # mean prediction
        try:
            if len(baseline_pred) == 2:
                baseline_pred = baseline_pred[0]
        except:
            pass   
        baseline_pred =  model.predict_proba(baseline.values)[0][class_to_analyze]

    return baseline_pred
    
def compute_predictions(model, Z, features, task, score_function, class_to_analyze):
    '''
    '''
    if score_function:
        # if the score function is available
        predictions = score_function( model, Z.drop(columns='quantile')[features].values )
        
    elif task in ['r','reg','regression'] or task in ['ad','anomaly detection']:
        # prediciton
        predictions = model.predict(Z.drop(columns='quantile')[features].values)
        try:
            if predictions.shape[1] == 2:
                predictions = predictions[:,0]
        except:
            pass
        
    elif task in ['c','class','classification']:
        # prediciton
        try:
            predictions = model.predict_proba(Z.drop(columns='quantile')[features].values)[:,class_to_analyze]
        except:
            predictions = model.predict_proba(Z.drop(columns='quantile')[features].values)[class_to_analyze]

    return predictions

def build_feature_exploration_table(table, feature):
    '''
    Params:
    ---------
    - table : pd.DataFrame
        table generate from computeACME
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

def create_quantile_feature_matrix(dataframe, feature, K, robust = False):
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

    Returns:
    -------
    - Z : pd.DataFrame
    '''
    
    x_base = dataframe.mean()

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
    
    percentile_function = _percentileofscore(dataframe,feature)
    baseline_exact_quantile = get_exact_quantile(percentile_function=percentile_function, baseline_value=x_base[feature])
    quantile = np.sort(list(quantile) + [baseline_exact_quantile])

    x_j = dataframe[feature].values
    x_j_k = np.quantile(x_j,quantile)
    
    Z = pd.DataFrame( x_base ).T
    Z = pd.concat([Z]*len(x_j_k), ignore_index=True)
    Z[feature] = x_j_k
    Z['quantile'] = quantile
   
    return Z, percentile_function

def create_level_variable_matrix(dataframe, feature):
    '''
    Params:
    -------
    - dataframe : pd.DataFrame
        input dataframe
    - feaure : str
        feature name

    Returns:
    -------
    - Z : pd.DataFrame
    '''
    
    x_j_k = np.unique(dataframe[feature].to_list())
    x_most_freq = dataframe.apply(lambda x: most_frequent(x),axis=0)
    
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

def computeACME(model, features, numeric_df, cat_df, task, K, robust = False, class_to_analyze = None, score_function = None ):
    '''
    Params:
    -------
    - model: object
        the object of the model
    - features: [str] 
        list of the features name in SAME ORDER as the the model input
    - numeric_df: pd.DataFrame
        splitted dataframe with only numerical (numeric) features
    - cat_df: pd.DataFrame
        splitted dataframe with only numerical (numeric) features
    - task: str
        type of model task {'regression','reg','r', 'c','class','classification', 'ad','anomaly detection'}
    - K: int
        number of quantiles to use
    - robust : bool (default False)
        if uses the range [0.05, 0.95] for the quantiles instead fo [0,1]
    - class_to_analyze: int, str (default None)
        class to analyze in case of classification. If None the entire classification system is analyzed. 
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
    percentile_functions={}

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
       
        ## ------------------
        ## find the baseline
        ## ------------------

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
        
        # save the baseline
        baseline = pd.DataFrame(pd.concat([numeric_baseline, cat_baseline],axis=1))[features]

        ## --------------------------------------------------------------------------------------------------------------------------------
        ## for every numeric feature, we compute the predictions based on the feature quantiles and create the variable-quantile matrix
        ## for every cat feature, we compute the predictions based on the feature levels and create the variable-levels matrix         
        ## --------------------------------------------------------------------------------------------------------------------------------
        
        if feature in cat_features:
            Z_numeric = numeric_baseline
            if cat_features != []:
                Z_cat = create_level_variable_matrix( cat_df, feature) 
                Z = pd.concat( [ Z_numeric.loc[Z_numeric.index.repeat(len(Z_cat))].reset_index(drop=True) , Z_cat.reset_index(drop=True) ] , axis = 1 )
            else:
                Z = Z_numeric
        else:
            Z_numeric, perc_function = create_quantile_feature_matrix( numeric_df, feature, K, robust = robust )
            percentile_functions[feature] = perc_function
            if cat_features != []:
                Z_cat = cat_baseline
                Z = pd.concat( [ Z_cat.loc[Z_cat.index.repeat(len(Z_numeric))].reset_index(drop=True), Z_numeric.reset_index(drop=True) ] , axis = 1  )
            else:
                Z = Z_numeric
        
        ## ------------------------------------
        ##  Get baseline predictions for task
        ## ------------------------------------

        baseline_pred = compute_baseline_predictions(model=model, baseline=baseline, task=task, score_function=score_function, class_to_analyze = class_to_analyze)
        predictions = compute_predictions(model=model, Z=Z, features = features, task=task, score_function=score_function, class_to_analyze=class_to_analyze)

        ## ------------------------------------------------------------------------------------------------
        ## build the dataframe with the standardize_effect, the predictions and the original effects
        ## ------------------------------------------------------------------------------------------------
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

        features_df['size'] = 0.05
        
        features_df.index = np.repeat(feature, len(predictions))
        features_df.index.name = 'feature'
        table = pd.concat([table, features_df])
    
    # calculate importance and merge with the table
    importance_table = table[['effect']].abs().reset_index().groupby('feature')[['effect']].mean().sort_values('effect',ascending=False).rename(columns={'effect':'importance'})
    table = table.merge(importance_table, how = 'left', right_index=True, left_index=True)

    #### TBD : at the moment, importance given in local interpretability and saved in local table is not the same obtained by global interpretability
    return table, importance_table, baseline_pred, baseline, percentile_functions

def predictACME(model, series, features, percentile_functions, meta_table, task, score_function, class_to_analyze):
    '''
    '''

    # cat and num features
    cat_features = meta_table.loc[meta_table['type_feature']=='categorical'].index.unique().tolist()
    num_features = meta_table.loc[meta_table['type_feature']=='numeric'].index.unique().tolist()

    # prediction on baseline
    baseline = pd.DataFrame(series[features]).T
    baseline_pred = compute_baseline_predictions(model=model, baseline=baseline, task=task, score_function=score_function, class_to_analyze=class_to_analyze)
    
    # take the necessary info from the table
    predict_meta_table = meta_table[['original','quantile','type_feature','importance']].copy()
    
    # prepare containers
    Z_list = []
    meta_list = []

    # for each feature we are:
    # 1) compute the quantile associated to the local feature value
    # 2) generate the perturbated observation using the quantile learned in the fit procedure
    # 3) generating metadata
    # Then we are calculating the associated predictions and save the output in the format required for the plot

    for feature in features:
        feature_table = predict_meta_table.loc[feature]
        if feature in num_features:
            Z_list.append(baseline)
            meta_list.append({'feature':feature_table.index.unique()[0],
                            'quantile':get_exact_quantile(percentile_functions[feature],baseline[feature])[0],
                            'original':baseline[feature].values[0],
                            'type_feature':feature_table['type_feature'].unique()[0],
                            })

        for idx,row in feature_table.iterrows():
            tmp = baseline.copy()
            tmp[feature] = row['original']
            Z_list.append(tmp)
            meta_list.append({'feature':idx,
                              'quantile':row['quantile'],
                              'original':row['original'],
                              'type_feature':row['type_feature']
                              })
    
    
    Z = pd.concat(Z_list).reset_index(drop=True)
    table = pd.DataFrame.from_dict(meta_list)
    Z = pd.concat([Z,table],axis=1).drop(columns=['original'])
    predictions = compute_predictions(model=model, Z=Z, features=features, task=task, score_function=score_function, class_to_analyze=class_to_analyze)
    
    table['predict'] = predictions
    table['baseline_prediction'] = baseline_pred
    table['effect'] = table['predict'] - table['baseline_prediction']
    table['baseline_quantile'] = None
    table['class'] = class_to_analyze
    table['size'] = 0.05

    for feature in features:
        features_df = table.loc[table['feature']==feature]
        near_quantile = nearest_quantile(features_df, baseline[feature].values[0], feature in cat_features)
        table.loc[table['feature']==feature,'baseline_quantile'] = near_quantile
        table.loc[np.logical_and(table['quantile'] == near_quantile,
                                table['feature']==feature),'size'] = 0.3

    table = table.set_index('feature')
    # calculate importance and merge with the table
    importance_table = table[['effect']].abs().reset_index().groupby('feature')[['effect']].mean().sort_values('effect',ascending=False).rename(columns={'effect':'importance'})
    table = table.merge(importance_table, how = 'left', right_index=True, left_index=True)

    return table, importance_table, baseline_pred, baseline