import pandas as pd
import numpy as np
from ACME.utils import create_quantile_feature_matrix, create_level_variable_matrix, most_frequent, calculate_quantile_position, nearest_quantile

def _computeACME(model, dataframe, features, numeric_df, cat_df, importance_table, label, task, local, K, robust = False, class_to_analyze = None, 
                table = None, local_table = None, score_function = None ):
    '''
    Parameters:
    ----------
    - model: object
        the object of the model
    - dataframe: pd.DataFrame
        the dataframe used to train the model or to predict
    - features: [str] 
        list of the features name in SAME ORDER as the the model input
    - numeric_df: pd.DataFrame
        splitted dataframe with only numerical (quantitative) features
    - cat_df: pd.DataFrame
        splitted dataframe with only numerical (quantitative) features
    - importance_table: pd.DataFrame
        dataframe with as index the name of the feature
    - label: str
        name of the target_feature
    - task: str
        type of model task {'regression','reg','r','c','class','classification'}
    - local: int, str
        index of the local observation
    - K: int
        number of quantiles to use
    - robust : bool (default False)
        if uses the range [0.05, 0.95] for the quantiles instead fo [0,1]
    - class_to_analyze: int, str (default None)
        class to analyze in case of classification. If None the entire classification system is analyzed. Must be specified in case of local AcME
    - score_function: function
        function used to compute the predictions
    - table : pd.DataFrame()
        empty df
    - local_table : pd.DataFrame()
        empty df
    - score_function: fun(model, x)
        a function that has in input the model and the input data to realize the prediction. It must return a numeric score

    Returns:
    ----------
    - table : pd.DataFrame
    - importance_table : pd.DataFrame
    '''    

    # for every feature, we compute the predictions based on the feature quantiles
    # create the variable-quantile matrix
    if type(numeric_df) != type(None): #chek if the dataframe is empty
        numeric_features = numeric_df.columns.to_list()
    else:
        numeric_features = []
    if type(cat_df) != type(None): #chek if the dataframe is empty
        cat_features = cat_df.columns.to_list()
    else:
        cat_features = []

    for feature in importance_table.index:

        df = pd.DataFrame()
        local_df = pd.DataFrame()
    
        if local is None:
            
            #for every quantitative feature, we compute the predictions based on the feature quantiles and create the variable-quantile matrix
            #for every qualitative feature, we compute the predictions based on the feature levels and create the variable-levels matrix         
            
            if feature in cat_features:
                Z_quantitative = pd.DataFrame( numeric_df.mean() ).T
                if cat_features != []:
                    Z_qualitative = create_level_variable_matrix( cat_df, feature,local = None) 
                    Z = pd.concat( [ Z_quantitative.loc[Z_quantitative.index.repeat(len(Z_qualitative))].reset_index(drop=True) , Z_qualitative.reset_index(drop=True) ] , axis = 1 )
                else:
                    Z = Z_quantitative
            else:
                Z_quantitative = create_quantile_feature_matrix( numeric_df, feature, K, local = None, robust = robust )
                if cat_features != []:
                    Z_qualitative = pd.DataFrame( cat_df.apply(lambda x: most_frequent(x), axis=0) ).T
                    Z = pd.concat( [ Z_qualitative.loc[Z_qualitative.index.repeat(len(Z_quantitative))].reset_index(drop=True), Z_quantitative.reset_index(drop=True) ] , axis = 1  )
                else:
                    Z = Z_quantitative

            x_mean = pd.DataFrame( numeric_df.mean() ).T

            if cat_features != []:
                x_most_freq = pd.DataFrame( cat_df.apply( lambda x : most_frequent(x) )).T
                x_mean = pd.concat( [x_mean,x_most_freq], axis = 1 )

            if score_function:
                #if the score function is available
                predictions = score_function( model, Z.drop(columns='quantile')[features] )
                mean_pred = score_function( model, x_mean[features])[0]
                
            elif task  == 'r' or task =='reg' or task =='regression':
                
                #mean prediction
                mean_pred = model.predict(x_mean[features])[0]
                try:
                    if len(mean_pred) == 2:
                        mean_pred = mean_pred[0]
                except:
                    pass
                
                #prediciton
                predictions = model.predict(Z.drop(columns='quantile')[features])
                try:
                    if predictions.shape[1] == 2:
                        predictions = predictions[:,0]
                except:
                    pass
                
            elif task  == 'c' or task =='class' or task =='classification':

                #mean prediction
                mean_pred = model.predict_proba(x_mean[features])[0][class_to_analyze]
                #prediciton
                try:
                    predictions = model.predict_proba(Z.drop(columns='quantile')[features])[:,class_to_analyze]
                except:
                    predictions = model.predict_proba(Z.drop(columns='quantile')[features])[class_to_analyze]

            #build the dataframe with the standardize_effect, the predictions and the original effects
            effects = predictions - mean_pred
            df['effect'] = ( effects - np.mean(effects) ) / np.sqrt( np.var(effects)+0.0001 ) * ( max(predictions) - min(predictions) )
            df['predictions'] = predictions
            df['mean_prediction'] = mean_pred
            df['original'] = Z[feature].values
            df['quantile'] = Z['quantile'].values            
            importance_table.loc[feature,'Importance'] = np.mean( np.abs(df['effect'].values) )
            df['Importance'] = importance_table.loc[feature,'Importance']
            if task  == 'c' or task =='class' or task =='classification':
                df['class'] = class_to_analyze
            if feature in cat_features:
                df['type_feature'] = 'categorical'
            else:
                df['type_feature'] = 'numeric'
            df.index = np.repeat(feature, len(predictions))

            table = pd.concat([table, df])

        else: #if local
            #the procedure is the same but we must change the baseline and the scale of the effect (now the original prediction scale)
            if feature in cat_features:
                Z_quantitative = pd.DataFrame(numeric_df.loc[local]).T
                if cat_features != []:
                    Z_qualitative = create_level_variable_matrix(cat_df, feature,local = local)
                    Z = pd.concat( [ Z_quantitative.loc[Z_quantitative.index.repeat(len(Z_qualitative))].reset_index(drop=True) , Z_qualitative.reset_index(drop=True) ] , axis = 1 )
                else:
                    Z = Z_quantitative
            else:
                Z_quantitative = create_quantile_feature_matrix( numeric_df, feature, K, local = local, robust = robust )
                if cat_features != []:
                    Z_qualitative = pd.DataFrame( cat_df.loc[local] ).T
                    Z = pd.concat( [ Z_qualitative.loc[Z_qualitative.index.repeat(len(Z_quantitative))].reset_index(drop=True), Z_quantitative.reset_index(drop=True) ] , axis = 1  )
                else:
                    Z = Z_quantitative
            
            if score_function:
                #if the score function is available
                if label in dataframe.columns.tolist():
                    x_local = pd.DataFrame(dataframe.drop(columns = [label]).loc[local]).T
                else:
                    x_local = pd.DataFrame(dataframe).loc[local].T
                predictions = score_function( model, Z.drop(columns='quantile')[features] )
                local_pred = score_function( model, x_local[features].values)[0]

            elif task == 'r' or task=='reg' or task=='regression':
                #mean prediction
                if label in dataframe.columns.tolist():
                    x_local = pd.DataFrame(dataframe.drop(columns = [label]).loc[local]).T
                else:
                    x_local = pd.DataFrame(dataframe).loc[local].T
                local_pred = model.predict(x_local[features])[0]
                #prediciton
                predictions = model.predict(Z.drop(columns='quantile')[features])
                
            elif task == 'c' or task=='class' or task=='classification':

                #mean prediction
                local_pred = model.predict_proba( dataframe.drop(columns = label).loc[[local]] )[:,class_to_analyze][0] 
                #prediciton
                try:
                    predictions = model.predict_proba(Z.drop(columns='quantile')[features])[:,class_to_analyze]
                except:
                    predictions = model.predict_proba(Z.drop(columns='quantile')[features])[class_to_analyze]
                    
             #build the dataframe with the standardize_effect, the predictions and the original 

            local_value = dataframe.loc[local][feature]

            local_df['effect'] = predictions
            local_df['predictions'] = predictions
            local_df['mean_prediction'] = local_pred
            local_df['original'] = Z[feature].values
            local_df['quantile'] = Z['quantile'].values
            local_df['Importance'] = importance_table.loc[feature,'Importance']
           
            near_quantile = nearest_quantile(local_df, local_value, feature in cat_features)
            
            local_df['size'] = 0.2
            local_df.loc[local_df['quantile'] == near_quantile,'size'] = 1.0

            local_df['local_quantile'] = near_quantile
            local_df.index = np.repeat(feature, len(predictions))
            local_table = pd.concat([local_table, local_df])
        
    if local is None:            
        return table, importance_table, mean_pred
    else:
        return local_table, importance_table


##------------------------------------------------------------ ANOMALY DETECTION ------------------------------------------------------------

def _build_anomaly_detection_feature_importance_table(local_table, feature):
    '''
    Params:
    ---------
    - feature: str
        name of the feature to explore
    '''
    imp_table = local_table.loc[feature]   
    imp_table['direction'] = 'normal'
    imp_table.loc[imp_table.predictions > 0,'direction'] = 'anomalies'
    imp_table['effect'] = np.abs(imp_table['predictions'] - imp_table['mean_prediction'] )*np.sign(imp_table['predictions'])

    return imp_table

def _computeAnomalyDetectionImportance(local_table):
    '''
    '''
    change_anomalies = {}
    no_change_anomalies = {}
    
    importance_change = {}
    importance_no_change = {}

    for feature in local_table.index.unique():
        
        min_pred = local_table.loc[feature,'predictions'].min()
        max_pred = local_table.loc[feature,'predictions'].max()

        if np.sign(min_pred) !=  np.sign(max_pred):
            change_anomalies[feature] = [min_pred,max_pred]
            importance_change[feature] = max_pred-min_pred
        else:
            no_change_anomalies[feature] = [min_pred,max_pred]
            if np.sign(max_pred) == 1: #if anomalous
                np.abs(0 - max_pred) #the score must be higher if the feature can push to a more anomalous state the feature
            if np.sign(min_pred) == -1:
                np.abs(0 - min_pred)

            importance_no_change = np.abs(max_pred)
    
    importance_change_df = pd.concat([
                pd.DataFrame(importance_change,index=['AD importance score']).T,
                pd.DataFrame(change_anomalies,index=['Min','Max']).T]
                ,axis=1).sort_values('AD importance score',ascending=False)
    
    #importance_no_change = pd.concat([
    #            pd.DataFrame(importance_no_change,index=['AD importance score']).T,
    #            pd.DataFrame(no_change_anomalies,index=['Min','Max']).T]
    #            ,axis=1).sort_values('AD importance score',ascending=False)

    return importance_change_df