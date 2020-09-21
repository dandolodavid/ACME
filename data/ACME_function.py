import pandas as pd
import numpy as np
from utils import create_quantile_feature_matrix, create_level_variable_matrix, most_frequent, clean_list

def _computeACME(model, dataframe, features, quantitative_df, qualitative_df, importance_table, qualitative_features, quantitative_features, target_feature, task, local, K, class_to_analyze = None,table = None, local_table = None ):
    # for every feature, we compute the predictions based on the feature quantiles
    # create the variable-quantile matrix

    for feature in importance_table.index:
        df = pd.DataFrame()
        local_df = pd.DataFrame()

        if local is None:
            #for every quantitative feature, we compute the predictions based on the feature quantiles and create the variable-quantile matrix
            #for every quantitative feature, we compute the predictions based on the feature levels and create the variable-levels matrix         
            if feature in qualitative_features:
                
                Z_quantitative = pd.DataFrame( quantitative_df.mean() ).T
                if qualitative_features != []:
                    Z_qualitative = create_level_variable_matrix( qualitative_df, feature,local = None) 
                    Z = pd.concat( [ Z_quantitative.loc[Z_quantitative.index.repeat(len(Z_qualitative))].reset_index(drop=True) , Z_qualitative.reset_index(drop=True) ] , axis = 1 )
                else:
                    Z = Z_quantitative
            else:

                Z_quantitative = create_quantile_feature_matrix( quantitative_df, feature, K, local = None )
                if qualitative_features != []:
                    Z_qualitative = pd.DataFrame( qualitative_df.apply(lambda x: most_frequent(x), axis=0) ).T
                    Z = pd.concat( [ Z_qualitative.loc[Z_qualitative.index.repeat(len(Z_quantitative))].reset_index(drop=True), Z_quantitative.reset_index(drop=True) ] , axis = 1  )
                else:
                    Z = Z_quantitative


            x_mean = pd.DataFrame( quantitative_df.mean() ).T
            if qualitative_features != []:
                x_most_freq = pd.DataFrame( qualitative_df.apply( lambda x : most_frequent(x) )).T
                x_mean = pd.concat( [x_mean,x_most_freq], axis = 1 )
                
            if task  == 'r' or task =='reg' or task =='regression':
                
                #mean prediction
                mean_pred = model.predict(x_mean[features])[0]
                
                #prediciton
                predictions = model.predict(Z.drop(columns='quantile')[features])

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
            df.index = np.repeat(feature, len(predictions))

            table = pd.concat([table, df])

        else:

            if feature in qualitative_features:
                Z_quantitative = pd.DataFrame(quantitative_df.loc[local]).T
                if qualitative_features != []:
                    Z_qualitative = create_level_variable_matrix(qualitative_df, feature,local = local)
                    Z = pd.concat( [ Z_quantitative.loc[Z_quantitative.index.repeat(len(Z_qualitative))].reset_index(drop=True) , Z_qualitative.reset_index(drop=True) ] , axis = 1 )
                else:
                    Z = Z_quantitative
            else:
                Z_quantitative = create_quantile_feature_matrix( quantitative_df, feature, K, local = local )
                if qualitative_features != []:
                    Z_qualitative = pd.DataFrame( qualitative_df.apply(lambda x: most_frequent(x), axis=0) ).T
                    Z = pd.concat( [ Z_qualitative.loc[Z_qualitative.index.repeat(len(Z_quantitative))].reset_index(drop=True), Z_quantitative.reset_index(drop=True) ] , axis = 1  )
                else:
                    Z = Z_quantitative
                
            if task == 'r' or task=='reg' or task=='regression':

                #mean prediction
                x_local = pd.DataFrame(dataframe.drop(columns = [target_feature]).loc[local]).T
                local_pred = model.predict(x_local[features])[0]
                #prediciton
                predictions = model.predict(Z.drop(columns='quantile')[features])

            elif task == 'c' or task=='class' or task=='classification':

                #mean prediction
                local_pred = model.predict_proba( dataframe.drop(columns = target_feature).loc[[local]] )[:,class_to_analyze][0] 
                #prediciton
                try:
                    predictions = model.predict_proba(Z.drop(columns='quantile')[features])[:,class_to_analyze]
                except:
                    predictions = model.predict_proba(Z.drop(columns='quantile')[features])[class_to_analyze]

            #build the dataframe with the standardize_effect, the predictions and the original 

            local_df['effect'] = predictions
            local_df['predictions'] = predictions
            local_df['mean_prediction'] = local_pred
            local_df['original'] = Z[feature].values
            local_df['quantile'] = Z['quantile'].values
            local_df['Importance'] = importance_table.loc[feature,'Importance']
            local_df.index = np.repeat(feature, len(predictions))
            local_table = pd.concat([local_table, local_df])

        
    if local is None:            
        return table, importance_table
    else:
        return local_table, importance_table