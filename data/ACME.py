import pandas as pd
import numpy as np
from utils import create_quantile_feature_matrix, plot_express, create_level_variable_matrix, most_frequent, clean_list

class ACME():
    
    def __init__(self, model, target, quantitative_features = [], qualitative_features = [], K = 50, task = 'regression' ):
        self._model = model
        self._target = target
        self._quantitative_features = quantitative_features
        self._qualitative_features = qualitative_features
        self._task = task
        self._meta = None
        self._K = K

    def fit(self, dataframe, label_class = None ):
        
        self._label_class = label_class

        #if qualitative features and quantitative features are not specified then all the columns of the dataset (not the target) are used as quantitative feature
        if self._quantitative_features == [] and self._qualitative_features == []:
            self._quantitative_features = dataframe.drop(columns=self._target).columns.to_list()

        #create the dataframe of quantitative feature
        if self._quantitative_features == []:
            if self._qualitative_features == [] :
                self._quantitative_features = dataframe.drop(columns= [self._target]).columns.to_list()
            else:
                self._quantitative_features = dataframe.drop(columns= [self._target] + self._qualitative_features ).columns.to_list()

        self._quantitative_df = dataframe[ self._quantitative_features ].copy()
        
        #create the dataframe for qualitative feature
        if not self._qualitative_features == []:
           self._qualitative_df = dataframe[ self._qualitative_features ].copy()

        #we save the features used by the model in the original order (necessary to correctly compute the predictiom)
        self._features = clean_list( dataframe.columns.to_list(), self._quantitative_features + self._qualitative_features)

        #if the task is the classification we must set the class to analyze 
        #in input the name of the class is passed, so we create a procedure to map from the label to the class number in the probability matrix
        if self._task  == 'c' or self._task =='class' or self._task =='classification':
            if label_class is None:
                label_class = 0 
                print('WARNING: parameter "label_class" not passed, by default it is used the first class in self._model.classes_')    
            class_map = np.array(self._model.classes_)
            try:
                class_to_analyze = np.where(class_map == label_class)[0][0] 
            except:
                class_to_analyze = np.where(class_map == str(label_class))[0][0] #if the name is passed as array but the model used it as str

            self._class_to_analyze = class_to_analyze
    
        out = pd.DataFrame(index= self._quantitative_features  + self._qualitative_features )
        
        table = pd.DataFrame()

        for feature in out.index:
            
            df = pd.DataFrame()

            #for every quantitative feature, we compute the predictions based on the feature quantiles and create the variable-quantile matrix
            #for every quantitative feature, we compute the predictions based on the feature levels and create the variable-levels matrixx         
            if feature in self._qualitative_features:

                Z_quantitative = pd.DataFrame( self._quantitative_df.mean() ).T
                if self._qualitative_features != []:
                    Z_qualitative = create_level_variable_matrix( self._qualitative_df, feature,local = None) 
                    Z = pd.concat( [ Z_quantitative.loc[Z_quantitative.index.repeat(len(Z_qualitative))].reset_index(drop=True) , Z_qualitative.reset_index(drop=True) ] , axis = 1 )
                else:
                    Z = Z_quantitative
            else:
                Z_quantitative = create_quantile_feature_matrix( self._quantitative_df, feature, self._K, local = None )
                if self._qualitative_features != []:
                    Z_qualitative = pd.DataFrame( self._qualitative_df.apply(lambda x: most_frequent(x), axis=0) ).T
                    Z = pd.concat( [ Z_qualitative.loc[Z_qualitative.index.repeat(len(Z_quantitative))].reset_index(drop=True), Z_quantitative.reset_index(drop=True) ] , axis = 1  )
                else:
                    Z = Z_quantitative
 

            x_mean = pd.DataFrame( self._quantitative_df.mean() ).T
            if self._qualitative_features != []:
                x_most_freq = pd.DataFrame( self._qualitative_df.apply( lambda x : most_frequent(x) )).T
                x_mean = pd.concat( [x_mean,x_most_freq], axis = 1 )

            if self._task  == 'r' or self._task =='reg' or self._task =='regression':
                
                #mean prediction
                mean_pred = self._model.predict(x_mean[self._features])[0]
                
                #prediciton
                predictions = self._model.predict(Z.drop(columns='quantile')[self._features])

            elif self._task  == 'c' or self._task =='class' or self._task =='classification':

                #mean prediction
                mean_pred = self._model.predict_proba(x_mean[self._features])[0][self._class_to_analyze]
                #prediciton
                try:
                    predictions = self._model.predict_proba(Z.drop(columns='quantile')[self._features])[:,self._class_to_analyze]
                except:
                    predictions = self._model.predict_proba(Z.drop(columns='quantile')[self._features])[self._class_to_analyze]

            #build the dataframe with the standardize_effect, the predictions and the original 
            #effects
            effects = predictions - mean_pred
            
            #save all the information in a dataframe
            df['effect'] = ( effects - np.mean(effects) ) / np.sqrt( np.var(effects)+0.0001 ) * ( max(predictions) - min(predictions) )
            df['predictions'] = predictions
            df['mean_prediction'] = mean_pred
            df['original'] = Z[feature].values
            df['quantile'] = Z['quantile'].values            
            out.loc[feature,'Importance'] = np.mean( np.abs(df['effect'].values) )
            df['Importance'] = out.loc[feature,'Importance']
            df.index = np.repeat(feature, len(effects))
            table = pd.concat([table, df])

        out.sort_values('Importance', ascending = False, inplace = True)

        self._meta = table
        self._feature_importance = out

        return self

    def fit_local(self, dataframe, local, label_class = None):
        
        #if the fitting procedure is not done, we frist compute the overall importance and create the quantitative and qualitative dataframe
        if self._meta is None:
            self = self.fit(dataframe, label_class = label_class)
        
        table = pd.DataFrame()

        for feature in self._feature_importance.index:
            df = pd.DataFrame()
            
            #for every quantitative feature, we compute the predictions based on the feature quantiles and create the variable-quantile matrix
            #for every quantitative feature, we compute the predictions based on the feature levels and create the variable-levels matrix
            if feature in self._qualitative_features:
                Z_quantitative = pd.DataFrame(self._quantitative_df.loc[local]).T
                if self._qualitative_features != []:
                    Z_qualitative = create_level_variable_matrix(self._qualitative_df, feature,local = local)
                    Z = pd.concat( [ Z_quantitative.loc[Z_quantitative.index.repeat(len(Z_qualitative))].reset_index(drop=True) , Z_qualitative.reset_index(drop=True) ] , axis = 1 )
                else:
                    Z = Z_quantitative
            else:
                Z_quantitative = create_quantile_feature_matrix( self._quantitative_df, feature, self._K, local = local )
                if self._qualitative_features != []:
                    Z_qualitative = pd.DataFrame( self._qualitative_df.apply(lambda x: most_frequent(x), axis=0) ).T
                    Z = pd.concat( [ Z_qualitative.loc[Z_qualitative.index.repeat(len(Z_quantitative))].reset_index(drop=True), Z_quantitative.reset_index(drop=True) ] , axis = 1  )
                else:
                    Z = Z_quantitative

            if self._task == 'r' or self._task=='reg' or self._task=='regression':
                
                #mean prediction
                x_local = pd.DataFrame(dataframe.drop(columns = [self._target]).loc[local]).T                
                local_pred = self._model.predict(x_local[self._features])[0]
               
                #prediciton
                predictions = self._model.predict(Z.drop(columns='quantile')[self._features])

            elif self._task == 'c' or self._task=='class' or self._task=='classification':

                #mean prediction
                local_pred = self._model.predict_proba( dataframe.drop(columns=self._target)[self._features].loc[[local]] )[:,self._class_to_analyze][0] 
                #prediciton
                try:
                    predictions = self._model.predict_proba(Z.drop(columns='quantile')[self._features])[:,self._class_to_analyze]
                except:
                    predictions = self._model.predict_proba(Z.drop(columns='quantile')[self._features])[self._class_to_analyze]

            #build the dataframe with the standardize_effect, the predictions and the original             
            df['effect'] = predictions
            df['predictions'] = predictions
            df['mean_prediction'] = local_pred
            df['original'] = Z[feature].values
            df['quantile'] = Z['quantile'].values
            df['Importance'] = self._feature_importance.loc[feature,'Importance']
            df.index = np.repeat(feature, len(predictions))
            table = pd.concat([table, df])

        self._local_meta = table

        return self
    
    def summary_plot(self, local = False):
        
        from data_science.output import Output
        
        if local:
            table = self._local_meta
        else:
            table = self._meta

        plot_df = pd.DataFrame()
        out = self._feature_importance.sort_values('Importance')
        for idx in out.index:
            prova = table.loc[idx].sort_values('original')
            plot_df = pd.concat([plot_df,prova])

        plot_df.drop_duplicates(subset = ['effect','predictions','quantile'], keep ='first')
        plot_df.reset_index(inplace=True)
        plot_df.rename(columns={'index':'feature'}, inplace=True)
        
        meta = dict()
        if local:
            meta['x'] = table['mean_prediction'].values[0]
        else: 
            meta['x'] = 0
        meta['y_bottom'] = plot_df['feature'].values[0]
        meta['y_top'] = plot_df['feature'].values[len(plot_df)-1]

        plot_express(plot_df, meta).show()
    
    def bar_plot(self):
        import plotly.express as px

        fig = px.bar(self._feature_importance.reset_index().sort_values('Importance').rename(columns={'index':'Feature'}), x='Importance',y="Feature", orientation='h')
        fig.show()

    def feature_importance(self):
        return self._feature_importance
    
    def summary_table(self):
        return self._meta

    def local_table(self):
        return self._local_meta


