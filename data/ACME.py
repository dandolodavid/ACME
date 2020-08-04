import pandas as pd
import numpy as np
from utils import create_quantile_feature_matrix, plot_express

class ACME():
    
    def __init__(self, model, target, features = None, K = 50, task = 'regression' ):
        self._model = model
        self._target = target
        self._features = features
        self._task = task
        self._meta = None
        self._K = K

    def fit(self, dataframe, label_class = None ):
        
        self._label_class = label_class

        if self._features is None:
            self._features = dataframe.drop(columns=self._target).columns.to_list()

        if self._task  == 'c' or self._task =='class' or self._task =='classification':
            if label_class is None:
                label_class = 0 
                print('WARNING: parameter "label_class" not passed, by default it is used the first class in self._model.classes_')    
            class_map = np.array(self._model.classes_)
            try:
                class_to_analyze = np.where(class_map == label_class)[0][0]
            except:
                class_to_analyze = np.where(class_map == str(label_class))[0][0]

            self._class_to_analyze = class_to_analyze
        
        out = pd.DataFrame(index=self._features)
        df = pd.DataFrame()
        table = pd.DataFrame()

        out['Importance'] = None

        for feature in out.index:
            # for every feature, we compute the predictions based on the feature quantiles
            #create the variable-quantile matrix
            Z = create_quantile_feature_matrix(dataframe, feature, self._target, self._K, local = None)
            x_mean = pd.DataFrame(dataframe.drop(columns = [self._target]).mean()).T

            if self._task  == 'r' or self._task =='reg' or self._task =='regression':
                
                #mean prediction
                mean_pred = self._model.predict(x_mean)[0]
                
                #prediciton
                predictions = self._model.predict(Z.drop(columns='quantile'))

            elif self._task  == 'c' or self._task =='class' or self._task =='classification':

                #mean prediction
                mean_pred = self._model.predict_proba(x_mean)[0][self._class_to_analyze]
                #prediciton
                try:
                    predictions = self._model.predict_proba(Z.drop(columns='quantile'))[:,self._class_to_analyze]
                except:
                    predictions = self._model.predict_proba(Z.drop(columns='quantile'))[self._class_to_analyze]

            #build the dataframe with the standardize_effect, the predictions and the original 
            #effects
            effects = predictions - mean_pred
            
            df['effect'] = ( effects - np.mean(effects) ) / np.sqrt( np.var(effects)+0.0001 ) * ( max(predictions) - min(predictions) )
            df['predictions'] = predictions
            df['mean_prediction'] = mean_pred
            df['original'] = Z[feature].values
            df['quantile'] = Z['quantile'].values            
            out.loc[feature,'Importance'] = np.mean( np.abs(df['effect'].values) )
            df['Importance'] = out.loc[feature,'Importance']
            df.index = np.repeat(feature, self._K)
            table = pd.concat([table, df])

        out.sort_values('Importance', ascending = False, inplace = True)

        self._meta = table
        self._feature_importance = out

        return self

    def fit_local(self, dataframe, local, label_class = None):
        
        if self._meta is None:
            self = self.fit(dataframe, label_class = label_class)
        
        df = pd.DataFrame()
        table = pd.DataFrame()

        for feature in self._feature_importance.index:
            #for every feature, we compute the predictions based on the feature quantiles
            #create the variable-quantile matrix
            Z = create_quantile_feature_matrix(dataframe, feature, self._target, self._K, local = local)
            
            if self._task == 'r' or self._task=='reg' or self._task=='regression':
                
                #mean prediction
                x_local = pd.DataFrame(dataframe.drop(columns = [self._target]).loc[local]).T
                local_pred = self._model.predict(x_local)[0]
                #prediciton
                predictions = self._model.predict(Z.drop(columns='quantile'))

            elif self._task == 'c' or self._task=='class' or self._task=='classification':

                #mean prediction
                local_pred = self._model.predict_proba( dataframe.drop(columns=self._target).loc[[local]] )[:,self._class_to_analyze][0] 
                #prediciton
                try:
                    predictions = self._model.predict_proba(Z.drop(columns='quantile'))[:,self._class_to_analyze]
                except:
                    predictions = self._model.predict_proba(Z.drop(columns='quantile'))[self._class_to_analyze]

            #build the dataframe with the standardize_effect, the predictions and the original             
            df['effect'] = predictions
            df['predictions'] = predictions
            df['mean_prediction'] = local_pred
            df['original'] = Z[feature].values
            df['quantile'] = Z['quantile'].values
            df['Importance'] = self._feature_importance.loc[feature,'Importance']
            df.index = np.repeat(feature, self._K)
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
            prova['quantile']=np.linspace(0,1,self._K)
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


