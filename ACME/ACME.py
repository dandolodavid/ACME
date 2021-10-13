import pandas as pd
import numpy as np
from ACME.utils import plot_express, clean_list
from ACME.ACME_function import _computeACME
import plotly.express as px

class ACME():
    
    def __init__(self, model, target, quantitative_features = [], qualitative_features = [], K = 50, task = 'regression', score_function = None ):

        self._model = model
        self._target = target
        self._quantitative_features = quantitative_features
        self._qualitative_features = qualitative_features
        self._task = task
        self._meta = None
        self._K = K
        self._score_function = score_function

    def fit(self, dataframe, robust = False, label_class = None ):    
    
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
        self._qualitative_df = None
        if not self._qualitative_features == []:
           self._qualitative_df = dataframe[ self._qualitative_features ].copy()

        #we save the features used by the model in the original order (necessary to correctly compute the predictiom)
        self._features = clean_list( dataframe.columns.to_list(), self._quantitative_features + self._qualitative_features)

        #if the task is the classification we must set the class to analyze 
        #in input the name of the class is passed, so we create a procedure to map from the label to the class number in the probability matrix
        if self._task  == 'c' or self._task =='class' or self._task =='classification':
            class_map = np.array(self._model.classes_)

            if label_class is not None:
                try:
                    class_to_analyze = np.where(class_map == label_class)[0][0]
                except:
                    class_to_analyze = np.where(class_map == str(label_class))[0][0]
                self._class_to_analyze = class_to_analyze
            if label_class is None:
                label_class = list(class_map)
            
            self._label_class = label_class

        #create conteiners for acme results
        out = pd.DataFrame(index= self._quantitative_features  + self._qualitative_features )
        out_table = pd.DataFrame()

        #if regression
        if self._task  == 'r' or self._task =='reg' or self._task =='regression':
            out_table, out, mean_pred = _computeACME( model = self._model, dataframe = dataframe, features = self._features,  
            numeric_df = self._quantitative_df, cat_df = self._qualitative_df, importance_table = out, score_function = self._score_function,
            label = self._target, task = self._task, local = None, K = self._K, robust = robust, table = out_table )

        #if classification
        if self._task  == 'c' or self._task =='class' or self._task =='classification':
            class_stack_importance = None
            if type(label_class) is list:
                label_list = range(0,len(label_class))
            else:
                label_list = [class_to_analyze]
            
            for i in label_list:
                out_table, out, mean_pred = _computeACME( model = self._model, dataframe = dataframe, features = self._features, 
                numeric_df = self._quantitative_df, cat_df = self._qualitative_df, importance_table = out, score_function = self._score_function,
                label = self._target, task = self._task,local = None, K=self._K, class_to_analyze = i, table = out_table )
                
                if len(label_list) > 1:
                    out.rename( columns = { 'Importance' : 'Importance_class_' + str(i) }, inplace=True )
                if class_stack_importance is None:
                    class_stack_importance = out
                else:
                    class_stack_importance.merge( out, left_index=True, right_index=True )
            
            if len(label_list) > 1:
                class_stack_importance['Importance'] = class_stack_importance.sum(axis=1).values

            #class_stack_importance.sort_values('Importance', ascending = False, inplace=True)
            out = class_stack_importance

        #define the output
        self._meta = out_table
        self._feature_importance = out.sort_values('Importance', ascending = False)
        self._mean_pred = mean_pred

        return self

    def fit_local(self, dataframe,local, robust = False, label_class = None):

        local_table = pd.DataFrame()
        class_to_analyze = None

        self._local = local
        
        if self._task  == 'c' or self._task =='class' or self._task =='classification':
            class_map = np.array(self._model.classes_)
            if label_class is not None:
                try:
                    class_to_analyze = np.where(class_map == label_class)[0][0]
                except:
                    class_to_analyze = np.where(class_map == str(label_class))[0][0]
            else:
                class_to_analyze = 0
                label_class = class_map[class_to_analyze]
                print( 'WARNING: in local interpretation, the label_class must be specified and not None. To default it\'s setted to class:' + str(class_map[0]) )
            
        self._class_to_analyze = class_to_analyze
        self._label_class = label_class

        #if the fitting procedure is not done, we frist compute the overall importance and create the quantitative and qualitative dataframe
        if self._meta is None:
            self = self.fit(dataframe, label_class = self._label_class)
            importance_table = self._feature_importance
        else:
            if self._feature_importance.shape[1] > 1:
                importance_table = self._feature_importance[ 'Importance_class_'+str(class_to_analyze)]
                importance_table.columns=['Importance']
                
        if self._task  == 'r' or self._task =='reg' or self._task =='regression': 
            local_table, out = _computeACME( model = self._model, dataframe = dataframe, features = self._features, 
                numeric_df = self._quantitative_df, cat_df = self._qualitative_df, score_function = self._score_function,
                importance_table = self._feature_importance.sort_values('Importance', ascending = False), label = self._target,
                task = self._task, local = local, K = self._K, local_table = local_table )
        
        if self._task  == 'c' or self._task =='class' or self._task =='classification':
            local_table, out =_computeACME( model = self._model, dataframe=dataframe, features = self._features, 
                numeric_df = self._quantitative_df, cat_df = self._qualitative_df, score_function = self._score_function,
                importance_table = self._feature_importance.sort_values('Importance', ascending = False), label = self._target,
                task = self._task, local = local, K = self._K, class_to_analyze = class_to_analyze, local_table = local_table )
            
        self._local_meta = local_table

        return self   
    
    def summary_plot(self, local = False):

        if (self._task  == 'c' or self._task =='class' or self._task =='classification') and type(self._label_class) is list and not local:
            plot_df = pd.DataFrame()
            i=0
            for label in self._label_class:
                tmp = pd.DataFrame(self._feature_importance.iloc[:,i])
                tmp.columns=['Importance']
                tmp['class'] = str(label)
                plot_df=pd.concat([plot_df,tmp],axis=0)
                i=i+1
            
            fig = px.bar(plot_df.iloc[::-1].reset_index().rename(columns={'index':'Feature'}), x='Importance',y="Feature", color='class', orientation='h', title='Overall Classification Importance')
        else:       
            meta = dict()
            meta['task'] = self._task
            if self._task  == 'c' or self._task =='class' or self._task =='classification':
                meta['label_class'] = self._label_class
            if local:
                table = self._local_meta
                meta['local'] = True
                meta['index'] = self._local 
                meta['base_line'] = self._mean_pred
            else:
                table = self._meta
                meta['local'] = False

            plot_df = pd.DataFrame()
            out = self._feature_importance.sort_values('Importance')
            for idx in out.index:
                prova = table.loc[idx].sort_values('original')
                plot_df = pd.concat([plot_df,prova])

            plot_df.drop_duplicates(subset = ['effect','predictions','quantile'], keep ='first')
            plot_df.reset_index(inplace=True)
            plot_df.rename(columns={'index':'feature'}, inplace=True)

            if local:
                meta['x'] = table['mean_prediction'].values[0]
            else: 
                meta['x'] = 0

            meta['y_bottom'] = plot_df['feature'].values[0]
            meta['y_top'] = plot_df['feature'].values[len(plot_df)-1]

            fig = plot_express(plot_df, meta)
        
        return fig.update_layout( title={ 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'})

    def bar_plot(self):
        import plotly.express as px
        if self._task == 'r' or self._task == 'reg' or self._task == 'regression':
            title = 'Barplot of feature importance: regression'
        else:
            title = 'Barplot of feature importance: classification'
        fig = px.bar(self._feature_importance.reset_index().sort_values('Importance').rename(columns={'index':'Feature'}), x='Importance',y="Feature", orientation='h', title = title)
        
        return fig.update_layout( title={ 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'})

    def feature_importance(self):
        return self._feature_importance
    
    def summary_table(self):
        return self._meta

    def local_table(self):
        return self._local_meta.drop(columns='size')

    # def local_feature_importance(self):
    #     features_quantile = [ self._local_meta.drop(columns='size').loc[f].local_quantile.unique()[0] for f in self._features ]
    #     features_quantile = pd.Series(features_quantile, index = self._features)
    #     local_effect = []
    #     for f in self._features:
    #         meta_f = self._meta.loc[f]
    #         local_effect.append( meta_f.loc[meta_f['quantile'] == features_quantile[f], 'effect' ].values[0] )
        
    #     return pd.DataFrame(local_effect, index=self._features, columns=['local_effect']).sort_values('local_effect',ascending=False,key=abs)

