import pandas as pd
import numpy as np
from ACME.utils import plot_express, clean_list
from ACME.ACME_function import _computeACME, _build_anomaly_detection_feature_importance_table, _computeAnomalyDetectionImportance
import plotly.express as px
import plotly.graph_objects as go

class ACME():
    
    def __init__(self, model, target, features = [], cat_features = [], K = 50, task = 'regression', score_function = None ):

        self._model = model
        self._target = target
        self._features = features
        self._numeric_features = list(np.setdiff1d(features, cat_features))
        self._cat_features = cat_features
        self._task = task
        self._meta = None
        self._K = K
        self._score_function = score_function

    def fit(self, dataframe, robust = False, label_class = None ):    
    
        #if cat features and numeric features are not specified then all the columns of the dataset (not the target) are used as numeric feature
        if self._numeric_features == [] and self._cat_features == []:
            self._numeric_features = dataframe.drop(columns=self._target).columns.to_list()

        #create the dataframe of numeric feature
        if self._numeric_features == []:
            if self._cat_features == [] :
                self._numeric_features = dataframe.drop(columns= [self._target]).columns.to_list()
            else:
                self._numeric_features = dataframe.drop(columns= [self._target] + self._cat_features ).columns.to_list()

        self._numeric_df = dataframe[ self._numeric_features ].copy()
        
        #create the dataframe for cat feature
        self._cat_df = None
        if not self._cat_features == []:
           self._cat_df = dataframe[ self._cat_features ].copy()

        #we save the features used by the model in the original order (necessary to correctly compute the predictiom)
        self._features = clean_list( dataframe.columns.to_list(), self._numeric_features + self._cat_features)

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
        out = pd.DataFrame(index= self._numeric_features  + self._cat_features )
        out_table = pd.DataFrame()

        #if regression
        if self._task  == 'r' or self._task =='reg' or self._task =='regression':
            out_table, out, mean_pred = _computeACME( model = self._model, dataframe = dataframe, features = self._features,  
            numeric_df = self._numeric_df, cat_df = self._cat_df, importance_table = out, score_function = self._score_function,
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
                numeric_df = self._numeric_df, cat_df = self._cat_df, importance_table = out, score_function = self._score_function,
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
                print('WARNING: in local interpretation, the label_class must be specified and not None. To default it\'s setted to class:' + str(class_map[0]))
            
        self._class_to_analyze = class_to_analyze
        self._label_class = label_class

        #if the fitting procedure is not done, we frist compute the overall importance and create the numeric and cat dataframe
        if self._meta is None:
            self = self.fit(dataframe, label_class = self._label_class)
            importance_table = self._feature_importance
        else:
            if self._feature_importance.shape[1] > 1:
                importance_table = self._feature_importance[ 'Importance_class_'+str(class_to_analyze)]
                importance_table.columns=['Importance']
                
        if self._task  == 'r' or self._task =='reg' or self._task =='regression': 
            local_table, out = _computeACME( model = self._model, dataframe = dataframe, features = self._features, 
                numeric_df = self._numeric_df, cat_df = self._cat_df, score_function = self._score_function,
                importance_table = self._feature_importance.sort_values('Importance', ascending = False), label = self._target,
                task = self._task, local = local, K = self._K, local_table = local_table )
        
        if self._task  == 'c' or self._task =='class' or self._task =='classification':
            local_table, out =_computeACME( model = self._model, dataframe=dataframe, features = self._features, 
                numeric_df = self._numeric_df, cat_df = self._cat_df, score_function = self._score_function,
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

    def anomaly_detection_importance(self):
        '''
        Provides an ad hoc explaination for anomaly detection, studied for local interpretability
        The score will show what features can altered the prediction from normal to anomalies and viceversa.
        Please note that require to already called the local acme interpretability on a specific observation
        '''
        local_table = self._local_meta.drop(columns='size').copy()
        return _computeAnomalyDetectionImportance(local_table)      

    def anomaly_detection_feature_exploration_plot(self, feature, anomalies_direction = 'negative'):
        '''
        Generate a plot for local observation that, choosen a specific feature, shows how the anomaly score can change beacuse of the feature.
        
        Params:
        ---------
        - feature: str
            name of the feature to explore
        
        Return:
        ---------
        - plolty figure
        '''
        local_table = self._local_meta.drop(columns='size').copy()
        imp_table = _build_anomaly_detection_feature_importance_table(local_table, feature)
        actual_score = imp_table['mean_prediction'].values[0]
        actual_values = imp_table.loc[imp_table['quantile'] == imp_table['local_quantile'].values[0], 'original'].values[0]
        color = 'red' if actual_score > 0 else 'blue'
        fig = go.Figure()

        fig.add_bar(x = imp_table.loc[imp_table.direction=='anomalies','effect'], 
                    y = imp_table.loc[imp_table.direction=='anomalies','original'].values, 
                    base =  imp_table['mean_prediction'].values[0], 
                    marker=dict(color = 'red'), name = 'Anomalies',orientation='h')

        fig.add_bar(x = imp_table.loc[imp_table.direction=='normal','effect'], 
                    y = imp_table.loc[imp_table.direction=='normal','original'].values, 
                    base =  imp_table['mean_prediction'].values[0],
                    marker=dict(color = 'blue'), name = 'Normal', orientation='h')

        fig.add_scatter( y = [ imp_table['original'].values[0]*0.9 ,imp_table['original'].values[-1]*1.05 ],
                        x = [ actual_score,actual_score ], mode='lines',
                        name = 'actual score', line=dict(color = color ,width=2,dash="dash") )

        fig.add_scatter( y = [ imp_table['original'].values[0]*0.9 ,imp_table['original'].values[-1]*1.05 ],
                        x = [ 0,0 ], mode='lines',
                        line=dict(color="black",width=2),  name = 'change point')

        fig.add_scatter( x = [ actual_score ],
                        y = [ actual_values], mode='markers',
                        marker=dict(size=20,color=color),  name = 'current value')

        fig.update_layout(title='Feature ' + str(feature), 
                        yaxis_title = "Feature values",
                        xaxis_title = "Anomaly Score", autosize=True )

        return fig
    
    def anomaly_detection_feature_exploration_table(self, feature):
        '''
        '''
        local_table = self._local_meta.drop(columns='size').copy()
        imp_table = _build_anomaly_detection_feature_importance_table(local_table, feature)
        return imp_table
