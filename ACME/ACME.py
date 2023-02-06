import pandas as pd
import numpy as np
from ACME.utils import clean_list
from ACME.ACME_function import computeACME, build_feature_exploration_table, predictACME
from ACME.ACME_plot import ACME_summary_plot, feature_exploration_plot, ACME_barplot_multicalss
from ACME.ACME_anomaly_detection import build_anomaly_detection_feature_exploration_table, computeAnomalyDetectionImportance
import plotly.express as px
import plotly.graph_objects as go

class ACME():
    
    def __init__(self, model, target, features=[], cat_features=[], K=50, task='regression', score_function=None):
        '''
        Initialization
        
        Params:
        ------
        - model: object
            the object of the model
        - target: str
            name of the target_feature
        - features: [str] 
            list of the features name in SAME ORDER as the the model input
        - cat_features: [str]
            categorical features name
        - K: int
            number of quantiles to use
        - task: str
            type of model task {'regression','reg','r', 'c','class','classification', 'ad','anomaly detection'}
        - score_function: fun(model, x)
            a function that has in input the model and the input data to realize the prediction. It must return a numeric score

        '''

        self._model = model
        self._target = target
        self._features = features
        self._numeric_features = list(np.setdiff1d(features, cat_features))
        self._cat_features = cat_features
        self._task = task
        self._global_explain = {}
        self._local_explain = {}
        self._K = K
        self._score_function = score_function
        self._class_to_analyze = None
        self._label_class = None

    def explain(self, dataframe, robust = False, label_class = None):    
        '''
        Fit the acme explainability.

        Params:
        -------
        - dataframe : pd.DataFrame
            input dataframe
        - robust : bool (default False)
            if True use only quantile from 0.05 and 0.95, if False use from 0 to 1
        - label_class : str,int (default None)
            if task is classification, specify the class of interest
        '''

        # if cat features and numeric features are not specified then all the columns of the dataset (not the target) are used as numeric feature
        if self._numeric_features == [] and self._cat_features == []:
            self._numeric_features = dataframe.drop(columns=self._target).columns.to_list()

        # create the dataframe of numeric feature
        if self._numeric_features == []:
            if self._cat_features == [] :
                self._numeric_features = dataframe.drop(columns= [self._target]).columns.to_list()
            else:
                self._numeric_features = dataframe.drop(columns= [self._target] + self._cat_features ).columns.to_list()

        self._numeric_df = dataframe[ self._numeric_features ].copy()
        
        # create the dataframe for cat feature
        self._cat_df = None
        if not self._cat_features == []:
           self._cat_df = dataframe[ self._cat_features ].copy()

        # we save the features used by the model in the original order (necessary to correctly compute the predictiom)
        self._features = clean_list( dataframe.columns.to_list(), self._numeric_features + self._cat_features)

        # if the label class is given, find the corrispective position in the model class map
        # else auto set as label class the first class
        # N.B. : label_class is the class name, class_to_analyze is the label_class corrispective number in the model class map
        if self._task  in ['c','class','classification']:
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

        # compute local acme for regression or classifiction  
        if self._task in ['r','reg','regression'] or self._task in ['ad','anomaly detection']: 
            feature_table, importance_table, baseline_pred, baseline, perc_functions = computeACME( model = self._model, task = self._task, 
                                                                                features = self._features, 
                                                                                numeric_df = self._numeric_df, cat_df = self._cat_df,
                                                                                score_function = self._score_function,
                                                                                K = self._K, robust = robust )
        if self._task  in ['c','class','classification']:
            
            class_stack_importance = []
            class_stack_feature_table = []
            label_dict = {}

            if type(label_class) is list:
                label_list = range(0,len(label_class))
            else:
                label_list = [class_to_analyze]
            
            # explore every label
            
            for lab in label_list:
                label_dict[lab] = class_map[lab]
                feature_table, importance_table, baseline_pred, baseline, perc_functions = computeACME( model = self._model, task = self._task,
                                                                                    features = self._features, class_to_analyze = lab,
                                                                                    numeric_df = self._numeric_df, cat_df = self._cat_df,
                                                                                    score_function = self._score_function, 
                                                                                    K=self._K, robust = robust )
                
                # rename the columns in case of multilabel
                if len(label_list) > 1:
                    importance_table.rename( columns = { 'importance' : 'importance_class_' + str(label_class[lab]) }, inplace=True )
                
                # save the results
                class_stack_importance.append(importance_table)
                class_stack_feature_table.append(feature_table)
            
            # concat the results
            feature_table = pd.concat(class_stack_feature_table)
            feature_table['class'] = feature_table['class'].map(label_dict)

            class_stack_importance = pd.concat(class_stack_importance,axis=1)  
            #if multilabel sum the importance of each feature in each 
            if len(label_list) > 1:
                class_stack_importance['importance'] = class_stack_importance.sum(axis=1).values

            class_stack_importance.sort_values('importance', ascending = False, inplace=True)
            importance_table = class_stack_importance

        # create the outputs
        self._global_explain = {'meta':feature_table.copy(),
                                'feature_importance':importance_table.sort_values('importance', ascending = False).copy(),
                                'baseline_pred':baseline_pred,
                                'baseline':baseline,
                                }
        #self._global_meta = feature_table.copy()
        #self._global_feature_importance = importance_table.sort_values('importance', ascending = False).copy()
        #self._global_baseline_pred = baseline_pred
        #self._global_baseline = baseline
        self._perc_functions = perc_functions

        return self

    def explain_local(self, series, label_class = None):
        '''
        Params:
        -------
        - series : pd.Series
            observation on which explain the prediction
        - label_class : str,int (default None)
            if task is classification, specify the class of interest

        Returns:
        --------
        - pd.DataFrame
        '''

        # if the label class is given, find the corrispective position in the model class map
        # else auto set as label class the first class
        # N.B. : label_class is the class name, class_to_analyze is the label_class corrispective number in the model class map

        if len(self._global_explain.keys())==0:
            raise InterruptedError("You must first fit the acme explaination model with the 'explain' command on the same data used to train the model")

        if self._task  in ['c','class','classification']:
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

            # save the class to analize and the label
            self._class_to_analyze = class_to_analyze
            self._label_class = label_class

        local_table, local_importance_table, local_baseline_pred, local_baseline = predictACME(model=self._model, 
                    series=series,
                    features=self._features,
                    meta_table=self._global_explain['meta'],
                    percentile_functions=self._perc_functions,
                    task=self._task, 
                    score_function=self._score_function, 
                    class_to_analyze=self._class_to_analyze)
        
        if self._task in ['c','class','classification']:
            local_table['class'] = local_table['class'].map({class_to_analyze : class_map[class_to_analyze]})

        self._local_explain = {'meta':local_table.copy(),
                                'feature_importance':local_importance_table.sort_values('importance', ascending = False).copy(),
                                'baseline_pred':local_baseline_pred,
                                'baseline':local_baseline,
                                'local_name':series.name}  # save the index of the local observation, it's the name of the series (corrisponding to the dataframe index)

        #self._local_meta = local_table
        #self._local_baseline = local_baseline
        #self._local_importance_table = local_importance_table
        #self._local_baseline_pred = local_baseline_pred

        return self

    def feature_importance(self, local=False, weights = {}):
        '''
        Returns the feature importance calculated by AcME.
        In case of Anomaly Detection task, it provides ad hoc explanation for anomaly detection, studied for local interpretability.
        The score will show what features can altered the prediction from normal to anomalies and viceversa.

        Params:
        ------
        - local : bool  
            if true and task is AD, it return the local AD version of feature importance
        - weights : dict 
            Dictionary with the importance for each element. Sum must be 1
            * ratio : float
                importance of local score position
            * distance : float
                importance of inter-quantile distance necessary to change
            * change : float
                importance of the possibility to change prediction
            * delta : float
                importance of the score delta

        Return : 
        -------
        - pd.DataFrame
        '''

        # if the task is anomaly detection and we have already fitted a local acme 
        # calculate the importance for the selected row anomaly detection task
        if self._task in ['ad','anomaly detection'] and local:

            local_table = self._local_explain['meta'].drop(columns='size').copy()
            importance_df = computeAnomalyDetectionImportance(local_table, weights = weights)   
        
            return importance_df

        # if local then we return the local importance
        elif local:
            return self._local_explain['feature_importance']

        # else simply return the importance calculated by acme for global explain
        else:
            return self._global_explain['feature_importance']

    def feature_exploration(self, feature, local=False, plot=False):
        '''
        Generate anomaly detection feature exploration table or a plot for local observation that, 
        choosen a specific feature, shows how the anomaly score can change beacuse of the feature.

        Params:
        -------
        - feature : str
            selected feature's name
        - plot : bool
            if true returns the plot, else returs the table

        Returns:
        --------
        - feature_table : pd.DataFrame
        '''

        # extract the desired table
        if local:
            table = self._local_explain['meta']
        else:
            table = self._global_explain['meta']

        # build the correct feature exploration table
        if self._task in ['ad','anomaly detection']:
            feature_table = build_anomaly_detection_feature_exploration_table(table, feature)
        else:
            feature_table = build_feature_exploration_table(table, feature)
        
        # if plot return plot, else return the table
        if plot:
            fig = feature_exploration_plot(feature_table, feature, self._task)
            return fig
        else:
            return feature_table

    def summary_plot(self, local=False):
        '''
        Generate the recap plot

        Params: 
        -------
        - local : bool
            if local or global plot 
        
        Returns:
        --------
        - plotly figure
        '''

        # if desired explainability is global, task is classification and there are multi label: produce the bar plot
        if self._task in ['c','class','classification'] and type(self._label_class) is list and not local:
            fig = ACME_barplot_multicalss(self._global_explain['feature_importance'], self._label_class)

        # generate the quantile/feature/effect plot
        else:       
            meta = dict()
            meta['task'] = self._task
            meta['local'] = False

            if self._task  in ['c','class','classification']:
                meta['label_class'] = self._label_class
                meta['local'] = False
                
            if local:
                table = self._local_explain['meta']
                meta['local'] = True
                meta['index'] = self._local_explain['local_name']
                meta['baseline'] = self._local_explain['baseline_pred']
            else:
                table = self._global_explain['meta']
                meta['local'] = False

            # prepare for the plotting
            plot_df = table.sort_values('original').reset_index().rename(columns={'index':'feature'}).copy()

            # if local set the refering x to the local values observation
            # for the global set to 0
            if local:
                meta['x'] = table['baseline_prediction'].values[0]
            else: 
                meta['x'] = 0

            # set the top and the bottom of the y-axis (first and last feature)
            meta['y_bottom'] = plot_df['feature'].values[0]
            meta['y_top'] = plot_df['feature'].values[len(plot_df)-1]

            # generate the plot
            fig = ACME_summary_plot(plot_df, meta)
        
        return fig

    def bar_plot(self, local=False):
        '''
        Feature importance plot
        '''
        if self._task in ['r', 'reg', 'regression']:
            title = 'Barplot of feature importance: regression'
        elif self._task in ['ad','anomaly detection']:
            title = 'Barplot of feature importance: anomaly detection'
        else:
            title = 'Barplot of feature importance: classification'

        if local:
            table = self._local_explain['feature_importance']
            title = 'Local importance observation ID: ' + str(self._local_explain['local_name']) + '.<br>'+title
        else:
            table = self._global_explain['feature_importance']

        table = round(table.reset_index().sort_values('importance').rename(columns={'index':'feature'}),3)
        fig = px.bar(table, 
                    x='importance',
                    y='feature', 
                    orientation='h', 
                    title = title)

        fig.update_traces(hovertemplate = 'Feature:<b>%{y}</b><br>Importance:%{x}')
        return fig.update_layout( title={ 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'} )
    
    def summary_table(self, local=False):
        '''
        Expose the global or local summary table
        '''
        if local:
            return self._local_explain['meta'].drop(columns='size')     
        else: 
            return self._global_explain['meta'].drop(columns='size')
            
    def baseline_values(self, local=False):
        '''
        Expose the baseline vector used for AcME
        '''
        if local:
            return self._local_explain['baseline']
        else:
            return self._global_explain['baseline']

    def metadata(self):

        return {'model':self._model,
        'target':self._target,
        'features':self._features,
        'numeric_features':self._numeric_features,
        'cat_features':self._cat_features,
        'task':self._task,
        'global_explain':self._global_explain,
        'local_explain':self._local_explain,
        'K':self._K,
        'score_function':self._score_function}