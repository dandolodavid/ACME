import pandas as pd
import numpy as np
from ACME.utils import clean_list
from ACME.ACME_function import computeACME, build_feature_exploration_table
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
        self._meta = None
        self._local = None
        self._label_class = None
        self._K = K
        self._score_function = score_function


    def fit(self, dataframe, robust = False, label_class = None):    
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
            feature_table, importance_table, baseline_pred, baseline = computeACME( model = self._model, dataframe = dataframe, task = self._task, 
                                                                                features = self._features, label = self._target,  
                                                                                numeric_df = self._numeric_df, cat_df = self._cat_df,
                                                                                score_function = self._score_function,
                                                                                local = None, K = self._K, robust = robust )
        if self._task  in ['c','class','classification']:
            
            class_stack_importance = []
            class_stack_feature_table = []

            if type(label_class) is list:
                label_list = range(0,len(label_class))
            else:
                label_list = [class_to_analyze]
            
            # explore every label
            for lab in label_list:
                feature_table, importance_table, baseline_pred, baseline = computeACME( model = self._model, dataframe = dataframe, task = self._task,
                                                                                    features = self._features, label = self._target, class_to_analyze = lab,
                                                                                    numeric_df = self._numeric_df, cat_df = self._cat_df,
                                                                                    score_function = self._score_function, 
                                                                                    local = None, K=self._K, robust = robust )
                
                # rename the columns in case of multilabel
                if len(label_list) > 1:
                    importance_table.rename( columns = { 'importance' : 'importance_class_' + str(label_class[lab]) }, inplace=True )
                
                # save the results
                class_stack_importance.append(importance_table)
                class_stack_feature_table.append(feature_table)
            
            # concat the results
            feature_table = pd.concat(class_stack_feature_table)

            class_stack_importance = pd.concat(class_stack_importance,axis=1)  
            #if multilabel sum the importance of each feature in each 
            if len(label_list) > 1:
                class_stack_importance['importance'] = class_stack_importance.sum(axis=1).values

            class_stack_importance.sort_values('importance', ascending = False, inplace=True)
            importance_table = class_stack_importance

        # create the outputs
        self._meta = feature_table.copy()
        self._feature_importance = importance_table.sort_values('importance', ascending = False).copy()
        self._baseline_pred = baseline_pred
        self._global_baseline = baseline

        return self

    def fit_local(self, dataframe, local, robust = False, label_class = None):
        '''
        Fit the local version of AcME explainability.

        Params:
        -------
        - dataframe : pd.DataFrame
            input dataframe
        - local : int,str
            dataframe index of the desired row
        - robust : bool (default False)
            if True use only quantile from 0.05 and 0.95, if False use from 0 to 1
        - label_class : str,int (default None)
            if task is classification, specify the class of interest
        '''

        # save the index of the local observation
        self._local = local
        
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
            else:
                class_to_analyze = 0
                label_class = class_map[class_to_analyze]
                print('WARNING: in local interpretation, the label_class must be specified and not None. To default it\'s setted to class:' + str(class_map[0]))

            # save the class to analize and the label
            self._class_to_analyze = class_to_analyze
            self._label_class = label_class

        # if the fitting procedure is not done, we frist compute the overall importance and create the numeric and cat dataframe
        # this is done to have the same ranking of the global score and common to all the local explaination
        if self._meta is None:
            self = self.fit(dataframe, label_class=self._label_class)
            importance_table = self._feature_importance
        else:
            if self._feature_importance.shape[1] > 1:
                importance_table = self._feature_importance[ 'importance_class_'+str(class_to_analyze) ]
                importance_table.columns = ['importance']

        # compute local acme for regression or classifiction     
        if self._task in ['r','reg','regression'] or self._task in ['ad','anomaly detection']: 
            local_table, out, baseline_pred, baseline = computeACME( model = self._model, dataframe = dataframe, task = self._task,
                                                    features = self._features, label = self._target, 
                                                    numeric_df = self._numeric_df, cat_df = self._cat_df, 
                                                    score_function = self._score_function,
                                                    local = local, K = self._K, robust = robust )
        
        if self._task in ['c','class','classification']:
            local_table, out, baseline_pred, baseline = computeACME( model = self._model, dataframe = dataframe, task = self._task,
                                                    features = self._features, label = self._target, class_to_analyze = class_to_analyze,
                                                    numeric_df = self._numeric_df, cat_df = self._cat_df, 
                                                    score_function = self._score_function,
                                                    local = local, K = self._K, robust = robust )

        # save the local table
        self._local_meta = local_table
        self._local_baseline = baseline

        return self   

    def feature_importance(self, local=False, weights = {}):
        '''
        Returns the feature importance calculated by AcME.
        In case of Anomaly Detection task, it provides ad hoc explaination for anomaly detection, studied for local interpretability.
        The score will show what features can altered the prediction from normal to anomalies and viceversa.

        Params:
        ------
        - local : bool  
            if true and task is AD, it return the local AD version of feature importance
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

        Return : 
        -------
        - pd.DataFrame
        '''

        # if the task is anomaly detection and we have already fitted a local acme 
        # calculate the importance for the selected row anomaly detection task
        if self._task in ['ad','anomaly detection'] and local:

            local_table = self._local_meta.drop(columns='size').copy()
            importance_df = computeAnomalyDetectionImportance(local_table, weights = weights)   
        
            return importance_df

        # else simply return the importance calculated by acme
        else:
            return self._feature_importance


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
            table = self._local_meta
        else:
            table = self._meta

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
            fig = ACME_barplot_multicalss(self._feature_importance, self._label_class)

        # generate the quantile/feature/effect plot
        else:       
            meta = dict()
            meta['task'] = self._task
            if self._task  in ['c','class','classification']:
                meta['label_class'] = self._label_class
            if local:
                table = self._local_meta
                meta['local'] = True
                meta['index'] = self._local 
                meta['baseline'] = self._baseline_pred
            else:
                table = self._meta
                meta['local'] = False

            plot_df = pd.DataFrame()
            out = self._feature_importance.sort_values('importance')

            # for each feature we add the feature's table sorted to the plot dataframe
            for idx in out.index:
                tmp = table.loc[idx].sort_values('original')
                plot_df = pd.concat([plot_df,tmp])

            # prepare for the plotting
            plot_df.drop_duplicates(subset = ['effect','predict','quantile'], keep ='first')
            plot_df.reset_index(inplace=True)
            plot_df.rename(columns={'index':'feature'}, inplace=True)

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

    def bar_plot(self):
        '''
        Feature importance plot
        '''
        if self._task in ['r', 'reg', 'regression']:
            title = 'Barplot of feature importance: regression'
        else:
            title = 'Barplot of feature importance: classification'

        fig = px.bar(round(self._feature_importance.reset_index().sort_values('importance').rename(columns={'index':'feature'}),3), 
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
            return self._local_meta.drop(columns='size')     
        else: 
            return self._meta.drop(columns='size')     
            
    def baseline_values(self, local=False):
        '''
        Expose the baseline vector used for AcME
        '''
        if local:
            return self._local_baseline
        else:
            return self._global_baseline
