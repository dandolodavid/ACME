# ACME
ACME - Accelerated Model Explainability 


Model interpretability is nowadays a major field of research in machine learning, due to the increasing complexity of predictive model permitted by the technological improvements. In this paper, we propose ACME a simple procedure that studies the model behavior observing the changes in the model predictions' caused by using different quantiles of each variable used by the model.  To evaluate the impact of the predictions' changing, we introduce a new measure, named standardize effect, that keeps in count both the changing direction and the overall variable impact amplitude. Standardize effects are also used to compute the final scores that represent the importance of the features. We tested the procedure and we compare the results with the know model interpretability algorithm SHAP. The results of the procedure are very similar, both in term of results that in term of visualization, but considering the speed, ACME outperform SHAP in every situation, proving to be a very usable algorithm, in particular in prediction applications where time efficiency is crucial. Also, the algorithm presents the possibility to study a single observation prediction, giving a local perspective to how the model works, using a "what if" scenario to take real-time decisions.

## INSTALL 
Install with the command:

pip install statwolfACME

## ACME PACKAGE:

### 1) DOCS:

- ACME( model, target, quantitative_features = [], qualitative_features = [], K = 50, task = 'regression', score_function = None ):
    - model : the model object, it must have the *predict* method or the ad-hoc parameter *score_function* is required
    - target : str column name with the target features. Typically, it is the predicted features (regression and classification), while using the score function could be a particular column (example: in Anomaly detection, the column with the anomaly score)
    - quantitative_features :  list of string with the columns name for numerical features
    - qualitative_features : list of string with the columns name for categorical features
    - K : number of quantile used in the AcME procedure
    - task :  str with accepted values {'regression','reg','r','c','class','classification'}. It declares the task of the model. When score_function is not None, the parameters is not necessary
    - score_function : function that has as first input the model and second the input data to realize the prediction. It must return a numeric score

- acme.fit(dataframe, robust = False, label_class = None):
    - dataframe: input dataframe for the model
    - robust : bool, if True exclude the quantile under 0.05 and over 0.95 to remove possible outliers
    - label_class : when task is classification, the label of the predicted class must be specified

- acme.fit_local(dataframe,local, robust = False, label_class = None)
    - dataframe: input dataframe for the model
    - local: index of the dataframe row with the local observations we want to analyze 
    - robust : bool, if True exclude the quantile under 0.05 and over 0.95 to remove possible outliers
    - label_class : when task is classification, the label of the predicted class must be specified

- fitted_acme.summary_plot(local=False):
    - local : bool, if True return the local summary plot, else the global

- fitted_acme.bar_plot()

    return the feature importance barplot

- fitted_acme.feature_importance()

    return the dataframe with the feature importance score

- fitted_acme.summary_table()

    return table with all the info calculated by acme, like standardized effect, quantile with linked original values, etc. for global interpretability

- fitted_acme.local_table()

    return table with all the info calculated by acme, like standardized effect, quantile with linked original values, etc. for local interpretability

### 2) EXAMPLES

#### 2.1) REGRESSION 

The actual implementation works with model objects that have 'predict()' methods (sklearn style model).

``` python
acme_reg = ACME(model, 'target', K=50)
acme_reg = acme_reg.fit(dataset) 
```

``` python
acme_reg.summary_plot()
```

![ACME summary plot](image/readme/reg.png)

``` python
acme_reg.bar_plot()
```

![ACME bar plot](image/readme/bar.png)

##### 2.1.1) LOCAL

``` python
acme_local = acme_reg.fit_local(dataset, local=100)
acme_local.summary_plot(local=True)
```

![ACME local plot](image/readme/local.png)

#### 2.2) CLASSIFICATION

The actual implementation works with model objects that have 'predict_proba()' methods.
The classification acme version works as the regression, but requires to specify the class we are looking for explanation.

``` python
model.classes_
array([0, 1])
```

``` python
acme_clas = ACME(model, 'target', K=50, task = 'class', label_class = 1 )
acme_clas = acme_clas.fit(dataset) 
```

![ACME clas plot](image/readme/class.png)

#### 2.3) SCORE FUNCTION
The model in this case is an isolation forest model

```python

def score_function(model, data):
    return model.decision_function(data)

acme_ifo = ACME(ifo, 'AD_score', K=50, task='regression', score_function=score_function, quantitative_features=features)
acme_ifo = acme_ifo.fit(dataset, robust = True)
```