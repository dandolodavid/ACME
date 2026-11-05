"""
End-to-end checks of ACME on the synthetic datasets of make_datasets.py, where the right answer is known.

Run:  python -m pytest dataset/test_acme.py -q      (needs scikit-learn and pytest)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from ACME.ACME import ACME  # noqa: E402
import make_datasets  # noqa: E402


def load(name):
    if not (HERE / f'{name}.csv').exists():
        make_datasets.main()
    return pd.read_csv(HERE / f'{name}.csv')


def spearman(a, b):
    return pd.Series(np.asarray(a, dtype=float)).corr(pd.Series(np.asarray(b, dtype=float)), method='spearman')


# ----------------------------------------------------------------------------------------------------------------
# Regression: y = 4*x1 - 2*x2 + 1.5*c + 0*x3
# ----------------------------------------------------------------------------------------------------------------
REG_FEATURES = ['x1', 'x2', 'x3', 'c']


@pytest.fixture(scope='module')
def reg():
    df = load('regression')
    model = LinearRegression().fit(df[REG_FEATURES].values, df['y'])
    data = df[REG_FEATURES].copy()
    data['pred'] = model.predict(data.values)
    acme = ACME(model, 'pred', features=REG_FEATURES, cat_features=['c'], K=50).explain(data)
    return model, data, acme


def test_reg_ranking(reg):
    _, _, acme = reg
    imp = acme.feature_importance()['importance']
    assert list(imp.index) == ['x1', 'x2', 'c', 'x3']
    assert imp['x3'] < 0.01 * imp['x1']  # noise feature


def test_reg_direction(reg):
    _, _, acme = reg
    table = acme.summary_table()
    assert spearman(table.loc['x1', 'quantile'], table.loc['x1', 'effect']) > 0.99
    assert spearman(table.loc['x2', 'quantile'], table.loc['x2', 'effect']) < -0.99


def test_reg_baseline(reg):
    model, data, acme = reg
    baseline = acme.baseline_values()
    assert np.allclose(baseline[['x1', 'x2', 'x3']].values[0], data[['x1', 'x2', 'x3']].mean().values)
    assert baseline['c'].values[0] == data['c'].mode()[0]
    expected = model.predict(baseline[REG_FEATURES].values.astype(float))[0]
    assert np.allclose(acme.summary_table()['baseline_prediction'], expected)


def test_reg_local_matches_linear_model(reg):
    # for a linear model, moving feature f from the observed value v to value w changes the prediction by coef_f * (w - v)
    model, data, acme = reg
    obs = data.loc[7, REG_FEATURES]
    acme.explain_local(obs)
    table = acme.summary_table(local=True)
    assert np.isclose(table['baseline_prediction'].iloc[0], model.predict(obs.values.reshape(1, -1))[0])
    for feature, coef in zip(REG_FEATURES, model.coef_):
        rows = table.loc[feature]
        assert np.allclose(rows['predict'] - rows['baseline_prediction'], coef * (rows['original'] - obs[feature]), atol=1e-8)
    assert acme.feature_importance(local=True).index[0] == 'x1'


def test_reg_column_order_does_not_matter(reg):
    model, data, acme = reg
    shuffled = data[REG_FEATURES[::-1] + ['pred']]
    other = ACME(model, 'pred', features=REG_FEATURES, cat_features=['c'], K=50).explain(shuffled)
    pd.testing.assert_frame_equal(other.feature_importance(), acme.feature_importance())


def test_reg_features_inferred_from_dataframe(reg):
    model, data, acme = reg
    other = ACME(model, 'pred', cat_features=['c'], K=50).explain(data)
    pd.testing.assert_frame_equal(other.feature_importance(), acme.feature_importance())


def test_reg_robust_grid(reg):
    model, data, _ = reg
    table = ACME(model, 'pred', features=REG_FEATURES, cat_features=['c'], K=20).explain(data, robust=True).summary_table()
    numeric = table[table['type_feature'] == 'numeric']
    # the grid spans [0.05, 0.95]; the extra exact baseline quantile lies inside it
    assert numeric['quantile'].min() >= 0.05 - 1e-9 and numeric['quantile'].max() <= 0.95 + 1e-9


def test_reg_plots(reg):
    _, data, acme = reg
    acme.explain_local(data.loc[3, REG_FEATURES])
    for fig in [acme.summary_plot(), acme.summary_plot(local=True), acme.summary_plot(n_features=2),
                acme.bar_plot(), acme.bar_plot(local=True), acme.feature_exploration('x1', plot=True),
                acme.feature_exploration('x1', local=True, plot=True)]:
        assert len(fig.data) > 0
    assert len(acme.bar_plot(n_features=2).data[0].y) == 2


# ----------------------------------------------------------------------------------------------------------------
# Classification: class 0 / 1 / 2 by x1 thresholds, x2 and x3 noise
# ----------------------------------------------------------------------------------------------------------------
CLS_FEATURES = ['x1', 'x2', 'x3']


@pytest.fixture(scope='module')
def cls():
    df = load('classification')
    model = LogisticRegression(max_iter=1000).fit(df[CLS_FEATURES].values, df['y'])
    data = df[CLS_FEATURES].copy()
    data['pred'] = model.predict(data.values)
    return model, data


@pytest.mark.parametrize('label', [0, 1, 2])
def test_cls_x1_most_important_for_every_class(cls, label):
    model, data = cls
    acme = ACME(model, 'pred', features=CLS_FEATURES, task='class', K=30).explain(data, label_class=label)
    imp = acme.feature_importance()['importance']
    assert imp.index[0] == 'x1'
    assert imp['x1'] > 10 * imp[['x2', 'x3']].max()
    predict = acme.summary_table()['predict']
    assert predict.between(0, 1).all()


def test_cls_direction(cls):
    model, data = cls
    up = ACME(model, 'pred', features=CLS_FEATURES, task='class').explain(data, label_class=2).summary_table().loc['x1']
    down = ACME(model, 'pred', features=CLS_FEATURES, task='class').explain(data, label_class=0).summary_table().loc['x1']
    assert spearman(up['quantile'], up['predict']) > 0.99
    assert spearman(down['quantile'], down['predict']) < -0.99


def test_cls_label_as_string(cls):
    model, data = cls
    a = ACME(model, 'pred', features=CLS_FEATURES, task='class').explain(data, label_class=2)
    b = ACME(model, 'pred', features=CLS_FEATURES, task='class').explain(data, label_class='2')
    pd.testing.assert_frame_equal(a.feature_importance(), b.feature_importance())
    with pytest.raises(ValueError):
        ACME(model, 'pred', features=CLS_FEATURES, task='class').explain(data, label_class=7)


def test_cls_multiclass(cls):
    model, data = cls
    acme = ACME(model, 'pred', features=CLS_FEATURES, task='class', K=20).explain(data)
    imp = acme.feature_importance()
    per_class = imp[['importance_class_0', 'importance_class_1', 'importance_class_2']]
    assert np.allclose(per_class.sum(axis=1), imp['importance'])
    assert imp.index[0] == 'x1'
    assert acme.summary_plot().data[0].type == 'bar'

    # a local explanation must not change the global plot, and must equal the one from a single-class fit
    obs = data.loc[0, CLS_FEATURES]
    acme.explain_local(obs, label_class=2)
    assert acme.summary_plot().data[0].type == 'bar'
    single = ACME(model, 'pred', features=CLS_FEATURES, task='class', K=20).explain(data, label_class=2)
    single.explain_local(obs, label_class=2)
    pd.testing.assert_frame_equal(acme.feature_importance(local=True), single.feature_importance(local=True))
    assert len(acme.summary_table(local=True).loc['x1']) == len(single.summary_table(local=True).loc['x1'])
    assert np.isclose(acme.summary_table(local=True)['baseline_prediction'].iloc[0],
                      model.predict_proba(obs.values.reshape(1, -1))[0, 2])
    assert len(acme.summary_plot(local=True).data) > 0


# ----------------------------------------------------------------------------------------------------------------
# Anomaly detection: Isolation Forest, anomalies shifted only in a1 (type 1) or only in a2 (type 2)
# ----------------------------------------------------------------------------------------------------------------
AD_FEATURES = ['a1', 'a2', 'a3', 'a4']


def score_function(model, data):
    # map decision_function to [0, 1]: > 0.5 means anomaly
    return (-model.decision_function(data.reshape(-1, len(AD_FEATURES))) + 1) / 2


@pytest.fixture(scope='module')
def ad():
    df = load('anomaly')
    model = IsolationForest(random_state=0).fit(df[AD_FEATURES].values)
    data = df[AD_FEATURES].copy()
    data['score'] = score_function(model, data.values)
    acme = ACME(model, 'score', features=AD_FEATURES, task='ad', score_function=score_function, K=50)
    acme.explain(data, robust=True)
    return df['anomaly_type'], data, acme


def test_ad_scores_separate_anomalies(ad):
    kind, data, _ = ad
    assert data['score'].between(0, 1).all()
    assert (data.loc[kind > 0, 'score'] > 0.5).mean() > 0.9
    assert (data.loc[kind == 0, 'score'] > 0.5).mean() < 0.1


@pytest.mark.parametrize('anomaly_type, feature', [(1, 'a1'), (2, 'a2')])
def test_ad_local_explanation_finds_shifted_feature(ad, anomaly_type, feature):
    kind, data, acme = ad
    anomalies = data.loc[kind == anomaly_type]
    others = [f for f in AD_FEATURES if f != feature]
    # "pure" anomalies: only the shifted feature is unusual. Others can be extreme by chance in a second feature too
    # (e.g. a3 = -3.3), and then fixing the shifted feature alone correctly does NOT make them normal (change = 0)
    pure = (anomalies[others].abs() < 2).all(axis=1)
    top, change = [], []
    for idx in anomalies.index:
        acme.explain_local(data.loc[idx, AD_FEATURES])
        imp = acme.feature_importance(local=True)
        top.append(imp.index[0] == feature)
        change.append(imp.loc[feature, 'change'] == 1)
    assert np.mean(top) >= 0.9
    assert np.mean(np.array(change)[pure.values]) >= 0.9  # bringing the feature back to typical values makes it normal

    # detailed checks on a typical anomaly (median score)
    idx = (anomalies['score'] - anomalies['score'].median()).abs().idxmin()
    acme.explain_local(data.loc[idx, AD_FEATURES])
    assert acme.feature_importance(local=True).index[0] == feature
    exploration = acme.feature_exploration(feature, local=True)
    assert {'normal', 'anomalies'} <= set(exploration['direction'])
    fig = acme.summary_plot(local=True)
    assert fig.layout.shapes[0].line.color == 'red'


@pytest.mark.parametrize('anomaly_type, feature', [(1, 'a1'), (2, 'a2')])
def test_ad_aggregated_local_importance(ad, anomaly_type, feature):
    kind, data, acme = ad
    acme.reset_local_sum_importance()
    acme.sum_local_scores(data.loc[kind == anomaly_type])
    bars = acme.bar_plot().data[0]
    assert bars.y[-1] == feature  # bars are sorted ascending: the last one is the most important
    acme.reset_local_sum_importance()


def test_ad_weights(ad):
    kind, data, acme = ad
    acme.explain_local(data.loc[data['score'].idxmax(), AD_FEATURES])
    only_delta = acme.feature_importance(local=True, weights={'ratio': 0, 'distance': 0, 'change': 0, 'delta': 1})
    assert np.allclose(only_delta['importance'], only_delta['delta'])
    with pytest.raises(ValueError):
        acme.feature_importance(local=True, weights={'change': 1})
    with pytest.raises(AttributeError):
        acme.feature_importance(local=True, weights={'ratio': 0.5, 'distance': 0.5, 'change': 0.5, 'delta': 0.5})


def test_ad_global_plots(ad):
    _, _, acme = ad
    for fig in [acme.summary_plot(), acme.bar_plot(), acme.feature_exploration('a1', plot=True)]:
        assert len(fig.data) > 0


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
