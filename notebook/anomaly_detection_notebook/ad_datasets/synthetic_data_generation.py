import numpy as np
import pandas as pd
from random import uniform 
from pyod.utils.data import generate_data, generate_data_clusters 


def generate_axis(n_samples:int, axis:int, axis_min:float, axis_max:float, size:int) -> np.ndarray: 
    """
    Generate n_samples of dim dimensions, where the axis-th dimension is uniformly sampled from axis_min to axis_max 
    and the other dimensions are sampled from a normal distribution with mean 0 and std 1.
    
    Args:
        n_samples (int):    number of samples to generate
        axis (int):         axis along with generate, from 0 to dim-1
        axis_min (float):   minimum value of the axis-th dimension
        axis_max (float):   maximum value of the axis-th dimension
        dim (int):         number of dimensions
    
    Returns:
        np.ndarray          samples, shape (n_samples, size)
    """

    if axis < 0 or axis >= size: 
        raise ValueError("axis must be between 0 and size-1")
    
    x = np.random.uniform(axis_min, axis_max, size=(n_samples, 1))
    data = np.random.normal(0, 1, size=(n_samples, size))
    data[:, axis] = x[:, 0]
    return data


def generate_bisec_2d(n_samples, radius_min, radius_max, theta, size): 
    """
    Uniformly sample radius between radius_min and radius_max
    Generate samples so that x0 = r*cos(theta) + n, x1 = r*sin(theta) + n and other dimensions are sampled from N(0,1)
    """
    radius = np.random.uniform(radius_min, radius_max, size=(n_samples, 1))
    x0 = radius * np.cos(theta) + np.random.normal(0, 1, size=(n_samples, 1))
    x1 = radius * np.sin(theta) + np.random.normal(0, 1, size=(n_samples, 1))
    x_others = np.random.normal(0, 1, size=(n_samples, size-2))
    return np.concatenate([x0, x1, x_others], axis=1)



def DIFFI_dataset(n_inliers:int, n_outliers:int, in_radius_interval:list, out_radius_interval:list, size:int) -> pd.DataFrame: 
    """
    Generate a dataset of size dimensions such that each sample x = [radius*cos(theta), radius*sin(theta), n1, ..., np] where
    - for inliers radius ~ U(inliers_radius_interval) and theta ~ U(0, 2*pi)
    - for outliers radius ~ U(outliers_radius_interval) and theta ~ U(0, 2*pi)
    and n1, ..., np ~  N(0,1)

    Args: 
        n_inliers (int): number of inliers
        n_outliers (int): number of outliers
        in_radius_interval (list): interval from which to sample the radius of inliers
        out_radius_interval (list): interval from which to sample the radius of outliers
        size (int): number of dimensions
    
    Returns:
        pd.DataFrame: dataset of shape (n_inliers + n_outliers, size)
    """
    # inliers 
    theta = np.random.uniform(0, 2*np.pi, size=(n_inliers, 1))
    radius = np.random.uniform(in_radius_interval[0], in_radius_interval[1], size=(n_inliers, 1))
    inliers = np.concatenate([radius * np.cos(theta), radius * np.sin(theta), np.random.normal(0, 1, size=(n_inliers, size-2))], axis=1)

    # outliers
    theta = np.random.uniform(0, 2*np.pi, size=(n_outliers, 1))
    radius = np.random.uniform(out_radius_interval[0], out_radius_interval[1], size=(n_outliers, 1))
    outliers = np.concatenate([radius * np.cos(theta), radius * np.sin(theta), np.random.normal(0, 1, size=(n_outliers, size-2))], axis=1)

    # add true label
    inliers = pd.DataFrame(inliers, columns=[f"x{i}" for i in range(size)])
    inliers["Target"] = 0
    outliers = pd.DataFrame(outliers, columns=[f"x{i}" for i in range(size)])
    outliers["Target"] = 1

    return pd.concat([inliers, outliers], axis=0)



def synthetic_datasets(keys_list: list): 
    # if n < 1 or n > 30: 
    #     raise ValueError("n must be between 1 and 30 included")
    if not isinstance(keys_list, list): 
        raise ValueError("keys_list must be a list of integers")
    
    # Dataset 1 
    inliers = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([5, 5, 5, 5, 5, 5]), np.eye(6), 900), np.zeros((900,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    outliers1  = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([20, 15, 10, 8, 5, 5]), np.eye(6), 50), np.ones((50,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    outliers2 = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([-20, 9, -2, 4, 6, 5]), np.eye(6), 50), np.ones((50,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    outliers = pd.concat([outliers1, outliers2], axis = 0)
    data1 = pd.concat([inliers, outliers], axis = 0)
    features1 = data1.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 2
    inliers = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([3, -5, 10, 10, 0, 0, 0, 0, 0]), np.eye(9), 900), np.zeros((900,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'Target'])
    outliers1  = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([3, -6, -30, 4, 38, 1, 0.4, 0.4, 0.5]), 2*np.eye(9), 50), np.ones((50,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'Target'])
    outliers2 = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([3, -6, 10, 10, 0, 10, 10, 0, 0]), 3.2*np.eye(9), 50), np.ones((50,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'Target'])
    outliers = pd.concat([outliers1, outliers2], axis = 0)
    data2 = pd.concat([inliers, outliers], axis = 0)
    features2 = data2.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 3
    inliers = pd.DataFrame(np.concatenate((np.random.beta(2, 5, (900, 3)), np.random.normal(0, 0.2, (900, 3)), np.zeros((900,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    outliers1  = pd.DataFrame(np.concatenate((np.random.beta(15, 2, (50, 3)), np.random.normal(0, 0.2, (50, 3)), np.ones((50,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    outliers2 = pd.DataFrame(np.concatenate((np.random.uniform(-0.7, -1.0, (50, 4)), np.random.normal(0, 0.2, (50, 2)), np.ones((50,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    data3 = pd.concat([inliers, outliers1, outliers2], axis = 0)
    features3 = data3.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 4
    inliers1 = pd.DataFrame(np.concatenate((np.random.uniform(-1, 1, (275, 4)), np.zeros((275,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'Target'])
    inliers2 = pd.DataFrame(np.concatenate((np.random.uniform(-1, 1, (275, 4)), np.zeros((275,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'Target'])
    outliers1 = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([0, -4, 3, 0.5]), 0.5*np.eye(4), 25), np.ones((25,1))), axis = 1), columns =  ['x0', 'x1', 'x2', 'x3', 'Target'])
    data4 = pd.concat([inliers1, inliers2, outliers1], axis = 0)
    features4 = data4.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 5
    X_train, X_test, y_train, y_test = generate_data_clusters(n_train = 1000, n_test = 10, n_features = 9, contamination = 0.20, random_state = 0)
    data5 = pd.DataFrame(np.concatenate((X_train, y_train.reshape(-1,1)), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7','x8', 'Target'])
    features5 = data5.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 6 
    X_train, X_test, y_train, y_test = generate_data(n_train = 200, n_test = 10, n_features = 24, contamination = 0.05, random_state = 1)
    data6 = pd.DataFrame(np.concatenate((X_train, y_train.reshape(-1,1)), axis = 1), columns = ['x'+str(i) for i in range(24)] + ['Target'])
    features6 = data6.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 7
    X_train, X_test, y_train, y_test = generate_data_clusters(n_train = 100, n_test = 1, n_features = 11, contamination = 0.30, random_state = 3)
    data7 = pd.DataFrame(np.concatenate((X_train, y_train.reshape(-1,1)), axis = 1), columns = ['x'+str(i) for i in range(11)] + ['Target'])
    features7 = data7.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 8: uniform distribution along x1, x3, x5 axis and gaussian distribution along x2, x4 axis
    inliers = np.zeros((500, 5))
    inliers[:, 0] = np.random.uniform(3, 7, 500)
    inliers[:, 1] = np.random.normal(5, 5, 500)
    inliers[:, 2] = np.random.uniform(0, 3, 500)
    inliers[:, 3] = np.random.normal(0, 1, 500)
    inliers[:, 4] = np.random.uniform(0, 3, 500)
    inliers = pd.DataFrame(np.concatenate((inliers, np.zeros((500, 1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'Target'])
    n_outliers = 10 
    outliers = np.zeros((n_outliers, 5))
    outliers[:, 0] = np.random.uniform(0, 10, n_outliers)
    outliers[:, 1] = np.random.normal(0, 1, n_outliers)
    outliers[:, 2] = np.random.uniform(0, 10, n_outliers)
    outliers[:, 3] = np.random.normal(0, 1, n_outliers)
    outliers[:, 4] = np.random.normal(0, 1, n_outliers)
    outliers = pd.DataFrame(np.concatenate((outliers, np.ones((n_outliers, 1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'Target'])
    data8 = pd.concat([inliers, outliers], axis=0)
    features8 = data8.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 9
    n_inliers = 250
    inliers = np.zeros((250, 5))
    inliers[:, 0] = np.sin(np.linspace(0, 25, n_inliers)) + np.random.normal(0, 0.05, n_inliers)
    inliers[:, 1] = np.cos(np.linspace(0, 25, n_inliers)) + np.random.normal(0, 0.05, n_inliers)
    inliers[:, 2] = np.random.uniform(5, 7, n_inliers)
    inliers[:, 3] = np.random.uniform(-1, 1, n_inliers)
    inliers[:, 4] = np.random.normal(0, 0.2, n_inliers)
    inliers = pd.DataFrame(np.concatenate((inliers, np.zeros((n_inliers, 1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'Target'])
    n_outliers = 17 
    outliers = np.zeros((n_outliers, 5))
    outliers[:,0] = np.random.normal(0, 1, n_outliers)
    outliers[:,1] = np.cos(np.linspace(0, 25, n_outliers) + np.random.normal(0, 1, n_outliers)) + np.random.normal(0, 0.1, n_outliers)
    outliers[:, 2] = np.random.normal(0, 1, n_outliers)
    outliers[:, 3] = np.random.uniform(-1.2, 0.9, n_outliers)
    outliers[:, 4] = np.random.normal(0, 0.2, n_outliers)
    outliers = pd.DataFrame(np.concatenate((outliers, np.ones((n_outliers, 1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'Target'])
    data9 = pd.concat([inliers, outliers], axis=0)
    features9 = data9.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 10 
    from sklearn.datasets import make_moons, make_blobs
    inliers = np.zeros((1000, 12))
    inliers[:, 0:2] = make_moons(n_samples=1000, noise=0.1, random_state=0)[0]
    inliers[:, 2:4] = make_blobs(n_samples=1000, n_features=2, centers=[[2, 2]], cluster_std=0.1, random_state=0)[0]
    inliers[:, 4:6] = make_blobs(n_samples=1000, n_features=2, centers=[[0, 0]], cluster_std=0.1, random_state=0)[0]
    inliers[:, 6:12] = np.random.multivariate_normal(np.array([0, 0, 0, 0, 0, 0]), np.eye(6), 1000)
    inliers = pd.DataFrame(np.concatenate([inliers, np.zeros((1000, 1))], axis=1), columns = ['x'+str(i) for i in range(12)] + ['Target'])

    outliers = np.zeros((50, 12))
    outliers[:, 0:2] = make_moons(n_samples=50, noise=0.12, random_state=0)[0]
    outliers[:, 2:10] = np.random.uniform(-2, 2, (50, 8))
    outliers[:, 10] = generate_axis(50, axis=0, axis_min=3, axis_max=5, size=1).reshape(1,-1)
    outliers[:, 11] = np.random.normal(0, 1, 50)
    outliers = pd.DataFrame(np.concatenate([outliers, np.ones((50, 1))], axis=1), columns = ['x'+str(i) for i in range(12)] + ['Target'])
    data10 = pd.concat([inliers, outliers], axis=0)
    features10 = data10.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 11
    inliers = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([3, -5, 10, 10, 0, 0, 0, 0, 0]), np.eye(9), 600), np.zeros((600,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'Target'])
    outliers1  = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([0, 0, 10, 0, 1, 1, 1, 2, 0]), np.eye(9), 13), np.ones((13,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'Target'])
    outliers2 = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([0, -10, 0, 0, 0, 0, 0, 0, 0]), np.eye(9), 5), np.ones((5,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'Target'])
    outliers = pd.concat([outliers1, outliers2], axis = 0)
    data11 = pd.concat([inliers, outliers], axis = 0)
    features11 = data11.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 12
    inliers1 = pd.DataFrame(np.concatenate((np.random.uniform(-1, 1, (200, 6)), np.zeros((200,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    inliers2 = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([0, 0, 10, 0, 1, 1]), np.eye(6), 300), np.ones((300,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    outliers = np.zeros((22, 6))
    outliers[:, 0] = np.random.uniform(-1, 1, 22)
    outliers[:, 1] = np.random.uniform(2, 1, 22)
    outliers[:, 2:6] = np.random.normal(0, 1, (22, 4))
    outliers = pd.DataFrame(np.concatenate((outliers, np.ones((22, 1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    data12 = pd.concat([inliers1, inliers2, outliers], axis = 0)
    features12 = data12.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 13
    inliers1 = pd.DataFrame(np.concatenate((np.random.uniform(0, 1, (78, 6)), np.zeros((78,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    inliers2 = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([0, 3, 0, 0, 5, 0]), np.eye(6), 56), np.ones((56,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    n_outliers = 11
    outliers = np.zeros((n_outliers, 6))
    outliers[:, 0] = np.random.uniform(-1, 1, n_outliers)
    outliers[:, 1] = np.random.uniform(2, 4, n_outliers)
    outliers[:, 2:6] = np.random.normal(0, 1, (n_outliers, 4))
    outliers = pd.DataFrame(np.concatenate([outliers, np.ones((n_outliers,1))], axis=1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    data13 = pd.concat([inliers1, inliers2, outliers], axis=0)
    features13 = data13.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 14
    inliers = pd.DataFrame(np.concatenate((np.random.beta(2, 5, (500, 2)), np.random.normal(0, 0.2, (500, 2)), np.zeros((500,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'Target'])
    outliers1  = pd.DataFrame(np.concatenate((np.random.beta(11, 1.4, (23, 2)), np.random.normal(3, 0.2, (23, 2)), np.ones((23,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'Target'])
    outliers2 = pd.DataFrame(np.concatenate((np.random.uniform(-0.7, -1.0, (50, 2)), np.random.normal(6, 0.2, (50, 2)), np.ones((50,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'Target'])
    data14 = pd.concat([inliers, outliers1, outliers2], axis = 0)
    features14 = data14.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 15: inliers are along xaxis and bisec, outliers along yaxis
    n_inliers = 800
    inliers1 = generate_axis(int(n_inliers/4), 0, axis_min = 10, axis_max = 40, size = 6)
    inliers2 = generate_axis(int(n_inliers/4), 0, axis_min = -40, axis_max = -10, size = 6)
    inliers3 = generate_bisec_2d(int(n_inliers/4), radius_min = 10, radius_max = 40, theta = np.pi/4, size = 6) 
    inliers4 = generate_bisec_2d(int(n_inliers/4), radius_min = -40, radius_max = -10, theta = np.pi/4, size = 6)
    inliers_np = np.concatenate([inliers1, inliers2, inliers3, inliers4], axis=0)
    inliers = pd.DataFrame(np.concatenate([inliers_np, np.zeros((n_inliers,1))], axis=1), columns=['x'+str(i) for i in range(6)] + ['Target'])
    n_outliers = 50
    theta = np.random.uniform(0, 2*np.pi, size=(n_outliers, 1))
    radius = np.random.uniform(0, 40, size=(n_outliers, 1))
    outliers = np.concatenate([radius * np.cos(theta), radius * np.sin(theta), np.random.normal(0, 1, size=(n_outliers, 4))], axis=1)
    outliers = np.concatenate([outliers, np.ones((n_outliers, 1))], axis = 1)
    data15 = pd.DataFrame(np.concatenate([inliers, outliers], axis=0), columns=['x'+str(i) for i in range(6)] + ['Target'])
    features15 = data15.drop(['Target'], axis = 1).columns.to_list()


    # Dataset 16
    inliers_np = np.concatenate((np.random.multivariate_normal(np.array([0]*32), np.eye(32), 356), np.zeros((356,1))), axis=1)
    inliers = pd.DataFrame(inliers_np, columns = ['x'+str(i) for i in range(32)]+['Target'])
    outliers_np = np.concatenate([np.random.multivariate_normal(np.array([20,15,10,8,5,5,1,1,10,4,0,0,0,1,0,0,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),np.eye(32), 27), np.ones((27,1))], axis=1)
    outliers = pd.DataFrame(outliers_np, columns = ['x'+str(i) for i in range(32)]+['Target'])
    data16 = pd.concat([pd.DataFrame(inliers, columns=['x'+str(i) for i in range(32)]+['Target']), outliers], axis=0)

    features16 = data16.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 17
    theta = np.random.uniform(0, 2*np.pi, size=(300, 1))
    radius = np.random.uniform(0, 3, size=(300, 1))
    inliers = np.concatenate([radius * np.cos(theta), radius * np.sin(theta), np.random.normal(0, 1, size=(300, 4))], axis=1)
    inliers = pd.DataFrame( np.concatenate([inliers, np.zeros((300, 1))], axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5','Target'])
    outliers = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([3, -6, 10, 10, 0, 10]), np.eye(6), 50), np.ones((50,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5','Target'])
    data17 = pd.concat([inliers, outliers], axis = 0)
    features17 = data17.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 18
    inliers = pd.DataFrame(np.concatenate((np.random.beta(5, 1, (789, 3)), np.random.normal(0, 0.2, (789, 3)), np.zeros((789,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    outliers1  = pd.DataFrame(np.concatenate((np.random.beta(1.6, 8, (30, 3)), np.random.normal(0, 0.2, (30, 3)), np.ones((30,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    data18 = pd.concat([inliers, outliers1], axis = 0)
    features18 = data18.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 19
    inliers1 = pd.DataFrame(np.concatenate((np.random.uniform(-1, 1, (275, 3)), np.zeros((275,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'Target'])
    inliers2 = pd.DataFrame(np.concatenate((np.random.uniform(8, 12, (275, 3)), np.zeros((275,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'Target'])
    outliers1 = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([7, -4, 3, ]), 0.5*np.eye(3), 8), np.ones((8,1))), axis = 1), columns =  ['x0', 'x1', 'x2', 'Target'])
    data19 = pd.concat([inliers1, inliers2, outliers1], axis = 0)
    features19 = data19.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 20
    X_train, X_test, y_train, y_test = generate_data_clusters(n_train = 1000, n_test = 10, n_features = 11, contamination = 0.20, random_state = 3)
    data20 = pd.DataFrame(np.concatenate((X_train, y_train.reshape(-1,1)), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7','x8', 'x9', 'x10', 'Target'])
    features20 = data20.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 21
    X_train, X_test, y_train, y_test = generate_data(n_train = 300, n_test = 10, n_features = 16, contamination = 0.13, random_state = 5)
    data21 = pd.DataFrame(np.concatenate((X_train, y_train.reshape(-1,1)), axis = 1), columns = ['x'+str(i) for i in range(16)] + ['Target'])
    features21 = data21.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 22
    X_train, X_test, y_train, y_test = generate_data_clusters(n_train = 260, n_test = 1, n_features = 6, contamination = 0.22, random_state = 7)
    data22 = pd.DataFrame(np.concatenate((X_train, y_train.reshape(-1,1)), axis = 1), columns = ['x'+str(i) for i in range(6)] + ['Target'])
    features22 = data22.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 23: uniform distribution along x1, x3, x5 axis and gaussian distribution along x2, x4 axis
    inliers = np.zeros((400, 5))
    inliers[:, 0] = np.random.normal(0, 0, 400)
    inliers[:, 1] = np.random.uniform(-2, -4, 400)
    inliers[:, 2] = np.random.normal(0, 1, 400)
    inliers[:, 3] = np.random.normal(0, 1, 400)
    inliers[:, 4] = np.random.uniform(0, 3, 400)
    inliers = pd.DataFrame(np.concatenate((inliers, np.zeros((400, 1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'Target'])
    outliers = np.zeros((16, 5))
    outliers[:, 0] = np.random.uniform(0, 10, 16)
    outliers[:, 1] = np.random.normal(0, 1, 16)
    outliers[:, 2] = np.random.uniform(0, 10, 16)
    outliers[:, 3] = np.random.uniform(0, 10, 16)
    outliers[:, 4] = np.random.normal(0, 1, 16)
    outliers = pd.DataFrame(np.concatenate((outliers, np.ones((16, 1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'Target'])
    data23 = pd.concat([inliers, outliers], axis=0)
    features23 = data23.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 24
    inliers = np.zeros((170, 5))
    inliers[:, 0] = np.sin(np.linspace(0, 25, 170)) + np.random.normal(0, 0.05, 170)
    inliers[:, 1] = np.cos(np.linspace(0, 25, 170)) + np.random.normal(0, 0.05, 170)
    inliers[:, 2] = np.sin(np.linspace(0, 25, 170)) + np.random.normal(0, 0.05, 170)
    inliers[:, 3] = np.random.uniform(-1, 1, 170)
    inliers[:, 4] = np.random.normal(0, 0.2, 170)
    inliers = pd.DataFrame(np.concatenate((inliers, np.zeros((170, 1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'Target'])
    outliers = np.zeros((10, 5))
    outliers[:,0] = np.random.normal(0, 1, 10)
    outliers[:,1] = np.cos(np.linspace(0, 25, 10) + np.random.normal(0, 1, 10)) + np.random.normal(0, 0.1, 10)
    outliers[:, 2] = np.random.normal(0, 1, 10)
    outliers[:, 3] = np.random.uniform(-1.2, 0.9, 10)
    outliers[:, 4] = np.random.normal(0, 0.2, 10)
    outliers = pd.DataFrame(np.concatenate((outliers, np.ones((10, 1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'Target'])
    data24 = pd.concat([inliers, outliers], axis=0)
    features24 = data24.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 25 
    from sklearn.datasets import make_moons, make_blobs
    inliers = np.zeros((1300, 10))
    inliers[:, 0:2] = make_moons(n_samples=1300, noise=0.1, random_state=0)[0]
    inliers[:, 2:4] = make_blobs(n_samples=1300, n_features=2, centers=[[2, 2]], cluster_std=0.1, random_state=0)[0]
    inliers[:, 4:6] = make_blobs(n_samples=1300, n_features=2, centers=[[0, 0]], cluster_std=0.1, random_state=0)[0]
    inliers[:, 6:10] = np.random.multivariate_normal(np.array([0, 0, 0, 0]), np.eye(4), 1300)
    inliers = pd.DataFrame(np.concatenate([inliers, np.zeros((1300, 1))], axis=1), columns = ['x'+str(i) for i in range(10)] + ['Target'])

    outliers = np.zeros((100, 10))
    outliers[:, 0:2] = make_moons(n_samples=100, noise=0.3, random_state=0)[0]
    outliers[:, 2:7] = np.random.uniform(-2, 2, (100, 5))
    outliers[:, 8] = generate_axis(100, axis=0, axis_min=3, axis_max=5, size=1).reshape(1,-1)
    outliers[:, 9] = np.random.uniform(-2, 2, 100)
    outliers = pd.DataFrame(np.concatenate([outliers, np.ones((100, 1))], axis=1), columns = ['x'+str(i) for i in range(10)] + ['Target'])
    data25 = pd.concat([inliers, outliers], axis=0)
    features25 = data25.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 26
    X_train, X_test, y_train, y_test = generate_data_clusters(n_train = 400, n_test = 1, n_features = 5, contamination = 0.15, random_state = 8)
    data26 = pd.DataFrame(np.concatenate((X_train, y_train.reshape(-1,1)), axis = 1), columns = ['x'+str(i) for i in range(5)] + ['Target'])
    features26 = data26.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 27
    X_train, X_test, y_train, y_test = generate_data_clusters(n_train = 300, n_test = 1, n_features = 18, contamination = 0.18, random_state = 9)
    data27 = pd.DataFrame(np.concatenate((X_train, y_train.reshape(-1,1)), axis = 1), columns = ['x'+str(i) for i in range(18)] + ['Target'])
    features27 = data27.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 28
    inliers1 = pd.DataFrame(np.concatenate((np.random.uniform(-3, 1.5, (100, 6)), np.zeros((100,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    inliers2 = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([2, 3, 0, 0, 5, 0]), np.eye(6), 60), np.ones((60,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    n_outliers = 20
    outliers = np.zeros((n_outliers, 6))
    outliers[:, 0] = np.random.uniform(-1, 1, n_outliers)
    outliers[:, 1] = np.random.normal(0, 1, n_outliers)
    outliers[:, 2] = np.random.uniform(2, 4, n_outliers)
    outliers[:, 2:6] = np.random.normal(0, 1, (n_outliers, 4))
    outliers = pd.DataFrame(np.concatenate([outliers, np.ones((n_outliers,1))], axis=1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    data28 = pd.concat([inliers1, inliers2, outliers], axis=0)
    features28 = data28.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 29
    inliers_np = np.concatenate([np.random.uniform(-2,2,(500,4)),np.zeros((500,1))], axis=1)
    inliers = pd.DataFrame(inliers_np, columns = ['x0', 'x1', 'x2', 'x3', 'Target'])
    outliers = pd.DataFrame(np.concatenate([make_blobs(n_samples=45, n_features=4, centers=[[6, 1, 10, 0]], cluster_std=0.3, random_state=0)[0], np.ones((45,1))], axis=1), columns = ['x0', 'x1', 'x2', 'x3', 'Target'])
    data29 = pd.concat([inliers, outliers], axis = 0)
    features29 = data29.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 30
    inliers1 = pd.DataFrame(np.concatenate([np.random.uniform(-2,2,(500,4)), np.zeros((500,1))], axis=1), columns = ['x0', 'x1', 'x2', 'x3', 'Target'])
    inliers2 = pd.DataFrame(np.concatenate([np.random.uniform(10,13,(300,4)), np.zeros((300,1))], axis=1), columns = ['x0', 'x1', 'x2', 'x3', 'Target'])
    inliers = pd.concat([inliers1, inliers2], axis=0)
    outliers = pd.DataFrame(np.concatenate([make_blobs(n_samples=45, n_features=4, centers=[[6, 1, 8, 0]], cluster_std=0.3, random_state=0)[0], np.ones((45,1))], axis=1), columns = ['x0', 'x1', 'x2', 'x3', 'Target'])
    data30 = pd.concat([inliers, outliers], axis = 0)
    features30 = data30.drop(['Target'], axis = 1).columns.to_list()


    datasets = {1: [data1, features1],
                2: [data2, features2],
                3: [data3, features3],
                4: [data4, features4],
                5: [data5, features5],
                6: [data6, features6],
                7: [data7, features7],
                8: [data8, features8],
                9: [data9, features9],
                10: [data10, features10],
                11: [data11, features11],
                12: [data12, features12],
                13: [data13, features13],
                14: [data14, features14],
                15: [data15, features15],
                16: [data16, features16],
                17: [data17, features17],
                18: [data18, features18],
                19: [data19, features19],
                20: [data20, features20],
                21: [data21, features21],
                22: [data22, features22],
                23: [data23, features23],
                24: [data24, features24],
                25: [data25, features25],
                26: [data26, features26],
                27: [data27, features27],
                28: [data28, features28],
                29: [data29, features29],
                30: [data30, features30]
    }
    # return {k: datasets[k] for k in list(datasets)[:n]}
    # return the dataset with the specified keys 
    return {k: datasets[k] for k in keys_list}


def synthetic_datasets2(keys_list: list): 
    # if n < 1 or n > 30: 
    #     raise ValueError("n must be between 1 and 30 included")
    if not isinstance(keys_list, list): 
        raise ValueError("keys_list must be a list of integers")
    
    # Dataset 1 
    inliers = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([5, 5, 5, 5, 5, 5]), np.eye(6), 900), np.zeros((900,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    outliers1  = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([20, 15, 10, 8, 5, 5]), np.eye(6), 50), np.ones((50,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    outliers2 = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([-20, 9, -2, 4, 6, 5]), np.eye(6), 50), np.ones((50,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    outliers = pd.concat([outliers1, outliers2], axis = 0)
    data1 = pd.concat([inliers, outliers], axis = 0)
    features1 = data1.drop(['Target'], axis = 1).columns.to_list()
    true_rankings1 = [1, 2, 3, 4, 5, 6]

    # Dataset 2
    inliers = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([3, -5, 10, 10, 0, 0, 0, 0, 0]), np.eye(9), 900), np.zeros((900,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'Target'])
    outliers1  = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([3, -6, -30, 4, 38, 1, 0.4, 0.4, 0.5]), np.eye(9), 50), np.ones((50,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'Target'])
    outliers2 = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([3, -6, 10, 10, 0, 10, 10, 0, 0]), np.eye(9), 50), np.ones((50,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'Target'])
    outliers = pd.concat([outliers1, outliers2], axis = 0)
    data2 = pd.concat([inliers, outliers], axis = 0)
    features2 = data2.drop(['Target'], axis = 1).columns.to_list()


    # Dataset 3
    inliers = pd.DataFrame(np.concatenate((np.random.beta(2, 5, (900, 3)), np.random.normal(0, 0.2, (900, 3)), np.zeros((900,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    outliers1  = pd.DataFrame(np.concatenate((np.random.beta(15, 2, (50, 3)), np.random.normal(0, 0.2, (50, 3)), np.ones((50,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    outliers2 = pd.DataFrame(np.concatenate((np.random.uniform(-0.7, -1.0, (50, 4)), np.random.normal(0, 0.2, (50, 2)), np.ones((50,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    data3 = pd.concat([inliers, outliers1, outliers2], axis = 0)
    features3 = data3.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 4
    inliers1 = pd.DataFrame(np.concatenate((np.random.uniform(-1, 1, (275, 4)), np.zeros((275,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'Target'])
    inliers2 = pd.DataFrame(np.concatenate((np.random.uniform(-1, 1, (275, 4)), np.zeros((275,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'Target'])
    outliers1 = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([0, -4, 3, 0.5]), 0.5*np.eye(4), 25), np.ones((25,1))), axis = 1), columns =  ['x0', 'x1', 'x2', 'x3', 'Target'])
    data4 = pd.concat([inliers1, inliers2, outliers1], axis = 0)
    features4 = data4.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 5
    X_train, X_test, y_train, y_test = generate_data_clusters(n_train = 1000, n_test = 10, n_features = 9, contamination = 0.20, random_state = 0)
    data5 = pd.DataFrame(np.concatenate((X_train, y_train.reshape(-1,1)), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7','x8', 'Target'])
    features5 = data5.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 6 
    X_train, X_test, y_train, y_test = generate_data(n_train = 200, n_test = 10, n_features = 24, contamination = 0.05, random_state = 1)
    data6 = pd.DataFrame(np.concatenate((X_train, y_train.reshape(-1,1)), axis = 1), columns = ['x'+str(i) for i in range(24)] + ['Target'])
    features6 = data6.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 7
    X_train, X_test, y_train, y_test = generate_data_clusters(n_train = 100, n_test = 1, n_features = 11, contamination = 0.30, random_state = 3)
    data7 = pd.DataFrame(np.concatenate((X_train, y_train.reshape(-1,1)), axis = 1), columns = ['x'+str(i) for i in range(11)] + ['Target'])
    features7 = data7.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 8: uniform distribution along x1, x3, x5 axis and gaussian distribution along x2, x4 axis
    inliers = np.zeros((500, 5))
    inliers[:, 0] = np.random.uniform(3, 7, 500)
    inliers[:, 1] = np.random.normal(5, 5, 500)
    inliers[:, 2] = np.random.uniform(0, 3, 500)
    inliers[:, 3] = np.random.normal(0, 1, 500)
    inliers[:, 4] = np.random.uniform(0, 3, 500)
    inliers = pd.DataFrame(np.concatenate((inliers, np.zeros((500, 1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'Target'])
    n_outliers = 10 
    outliers = np.zeros((n_outliers, 5))
    outliers[:, 0] = np.random.uniform(0, 10, n_outliers)
    outliers[:, 1] = np.random.normal(0, 1, n_outliers)
    outliers[:, 2] = np.random.uniform(0, 10, n_outliers)
    outliers[:, 3] = np.random.normal(0, 1, n_outliers)
    outliers[:, 4] = np.random.normal(0, 1, n_outliers)
    outliers = pd.DataFrame(np.concatenate((outliers, np.ones((n_outliers, 1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'Target'])
    data8 = pd.concat([inliers, outliers], axis=0)
    features8 = data8.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 9
    n_inliers = 250
    inliers = np.zeros((250, 5))
    inliers[:, 0] = np.sin(np.linspace(0, 25, n_inliers)) + np.random.normal(0, 0.05, n_inliers)
    inliers[:, 1] = np.cos(np.linspace(0, 25, n_inliers)) + np.random.normal(0, 0.05, n_inliers)
    inliers[:, 2] = np.random.uniform(5, 7, n_inliers)
    inliers[:, 3] = np.random.uniform(-1, 1, n_inliers)
    inliers[:, 4] = np.random.normal(0, 0.2, n_inliers)
    inliers = pd.DataFrame(np.concatenate((inliers, np.zeros((n_inliers, 1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'Target'])
    n_outliers = 17 
    outliers = np.zeros((n_outliers, 5))
    outliers[:,0] = np.random.normal(0, 1, n_outliers)
    outliers[:,1] = np.cos(np.linspace(0, 25, n_outliers) + np.random.normal(0, 1, n_outliers)) + np.random.normal(0, 0.1, n_outliers)
    outliers[:, 2] = np.random.normal(0, 1, n_outliers)
    outliers[:, 3] = np.random.uniform(-1.2, 0.9, n_outliers)
    outliers[:, 4] = np.random.normal(0, 0.2, n_outliers)
    outliers = pd.DataFrame(np.concatenate((outliers, np.ones((n_outliers, 1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'Target'])
    data9 = pd.concat([inliers, outliers], axis=0)
    features9 = data9.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 10 
    from sklearn.datasets import make_moons, make_blobs
    inliers = np.zeros((1000, 12))
    inliers[:, 0:2] = make_moons(n_samples=1000, noise=0.1, random_state=0)[0]
    inliers[:, 2:4] = make_blobs(n_samples=1000, n_features=2, centers=[[2, 2]], cluster_std=0.1, random_state=0)[0]
    inliers[:, 4:6] = make_blobs(n_samples=1000, n_features=2, centers=[[0, 0]], cluster_std=0.1, random_state=0)[0]
    inliers[:, 6:12] = np.random.multivariate_normal(np.array([0, 0, 0, 0, 0, 0]), np.eye(6), 1000)
    inliers = pd.DataFrame(np.concatenate([inliers, np.zeros((1000, 1))], axis=1), columns = ['x'+str(i) for i in range(12)] + ['Target'])

    outliers = np.zeros((50, 12))
    outliers[:, 0:2] = make_moons(n_samples=50, noise=0.12, random_state=0)[0]
    outliers[:, 2:10] = np.random.uniform(-2, 2, (50, 8))
    outliers[:, 10] = generate_axis(50, axis=0, axis_min=3, axis_max=5, size=1).reshape(1,-1)
    outliers[:, 11] = np.random.normal(0, 1, 50)
    outliers = pd.DataFrame(np.concatenate([outliers, np.ones((50, 1))], axis=1), columns = ['x'+str(i) for i in range(12)] + ['Target'])
    data10 = pd.concat([inliers, outliers], axis=0)
    features10 = data10.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 11
    inliers = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([3, -5, 10, 10, 0, 0, 0, 0, 0]), np.eye(9), 600), np.zeros((600,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'Target'])
    outliers1  = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([0, 0, 10, 0, 1, 1, 1, 2, 0]), np.eye(9), 13), np.ones((13,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'Target'])
    outliers2 = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([0, -10, 0, 0, 0, 0, 0, 0, 0]), np.eye(9), 5), np.ones((5,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'Target'])
    outliers = pd.concat([outliers1, outliers2], axis = 0)
    data11 = pd.concat([inliers, outliers], axis = 0)
    features11 = data11.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 12
    inliers1 = pd.DataFrame(np.concatenate((np.random.uniform(-1, 1, (200, 6)), np.zeros((200,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    inliers2 = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([0, 0, 10, 0, 1, 1]), np.eye(6), 300), np.ones((300,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    outliers = np.zeros((22, 6))
    outliers[:, 0] = np.random.uniform(-1, 1, 22)
    outliers[:, 1] = np.random.uniform(2, 1, 22)
    outliers[:, 2:6] = np.random.normal(0, 1, (22, 4))
    outliers = pd.DataFrame(np.concatenate((outliers, np.ones((22, 1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    data12 = pd.concat([inliers1, inliers2, outliers], axis = 0)
    features12 = data12.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 13
    inliers1 = pd.DataFrame(np.concatenate((np.random.uniform(0, 1, (78, 6)), np.zeros((78,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    inliers2 = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([0, 3, 0, 0, 5, 0]), np.eye(6), 56), np.ones((56,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    n_outliers = 11
    outliers = np.zeros((n_outliers, 6))
    outliers[:, 0] = np.random.uniform(-1, 1, n_outliers)
    outliers[:, 1] = np.random.uniform(2, 4, n_outliers)
    outliers[:, 2:6] = np.random.normal(0, 1, (n_outliers, 4))
    outliers = pd.DataFrame(np.concatenate([outliers, np.ones((n_outliers,1))], axis=1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    data13 = pd.concat([inliers1, inliers2, outliers], axis=0)
    features13 = data13.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 14
    inliers = pd.DataFrame(np.concatenate((np.random.beta(2, 5, (500, 2)), np.random.normal(0, 0.2, (500, 2)), np.zeros((500,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'Target'])
    outliers1  = pd.DataFrame(np.concatenate((np.random.beta(11, 1.4, (23, 2)), np.random.normal(3, 0.2, (23, 2)), np.ones((23,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'Target'])
    outliers2 = pd.DataFrame(np.concatenate((np.random.uniform(-0.7, -1.0, (50, 2)), np.random.normal(6, 0.2, (50, 2)), np.ones((50,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'Target'])
    data14 = pd.concat([inliers, outliers1, outliers2], axis = 0)
    features14 = data14.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 15: inliers are along xaxis and bisec, outliers along yaxis
    n_inliers = 800
    inliers1 = generate_axis(int(n_inliers/4), 0, axis_min = 10, axis_max = 40, size = 6)
    inliers2 = generate_axis(int(n_inliers/4), 0, axis_min = -40, axis_max = -10, size = 6)
    inliers3 = generate_bisec_2d(int(n_inliers/4), radius_min = 10, radius_max = 40, theta = np.pi/4, size = 6) 
    inliers4 = generate_bisec_2d(int(n_inliers/4), radius_min = -40, radius_max = -10, theta = np.pi/4, size = 6)
    inliers_np = np.concatenate([inliers1, inliers2, inliers3, inliers4], axis=0)
    inliers = pd.DataFrame(np.concatenate([inliers_np, np.zeros((n_inliers,1))], axis=1), columns=['x'+str(i) for i in range(6)] + ['Target'])
    n_outliers = 50
    theta = np.random.uniform(0, 2*np.pi, size=(n_outliers, 1))
    radius = np.random.uniform(0, 40, size=(n_outliers, 1))
    outliers = np.concatenate([radius * np.cos(theta), radius * np.sin(theta), np.random.normal(0, 1, size=(n_outliers, 4))], axis=1)
    outliers = np.concatenate([outliers, np.ones((n_outliers, 1))], axis = 1)
    data15 = pd.DataFrame(np.concatenate([inliers, outliers], axis=0), columns=['x'+str(i) for i in range(6)] + ['Target'])
    features15 = data15.drop(['Target'], axis = 1).columns.to_list()


    # Dataset 16
    inliers_np = np.concatenate((np.random.multivariate_normal(np.array([0]*32), np.eye(32), 356), np.zeros((356,1))), axis=1)
    inliers = pd.DataFrame(inliers_np, columns = ['x'+str(i) for i in range(32)]+['Target'])
    outliers_np = np.concatenate([np.random.multivariate_normal(np.array([20,15,10,8,5,5,1,1,10,4,0,0,0,1,0,0,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),np.eye(32), 27), np.ones((27,1))], axis=1)
    outliers = pd.DataFrame(outliers_np, columns = ['x'+str(i) for i in range(32)]+['Target'])
    data16 = pd.concat([pd.DataFrame(inliers, columns=['x'+str(i) for i in range(32)]+['Target']), outliers], axis=0)

    features16 = data16.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 17
    theta = np.random.uniform(0, 2*np.pi, size=(300, 1))
    radius = np.random.uniform(0, 3, size=(300, 1))
    inliers = np.concatenate([radius * np.cos(theta), radius * np.sin(theta), np.random.normal(0, 1, size=(300, 4))], axis=1)
    inliers = pd.DataFrame( np.concatenate([inliers, np.zeros((300, 1))], axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5','Target'])
    outliers = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([3, -6, 10, 10, 0, 10]), np.eye(6), 50), np.ones((50,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5','Target'])
    data17 = pd.concat([inliers, outliers], axis = 0)
    features17 = data17.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 18
    inliers = pd.DataFrame(np.concatenate((np.random.beta(5, 1, (789, 3)), np.random.normal(0, 0.2, (789, 3)), np.zeros((789,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    outliers1  = pd.DataFrame(np.concatenate((np.random.beta(1.6, 8, (30, 3)), np.random.normal(0, 0.2, (30, 3)), np.ones((30,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    data18 = pd.concat([inliers, outliers1], axis = 0)
    features18 = data18.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 19
    inliers1 = pd.DataFrame(np.concatenate((np.random.uniform(-1, 1, (275, 3)), np.zeros((275,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'Target'])
    inliers2 = pd.DataFrame(np.concatenate((np.random.uniform(8, 12, (275, 3)), np.zeros((275,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'Target'])
    outliers1 = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([7, -4, 3, ]), 0.5*np.eye(3), 8), np.ones((8,1))), axis = 1), columns =  ['x0', 'x1', 'x2', 'Target'])
    data19 = pd.concat([inliers1, inliers2, outliers1], axis = 0)
    features19 = data19.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 20
    X_train, X_test, y_train, y_test = generate_data_clusters(n_train = 1000, n_test = 10, n_features = 11, contamination = 0.20, random_state = 3)
    data20 = pd.DataFrame(np.concatenate((X_train, y_train.reshape(-1,1)), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7','x8', 'x9', 'x10', 'Target'])
    features20 = data20.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 21
    X_train, X_test, y_train, y_test = generate_data(n_train = 300, n_test = 10, n_features = 16, contamination = 0.13, random_state = 5)
    data21 = pd.DataFrame(np.concatenate((X_train, y_train.reshape(-1,1)), axis = 1), columns = ['x'+str(i) for i in range(16)] + ['Target'])
    features21 = data21.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 22
    X_train, X_test, y_train, y_test = generate_data_clusters(n_train = 260, n_test = 1, n_features = 6, contamination = 0.22, random_state = 7)
    data22 = pd.DataFrame(np.concatenate((X_train, y_train.reshape(-1,1)), axis = 1), columns = ['x'+str(i) for i in range(6)] + ['Target'])
    features22 = data22.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 23: uniform distribution along x1, x3, x5 axis and gaussian distribution along x2, x4 axis
    inliers = np.zeros((400, 5))
    inliers[:, 0] = np.random.normal(0, 0, 400)
    inliers[:, 1] = np.random.uniform(-2, -4, 400)
    inliers[:, 2] = np.random.normal(0, 1, 400)
    inliers[:, 3] = np.random.normal(0, 1, 400)
    inliers[:, 4] = np.random.uniform(0, 3, 400)
    inliers = pd.DataFrame(np.concatenate((inliers, np.zeros((400, 1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'Target'])
    outliers = np.zeros((16, 5))
    outliers[:, 0] = np.random.uniform(0, 10, 16)
    outliers[:, 1] = np.random.normal(0, 1, 16)
    outliers[:, 2] = np.random.uniform(0, 10, 16)
    outliers[:, 3] = np.random.uniform(0, 10, 16)
    outliers[:, 4] = np.random.normal(0, 1, 16)
    outliers = pd.DataFrame(np.concatenate((outliers, np.ones((16, 1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'Target'])
    data23 = pd.concat([inliers, outliers], axis=0)
    features23 = data23.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 24
    inliers = np.zeros((170, 5))
    inliers[:, 0] = np.sin(np.linspace(0, 25, 170)) + np.random.normal(0, 0.05, 170)
    inliers[:, 1] = np.cos(np.linspace(0, 25, 170)) + np.random.normal(0, 0.05, 170)
    inliers[:, 2] = np.sin(np.linspace(0, 25, 170)) + np.random.normal(0, 0.05, 170)
    inliers[:, 3] = np.random.uniform(-1, 1, 170)
    inliers[:, 4] = np.random.normal(0, 0.2, 170)
    inliers = pd.DataFrame(np.concatenate((inliers, np.zeros((170, 1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'Target'])
    outliers = np.zeros((10, 5))
    outliers[:,0] = np.random.normal(0, 1, 10)
    outliers[:,1] = np.cos(np.linspace(0, 25, 10) + np.random.normal(0, 1, 10)) + np.random.normal(0, 0.1, 10)
    outliers[:, 2] = np.random.normal(0, 1, 10)
    outliers[:, 3] = np.random.uniform(-1.2, 0.9, 10)
    outliers[:, 4] = np.random.normal(0, 0.2, 10)
    outliers = pd.DataFrame(np.concatenate((outliers, np.ones((10, 1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'Target'])
    data24 = pd.concat([inliers, outliers], axis=0)
    features24 = data24.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 25 
    from sklearn.datasets import make_moons, make_blobs
    inliers = np.zeros((1300, 10))
    inliers[:, 0:2] = make_moons(n_samples=1300, noise=0.1, random_state=0)[0]
    inliers[:, 2:4] = make_blobs(n_samples=1300, n_features=2, centers=[[2, 2]], cluster_std=0.1, random_state=0)[0]
    inliers[:, 4:6] = make_blobs(n_samples=1300, n_features=2, centers=[[0, 0]], cluster_std=0.1, random_state=0)[0]
    inliers[:, 6:10] = np.random.multivariate_normal(np.array([0, 0, 0, 0]), np.eye(4), 1300)
    inliers = pd.DataFrame(np.concatenate([inliers, np.zeros((1300, 1))], axis=1), columns = ['x'+str(i) for i in range(10)] + ['Target'])

    outliers = np.zeros((100, 10))
    outliers[:, 0:2] = make_moons(n_samples=100, noise=0.3, random_state=0)[0]
    outliers[:, 2:7] = np.random.uniform(-2, 2, (100, 5))
    outliers[:, 8] = generate_axis(100, axis=0, axis_min=3, axis_max=5, size=1).reshape(1,-1)
    outliers[:, 9] = np.random.uniform(-2, 2, 100)
    outliers = pd.DataFrame(np.concatenate([outliers, np.ones((100, 1))], axis=1), columns = ['x'+str(i) for i in range(10)] + ['Target'])
    data25 = pd.concat([inliers, outliers], axis=0)
    features25 = data25.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 26
    X_train, X_test, y_train, y_test = generate_data_clusters(n_train = 400, n_test = 1, n_features = 5, contamination = 0.15, random_state = 8)
    data26 = pd.DataFrame(np.concatenate((X_train, y_train.reshape(-1,1)), axis = 1), columns = ['x'+str(i) for i in range(5)] + ['Target'])
    features26 = data26.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 27
    X_train, X_test, y_train, y_test = generate_data_clusters(n_train = 300, n_test = 1, n_features = 18, contamination = 0.18, random_state = 9)
    data27 = pd.DataFrame(np.concatenate((X_train, y_train.reshape(-1,1)), axis = 1), columns = ['x'+str(i) for i in range(18)] + ['Target'])
    features27 = data27.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 28
    inliers1 = pd.DataFrame(np.concatenate((np.random.uniform(-3, 1.5, (100, 6)), np.zeros((100,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    inliers2 = pd.DataFrame(np.concatenate((np.random.multivariate_normal(np.array([2, 3, 0, 0, 5, 0]), np.eye(6), 60), np.ones((60,1))), axis = 1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    n_outliers = 20
    outliers = np.zeros((n_outliers, 6))
    outliers[:, 0] = np.random.uniform(-1, 1, n_outliers)
    outliers[:, 1] = np.random.normal(0, 1, n_outliers)
    outliers[:, 2] = np.random.uniform(2, 4, n_outliers)
    outliers[:, 2:6] = np.random.normal(0, 1, (n_outliers, 4))
    outliers = pd.DataFrame(np.concatenate([outliers, np.ones((n_outliers,1))], axis=1), columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'Target'])
    data28 = pd.concat([inliers1, inliers2, outliers], axis=0)
    features28 = data28.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 29
    inliers_np = np.concatenate([np.random.uniform(-2,2,(500,4)),np.zeros((500,1))], axis=1)
    inliers = pd.DataFrame(inliers_np, columns = ['x0', 'x1', 'x2', 'x3', 'Target'])
    outliers = pd.DataFrame(np.concatenate([make_blobs(n_samples=45, n_features=4, centers=[[6, 1, 10, 0]], cluster_std=0.3, random_state=0)[0], np.ones((45,1))], axis=1), columns = ['x0', 'x1', 'x2', 'x3', 'Target'])
    data29 = pd.concat([inliers, outliers], axis = 0)
    features29 = data29.drop(['Target'], axis = 1).columns.to_list()

    # Dataset 30
    inliers1 = pd.DataFrame(np.concatenate([np.random.uniform(-2,2,(500,4)), np.zeros((500,1))], axis=1), columns = ['x0', 'x1', 'x2', 'x3', 'Target'])
    inliers2 = pd.DataFrame(np.concatenate([np.random.uniform(10,13,(300,4)), np.zeros((300,1))], axis=1), columns = ['x0', 'x1', 'x2', 'x3', 'Target'])
    inliers = pd.concat([inliers1, inliers2], axis=0)
    outliers = pd.DataFrame(np.concatenate([make_blobs(n_samples=45, n_features=4, centers=[[6, 1, 8, 0]], cluster_std=0.3, random_state=0)[0], np.ones((45,1))], axis=1), columns = ['x0', 'x1', 'x2', 'x3', 'Target'])
    data30 = pd.concat([inliers, outliers], axis = 0)
    features30 = data30.drop(['Target'], axis = 1).columns.to_list()


    datasets = {1: [data1, features1],
                2: [data2, features2],
                3: [data3, features3],
                4: [data4, features4],
                5: [data5, features5],
                6: [data6, features6],
                7: [data7, features7],
                8: [data8, features8],
                9: [data9, features9],
                10: [data10, features10],
                11: [data11, features11],
                12: [data12, features12],
                13: [data13, features13],
                14: [data14, features14],
                15: [data15, features15],
                16: [data16, features16],
                17: [data17, features17],
                18: [data18, features18],
                19: [data19, features19],
                20: [data20, features20],
                21: [data21, features21],
                22: [data22, features22],
                23: [data23, features23],
                24: [data24, features24],
                25: [data25, features25],
                26: [data26, features26],
                27: [data27, features27],
                28: [data28, features28],
                29: [data29, features29],
                30: [data30, features30]
    }
    # return {k: datasets[k] for k in list(datasets)[:n]}
    # return the dataset with the specified keys 
    return {k: datasets[k] for k in keys_list}



# import pandas as pd
# import numpy as np

# # Calcolo delle medie delle feature per inliers e outliers
# inliers_mean = data2[data2['Target'] == 0].mean()
# outliers_mean = data2[data2['Target'] == 1].mean()

# # Calcolo delle differenze medie
# mean_diff = (inliers_mean - outliers_mean).abs()

# # Ordinamento delle differenze medie in ordine decrescente
# mean_diff_sorted = mean_diff.sort_values(ascending=False)

# # Stampare le feature in ordine di importanza
# print("Feature in ordine di importanza per la discriminazione tra inliers e outliers:")
# print(mean_diff_sorted.index.tolist())
