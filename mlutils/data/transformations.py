def generate_recurrence_matrix(timeseries_array, threshold):
    n_points = timeseries_array.shape[0]
    recurrence_matrix = np.zeros((n_points, n_points))
    for i, value in enumerate(timeseries_array):
        recurrence_matrix[i] = np.heaviside(threshold - np.abs(value - timeseries_array), 0)

    return recurrence_matrix

def generate_recurrence_matrix_fast(timeseries_array, threshold):
    n_points = timeseries_array.shape[0]
    timeseries_matrix = np.array([timeseries_array for i in range(n_points)])

    recurrence_matrix = np.heaviside(threshold - np.abs(timeseries_matrix - timeseries_matrix.T), 0)
    return recurrence_matrix


def logistic_map(start, rate, limit):
    y = []
    for i in range(limit):
        y.append(start)
        start = rate*start*(1-start)
    return np.array(y)
if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    from pyts.datasets import load_gunpoint, load_coffee, load_basic_motions, load_pig_central_venous_pressure
    import numpy as np
    from sequentia.classifiers.knn import KNNClassifier
    from sklearn.neighbors import KNeighborsClassifier
    y = np.array([1, 2, 4, 5, 1])
    plt.figure()
    plt.imshow(generate_recurrence_matrix(y, 0.1),
                  interpolation='nearest', origin='lower')
    plt.figure()
    plt.imshow(recurrence_alt(y, 0.1),
                  interpolation='nearest', origin='lower')
    plt.show()
    raise Exception()

    X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)
    X_train, X_test, y_train, y_test = load_coffee(return_X_y=True)
    # how do we look into the multidimensional version for recurrence plot?
    #X_train, X_test, y_train, y_test = load_basic_motions(return_X_y=True)

    X_train, X_test, y_train, y_test = load_pig_central_venous_pressure(return_X_y=True)

    from mlutils.plotting.utils import calculate_ax_grid


    print('Loaded dataset')
    X = X_train
    y = y_train
    n_timeseries, n_features = X.shape
    nrows, ncols = calculate_ax_grid(n_timeseries)
    plt.figure()
    colors = ['blue', 'red', 'green']

    """for ts, y_ in zip(X, y):
        plt.plot(range(ts.shape[0]), ts,
                 #c=colors[y_]
    )

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    colors = ['Greys', 'Greens']
    '''cmap_map = {
        val: colors[i] for i, val in enumerate(np.unique(y))
    }'''
    for ax, ts, y_ in zip(axes.flatten(), X, y):
        recurrence_matrix = generate_recurrence_matrix(ts, 0.01)
        ax.imshow(recurrence_matrix,
                  #cmap=cmap_map[y_],
                  interpolation='nearest', origin='lower')"""

    # -- raw timeseries
    classifier = KNeighborsClassifier(n_neighbors=1)
    start = time.time()
    classifier.fit(X_train, y_train)
    print(f'Time taken to fit: {time.time() - start:.3g}')
    start = time.time()
    y_predict = classifier.predict(X_test)
    print(f'Time taken to predict: {time.time() - start:.3g}')
    print(np.mean(y_predict == y_test))


    classifier = KNeighborsClassifier()
    start = time.time()

    classifier.fit(np.array([generate_recurrence_matrix(x, threshold=0.01).flatten() for x in X_train]), y_train)
    print(f'Time taken to fit: {time.time() - start:.3g}')
    start = time.time()
    y_predict = classifier.predict(np.array([generate_recurrence_matrix(x, threshold=0.01).flatten() for x in X_test]))
    print(f'Time taken to predict: {time.time() - start:.3g}')
    print(np.mean(y_predict == y_test))

    '''# -- dynamic time warping
    classifier = KNNClassifier(5, np.unique(y_train))
    start = time.time()
    classifier.fit([x for x in X_train], y_train)
    print(f'Time taken to fit: {time.time() - start:.3g}')

    start = time.time()
    y_predict = classifier.predict(X_test)
    print(f'Time taken to predict: {time.time() - start:.3g}')
    print(np.mean(y_predict == y_test))'''


    plt.show()
    '''import numpy as np

    n_points = 1000
    x = np.linspace(0, 2*np.pi, n_points)
    y = np.sin(x)
    y = np.random.normal(size=(n_points))
    plt.figure()
    plt.plot(x, y)
    plt.figure()


    plt.show()'''
