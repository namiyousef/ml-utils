def generate_recurrence_matrix(timeseries_array, threshold):
    n_points = timeseries_array.shape[0]
    recurrence_matrix = np.zeros((n_points, n_points))
    for i, value in enumerate(timeseries_array):
        recurrence_matrix[i] = np.heaviside(threshold - np.abs(value - timeseries_array), 0)

    return recurrence_matrix

def logistic_map(start, rate, limit):
    y = []
    for i in range(limit):
        y.append(start)
        start = rate*start*(1-start)
    return np.array(y)
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    import numpy as np

    n_points = 1000
    x = np.linspace(0, 2*np.pi, n_points)
    y = np.sin(x)
    y = np.random.normal(size=(n_points))
    plt.figure()
    plt.plot(x, y)
    plt.figure()

    recurrence_matrix = generate_recurrence_matrix(y, 0.01)

    plt.imshow(recurrence_matrix, cmap='Greys', interpolation='nearest', origin='lower')
    plt.show()
