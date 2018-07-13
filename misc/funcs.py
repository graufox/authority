from keras.metrics import top_k_categorical_accuracy
import numpy as np



# define top-3 accuracy metric
def top_3_acc(y_true, y_pred):
    """returns proportion of time that the actual was in top 3 predicted"""
    return top_k_categorical_accuracy(y_true, y_pred, k=3)



def expand_filtering(filtered_data, window_size):
    """"""
    if not type(filtered_data) == np.array:
        filtered_data = np.array(filtered_data)
    n = filtered_data.shape[0]
    transform_mat = np.zeros((n + window_size - 1, n))
    for i in range(window_size):
        for j in range(i + 1):
            transform_mat[i,j] = 1
    for i in range(window_size, n):
        for j in range(i - window_size + 1, i + 1):
            transform_mat[i,j] = 1
    for i in range(n, n + window_size):
        for j in range(i - window_size + 1, n):
            transform_mat[i,j] = 1

    for i in range(n + window_size - 1):
        transform_mat[i,:] /= transform_mat[i,:].sum()

    return np.dot(transform_mat, filtered_data)
