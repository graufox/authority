from keras.metrics import top_k_categorical_accuracy

# define top-3 accuracy metric
def top_3_acc(y_true, y_pred):
    """returns proportion of time that the actual was in top 3 predicted"""
    return top_k_categorical_accuracy(y_true, y_pred, k=3)
