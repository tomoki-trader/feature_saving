def root_mean_squared_percentage_error(y_true, y_pred, epsilon=1e-10):
    
    rmspe = np.sqrt(np.mean(np.square((y_true - y_pred) / (y_true + epsilon))))
    return rmspe
