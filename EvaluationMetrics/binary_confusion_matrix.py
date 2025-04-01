def binary_confusion_matrix(y_true,y_pred):
    """
    Calculate confusion matrix for binary classification problems
    
    Args:
        y_true (list): List of true labels (0s and 1s)
        y_pred (list): List of predicted labels (0s and 1s)
        
    Returns:
        list: 2x2 confusion matrix as [[true_positives, false_negatives],
                                     [false_positives, true_negatives]]
    
    Example:
        >>> y_true = [1, 0, 1, 1, 0, 0]
        >>> y_pred = [1, 0, 0, 1, 1, 0]
        >>> binary_confusion_matrix(y_true, y_pred)
        [[2, 1], [1, 2]]
    """
    # Initialize counters
    tp = tn = fp = fn = 0

    # Calculate values in confusion matrix 
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 1 and p == 0:
            fn += 1
        elif t == 0 and p == 1:
            fp += 1
        else:
            tn += 1

    # Return as 2x2 matrix       
    return [[tp, fn],
            [fp, tn]]

