def binary_confusion_matrix(y_true,y_pred):

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
    cf = [[tp, fn],
          [fp,tn]]
    return cf

