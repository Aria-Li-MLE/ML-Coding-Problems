def multi_class_confusion_matrix(y_true, y_pred, classes):
    """
    Compute multi-class confusion matrix
    
    Args:
        y_true: List of true class labels
        y_pred: List of predicted class labels
        classes: List of all possible classes (must include all labels)
        
    Returns:
        list: N x N confusion matrix where N is number of classes
        Format: Rows represent true classes, columns represent predicted classes
        
    Raises:
        ValueError: If inputs have different lengths or contain unseen classes
        
    Example:
        >>> y_true = ['cat', 'dog', 'cat']
        >>> y_pred = ['cat', 'cat', 'dog']
        >>> classes = ['cat', 'dog']
        >>> multi_class_confusion_matrix(y_true, y_pred, classes)
        [[1, 1], [1, 0]]
    """
    # Input validation
    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must be equal")
    if not all(c in classes for c in y_true) or not all(c in classes for c in y_pred):
        raise ValueError("All labels must appear in classes list")
    
    # Initialize matrix
    size = len(classes)
    cf = [[0] * size for _ in range(size)]
    class_map = {cls: idx for idx, cls in enumerate(classes)}
    
    # Populate matrix
    for true, pred in zip(y_true, y_pred):
        true_idx = class_map[true]
        pred_idx = class_map[pred]
        cf[true_idx][pred_idx] += 1
    
    return cf