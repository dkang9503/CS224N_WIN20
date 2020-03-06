import numpy as np

# calculate f_score from a confusion matrix
def f_score(c):
    p = c[1, 1] / (c[1, 1] + c[0, 1])
    r = c[1, 1] / (c[1, 1] + c[0, 0])
    f = 2 * p * r / (p + r)
    return f, p, r

# calcualte the specificity, sensitivity, and youden's stat from a confusion matrix
def info(c):
    sen = c[1, 1] / (c[1, 1] + c[1, 0])
    spe = c[0, 0] / (c[0, 0] + c[0, 1])
    J = sen + spe - 1
    return J, sen, spe

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)