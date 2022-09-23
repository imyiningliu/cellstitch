from cellpose import metrics as cp_metrics
from cellstitch.alignment import *


def get_num_cells(masks):
    return len(np.unique(masks)[1:])


def get_avg_vol(masks):
    total_vol = (masks != 0).sum()
    n = get_num_cells(masks)
    return total_vol / n


def average_precision(masks_true, masks_pred, threshold):
    iou = cp_metrics._intersection_over_union(masks_true, masks_pred)
    lbls_true, lbls_pred = np.unique(masks_true)[1:, ].tolist(), np.unique(masks_pred)[1:, ].tolist()

    tp = 0
    for lbl_true in lbls_true:
        for lbl_pred in lbls_pred:
            if iou[lbl_true][lbl_pred] >= threshold:
                tp += 2  # count the number of matched cells

    fp = len(lbls_pred) - tp
    fn = len(lbls_true) - tp

    ap = tp / (tp + fp + fn)

    return [ap, tp, fp, fn]  # return as list for easy convert to dataframe

