from scipy import ndimage as ndi
from cellpose import metrics as cp_metrics
from cellpose import utils as cp_utils
from cellmatch.alignment import *


def cellmatch_alignment_benchmark(filename):
    labels = np.load(filename)
    errors = []

    num_frame = labels.shape[0]
    prev_index = 0

    while Frame(labels[prev_index]).is_empty():
        prev_index += 1

    curr_index = prev_index + 1

    while curr_index < num_frame:
        if Frame(labels[curr_index]).is_empty():
            # if frame is empty, skip
            curr_index += 1
        else:
            error = benchmark_alignment(labels[prev_index].copy(), labels[curr_index].copy())
            errors.append(error)

            prev_index = curr_index
            curr_index += 1

    return errors


def cellpose_alignment_benchmark(filename, stitch_threshold):
    labels = np.load(filename)
    errors = []

    num_frame = labels.shape[0]

    prev_index = 0

    while (labels[prev_index]).sum() == 0:
        prev_index += 1

    curr_index = prev_index + 1

    while curr_index < num_frame:
        if (labels[curr_index]).sum() == 0:
            # if frame is empty, skip
            curr_index += 1
        else:
            mask0 = labels[prev_index]
            mask1 = labels[curr_index]

            stitched_masks = cp_utils.stitch3D(masks=[mask0, mask1], stitch_threshold=stitch_threshold)

            num_wrong_pixels = (stitched_masks[1] != mask1).sum()
            total_cell_pixels = (mask1 != 0).sum()

            error = num_wrong_pixels / total_cell_pixels

            errors.append(error)

            prev_index = curr_index
            curr_index += 1

    return errors


def benchmark_alignment(mask0, mask1):
    """
    Returns the proportion of incorrectly matched pixels;
    mask0 and mask1 are labels (i.e. mask1 is correctly aligned with mask0.
    """
    true_lbls = mask1.copy()

    pair = FramePair(mask0, mask1)
    pair.stitch()

    num_wrong_pixels = (true_lbls != pair.frame1.mask).sum()
    total_cell_pixels = (true_lbls != 0).sum()

    return num_wrong_pixels / total_cell_pixels


def agg_jaccard_index(labels, pred):
    """
    Calculate aggregated jaccard index
    """
    aji = cp_metrics.aggregated_jaccard_index(labels, pred)
    return aji.mean()


def avg_symmetric_surf_dist(mask, pred):
    """
    Calculate Average Symmetric Surface Distance for each paired mask label & prediction
    Ignore ground-truth masks without paired prediction (FN) with ASSD = 0
    """
    pred_lbls = _get_lbls(mask)
    lbl_map = {
        lbl: _match_lbls(mask, pred, lbl)
        for lbl in pred_lbls
    }
    
    assd = []
    for sc_lbl in lbl_map.keys():
        tg_lbl = lbl_map[sc_lbl]
        if tg_lbl != -1:
            assd.append(_calc_assd(mask == sc_lbl, pred == tg_lbl))
            
        break
    
    return np.mean(assd) if len(assd) > 0 else 0
    
    
def _calc_assd(mask, pred):
    # Take outline 
    mask_outline = cp_utils.masks_to_outlines(mask)
    pred_outline = cp_utils.masks_to_outlines(pred)
    
    # Calculate ASSD for each outline pixel
    m1 = np.ones_like(mask, dtype=np.uint8)
    m1[pred_outline] = 0
    m2 = np.ones_like(pred, dtype=np.uint8)
    m2[mask_outline] = 0
    
    dist1 = ndi.distance_transform_edt(m1)[np.nonzero(mask_outline)]
    dist2 = ndi.distance_transform_edt(m2)[np.nonzero(pred_outline)]
    assd = np.concatenate([dist1, dist2]).mean()
    
    return assd
    
    
def _get_lbls(labels):
    """Get unique labels"""
    return np.unique(labels)[1:]


def _match_lbls(source_mask, target_mask, sc_lbl):
    """
    Find matched label between source mask & target mask with largest IoU area,
    Return target label
    """
    coords = np.nonzero(source_mask == sc_lbl)
    cand_lbls, counts = np.unique(target_mask[coords], return_counts=True)  # Candidate labels
    
    if cand_lbls[0] == 0:
        cand_lbls = cand_lbls[1:]
        counts = counts[1:]
        
    if len(cand_lbls) == 0:
        tg_lbl == -1
    else:
        # First check whether argmax matched lbl has IoU exceeding threshold
        if cand_lbls[0] == 0: # Ignore label 0
            cand_lbls = cand_lbls[1:]
        
        max_intersect = counts.max()
        tg_lbl = cand_lbls[counts.argmax()]
        
    return tg_lbl

###########################
# Pipeline Benchmark Code #
###########################


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
                tp += 1

    fp = len(lbls_pred) - tp
    fn = len(lbls_true) - tp

    ap = tp / (tp + fp + fn)

    return [ap, tp, fp, fn]  # return as list for easy convert to dataframe

