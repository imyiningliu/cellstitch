from cellstitch.alignment import *
from cellpose.metrics import _label_overlap


def relabel_layer(masks, z, lbls):
    """
    Relabel the label in LBLS in layer Z of MASKS.
    """
    layer = masks[z]
    if z != 0:
        reference_layer = masks[z - 1]
    else:
        reference_layer = masks[z + 1]

    overlap = _label_overlap(reference_layer, layer)
    layer_lbls = np.unique(layer)

    for lbl in lbls:
        lbl1_index = np.where(layer_lbls == lbl)[0][0]
        lbl0 = np.argmax(overlap[:, lbl1_index])
        layer[layer == lbl] = lbl0


def overseg_correction(masks):
    lbls = np.unique(masks)[1:]

    # get a list of labels that need to be corrected
    layers_lbls = {}

    for lbl in lbls:
        existing_layers = np.sum(masks == lbl, axis=(1, 2), dtype="bool")
        depth = existing_layers.sum()

        if depth == 1:
            z = np.where(existing_layers != 0)[0][0]
            if z in layers_lbls.keys():
                layers_lbls[z].append(lbl)
            else:
                layers_lbls[z] = [lbl]

    for z, lbls in layers_lbls.items():
        relabel_layer(masks, z, lbls)


def full_stitch(masks, verbose=False):
    """
    Stitch masks in-place (top -> bottom).
    """
    num_frame = masks.shape[0]

    prev_index = 0
    max_lbl = 0

    while Frame(masks[prev_index]).is_empty():
        prev_index += 1

    curr_index = prev_index + 1

    while curr_index < num_frame:
        if Frame(masks[curr_index]).is_empty():
            # if frame is empty, skip
            curr_index += 1
        else:
            if verbose:
                print("===Stitching frame %s with frame %s ...===" % (curr_index, prev_index))

            fp = FramePair(masks[prev_index], masks[curr_index], max_lbl=max_lbl)
            fp.stitch()
            masks[curr_index] = fp.frame1.mask

            max_lbl = fp.max_lbl

            prev_index = curr_index
            curr_index += 1

    overseg_correction(masks)


def full_stitch_reverse(masks, verbose=False):
    """
    Stitch masks in-place (bottom -> top).
    """
    num_frame = masks.shape[0]

    prev_index = num_frame - 1
    max_lbl = 0

    while Frame(masks[prev_index]).is_empty():
        prev_index -= 1

    curr_index = prev_index - 1

    while curr_index > 0:
        if Frame(masks[curr_index]).is_empty():
            # if frame is empty, skip
            curr_index -= 1
        else:
            if verbose:
                print("===Stitching frame %s with frame %s ...===" % (curr_index, prev_index))

            fp = FramePair(masks[prev_index], masks[curr_index], max_lbl=max_lbl)
            fp.stitch()
            masks[curr_index] = fp.frame1.mask

            max_lbl = fp.max_lbl

            prev_index = curr_index
            curr_index -= 1

    overseg_correction(masks)