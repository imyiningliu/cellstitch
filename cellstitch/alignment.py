import ot
from cellpose.metrics import _label_overlap
from cellstitch.utils import *


class FramePair:
    def __init__(self, mask0, mask1, max_lbl=0):
        self.frame0 = mask0
        self.frame1 = mask1

        # store the max labels for stitching
        max_lbl_default = max(
            get_lbls(mask0).max(),
            get_lbls(mask1).max()
        )

        self.max_lbl = max(max_lbl, max_lbl_default)

    def get_plan(self, C):
        """
        Compute the transport plan between the two frames, given the cost matrix between the cells.
        """
        # get cell sizes
        sizes0 = get_sizes(self.frame0)
        sizes1 = get_sizes(self.frame1)

        # convert to distribution to compute transport plan
        dist0 = sizes0 / sum(sizes0)
        dist1 = sizes1 / sum(sizes1)

        # compute transportation plan
        plan = ot.emd(dist0, dist1, C)

        return plan

    def get_cost_matrix(self, overlap, epsilon=1e-10):
        """
        Return the cost matrix between cells in the two frame defined by IoU.
        """
        lbls0 = get_lbls(self.frame0)
        lbls1 = get_lbls(self.frame1)

        num_cells0 = len(lbls0)
        num_cells1 = len(lbls1)

        C = np.zeros((num_cells0, num_cells1))

        sizes0 = np.sum(overlap, axis=1)
        sizes1 = np.sum(overlap, axis=0)

        # for each pairs of cells, we want to compute the overlap proportion (intersect / min(size0, size1))
        for lbl0_index in range(num_cells0):
            for lbl1_index in range(num_cells1):
                lbl0, lbl1 = lbls0[lbl0_index], lbls1[lbl1_index]
                overlap_size = overlap[lbl0, lbl1]
                scaling_factor = overlap_size / (sizes0[lbl0] + sizes1[lbl1] - overlap_size)
                C[lbl0_index, lbl1_index] = 1 / (scaling_factor + epsilon)

        return C

    def get_stitched_mask(self):
        """Stitch frame1 using frame 0."""

        lbls0 = get_lbls(self.frame0)
        lbls1 = get_lbls(self.frame1)

        # get sizes
        overlap = _label_overlap(self.frame0, self.frame1)

        # compute matching
        C = self.get_cost_matrix(overlap)
        plan = self.get_plan(C)

        # get a soft matching from plan
        n, m = plan.shape
        soft_matching = np.zeros((n, m))

        for i in range(n):
            matched_index = plan[i].argmax()
            soft_matching[i, matched_index] = 1

        mask0, mask1 = self.frame0.mask, self.frame1.mask

        stitched_mask1 = np.zeros(mask1.shape)
        for lbl1_index in range(1, m):
            # find the cell with the lowest cost (i.e. lowest scaled distance)
            matching_filter = soft_matching[:, lbl1_index]
            filtered_C = C[:, lbl1_index].copy()
            filtered_C[matching_filter == 0] = np.Inf  # ignore the non-matched cells

            lbl0_index = np.argmin(filtered_C)  # this is the cell0 we will attempt to relabel cell1 with

            lbl0, lbl1 = lbls0[lbl0_index], lbls1[lbl1_index]

            if lbl0 != 0:
                stitched_mask1[mask1 == lbl1] = lbl0
            else:
                self.max_lbl += 1
                stitched_mask1[mask1 == lbl1] = self.max_lbl
        return stitched_mask1
