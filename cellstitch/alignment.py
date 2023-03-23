import ot
import numpy as np
from skimage import color
from cellstitch.frame import *
import matplotlib.pyplot as plt
from cellpose.metrics import _label_overlap


class FramePair:
    def __init__(self, mask0, mask1, max_lbl=0):
        self.frame0 = Frame(mask0)
        self.frame1 = Frame(mask1)

        # store the max labels for stitching
        max_lbl_default = max(
            self.frame0.get_lbls().max(),
            self.frame1.get_lbls().max()
        )

        self.max_lbl = max(max_lbl, max_lbl_default)

    def display(self):
        """
        Display frame0 and frame1 next to each other, with consistent colorings.
        """

        num_lbls = len(np.union1d(self.frame0.get_lbls(), self.frame1.get_lbls()))

        colors = np.random.random((num_lbls, 3))

        frames = np.array([self.frame0.mask, self.frame1.mask])
        rgb = color.label2rgb(frames, colors=colors, bg_label=0)

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].imshow(rgb[0])
        axes[1].imshow(rgb[1])

        plt.tight_layout()
        plt.show()

    def get_plan(self, C):
        """
        Compute the transport plan between the two frames, given the cost matrix between the cells.
        """
        # get cell sizes
        sizes0 = self.frame0.get_sizes()
        sizes1 = self.frame1.get_sizes()

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
        lbls0 = self.frame0.get_lbls()
        lbls1 = self.frame1.get_lbls()

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

    def stitch(self):
        """Stitch frame1 using frame 0."""

        lbls0 = self.frame0.get_lbls()
        lbls1 = self.frame1.get_lbls()

        # get sizes
        overlap = _label_overlap(self.frame0.mask, self.frame1.mask)

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
        self.frame1 = Frame(stitched_mask1)

    def stitch3D(self, z_stitched, epsilon=1e-10):

        lbls0 = self.frame0.get_lbls()
        lbls1 = self.frame1.get_lbls()

        overlap = _label_overlap(self.frame0.mask, fp.frame1.mask)

        # compute matching
        C = self.get_cost_matrix(overlap)

        label_indexes0, label_indexes1 = np.where(overlap != 0)  # only need to look at cells with overlap

        for label_index0, label_index1 in zip(label_indexes0, label_indexes1):
            if label_index0 != 0 and label_index1 != 0:  # only care about intersecting cells
                lbl0 = lbls0[label_index0]
                lbl1 = lbls1[label_index1]

                intersect_mask = (self.frame0.mask == lbl0) * (self.frame1.mask == lbl1)
                prop_stitched = z_stitched[np.where(intersect_mask)].sum() / intersect_mask.sum()

                C[label_index0][label_index1] /= (prop_stitched + epsilon)

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
        self.frame1 = Frame(stitched_mask1)


    ###################
    ### for testing ###
    ###################

    def display(self):
        """
        Display frame0 and frame1 next to each other, with consistent colorings.
        """

        num_lbls = len(np.union1d(self.frame0.get_lbls(), self.frame1.get_lbls()))

        colors = np.random.random((num_lbls, 3))

        frames = np.array([self.frame0.mask, self.frame1.mask])
        rgb = color.label2rgb(frames, colors=colors, bg_label=0)

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].imshow(rgb[0])
        axes[1].imshow(rgb[1])

        plt.tight_layout()
        plt.show()

    def stitch_euclidean(self):
        """Stitch frame1 using frame 0."""

        lbls0 = self.frame0.get_lbls()[1:]  # ignore background
        lbls1 = self.frame1.get_lbls()[1:]

        locs0 = self.frame0.get_locations()
        locs1 = self.frame1.get_locations()

        # compute matching
        C = ot.dist(locs0, locs1)
        # get cell sizes
        sizes0 = self.frame0.get_sizes()[1:]
        sizes1 = self.frame1.get_sizes()[1:]

        # convert to distribution to compute transport plan
        dist0 = sizes0 / sum(sizes0)
        dist1 = sizes1 / sum(sizes1)

        # compute transportation plan
        plan = ot.emd(dist0, dist1, C)

        # get a soft matching from plan
        n, m = plan.shape
        soft_matching = np.zeros((n, m))

        for i in range(n):
            matched_index = plan[i].argmax()
            soft_matching[i, matched_index] = 1

        for j in range(m):
            matched_index = plan[:, j].argmax()
            soft_matching[matched_index, j] = 1

        mask0, mask1 = self.frame0.mask, self.frame1.mask
        overlap = _label_overlap(self.frame0.mask, self.frame1.mask)

        stitched_mask1 = np.zeros(mask1.shape)
        for lbl1_index in range(m):
            # find the cell with the lowest cost (i.e. lowest scaled distance)
            matching_filter = soft_matching[:, lbl1_index]
            filtered_C = C[:, lbl1_index].copy()
            filtered_C[matching_filter == 0] = np.Inf  # ignore the non-matched cells

            lbl0_index = np.argmin(filtered_C)  # this is the cell0 we will attempt to relabel cell1 with

            lbl0, lbl1 = lbls0[lbl0_index], lbls1[lbl1_index]

            if overlap[lbl0][lbl1] != 0:
                stitched_mask1[mask1 == lbl1] = lbl0
            else:
                self.max_lbl += 1
                stitched_mask1[mask1 == lbl1] = self.max_lbl
        self.frame1 = Frame(stitched_mask1)

