# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 19:49:45 2026

@author: Junxuan Ma
"""

#Semi-automatically matching calcium imaging pixels with pixels in the immunofluoresent (IF) stained images
"""
The matching is performed through a registration process involving stretching and shifting, 
    while torsion and rotation are not considered. The goal is to align the pixel locations 
    in the calcium imaging image with those in the IF image. In other words, the objective 
    is to maximize the number of co-localized pixels (i.e., pixels with the same XY coordinates) 
    that are positive in both the calcium imaging and IF images. For example, if pixels 
    identified as cells (positive) in the calcium imaging image correspond to cell pixels 
    in the IF image, the matching is considered successful.

In this regard, IFmatch is composed of the following steps:
    1. Downsize the IF images if the resolution is different from that of calcium imaging.
    
    2. Binarize the IF images by optimizing the IF intensity threshold to obtain cell shapes 
        comparable to those in the calcium imaging.
       Matching_plot_IF_threshold is helpful for deciding the IF intensity threshold.
       
    3. Use manual shifting and stretching and plot to understand the stretching factor.
        This stretching is usually needed because of microscope errors, where the downsizing 
        performed in step one is insufficient to achieve equal scaling.
        It can be assumed that IF images taken via the same microscope can use the same stretching factor
       Manual_regi_plot is to decide on the stretching factors based on manual registration. 
           This is by inputting stretching factor a and c (fold change) in horizontal and vertical dimension
           with corresponding shifter factor b and d (unit pixel) in horizontal and vertical dimension.
        
    4. Use automatic shifting correction of IF image through a simple optimization function. 
        The objective function maximizes the number of shared positive pixels.
    5. Optionally check the matching between calcium imaging and IF.
    6. Output a dataframe of X and Y coordinates of calcium imaging and an additional column of 
        IF intensities.
"""
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
class IFmatch:
    def __init__(
        self,
        Calcium_imaging_df: pd.DataFrame,
        CGRP_raw: np.ndarray,
        NF_raw: np.ndarray,
        Calcium_height: int = 292, 
        Calcium_width: int = 384  ):
        
        self.df_clust= Calcium_imaging_df
        self.height= Calcium_height
        self.width= Calcium_width
        
        #The IF image is scaled to match the dimensions of the calcium imaging image.
        self.cgrp= cv2.resize(CGRP_raw,dsize=(self.width,self.height),interpolation=cv2.INTER_AREA)
        self.nf= cv2.resize(NF_raw,dsize=(self.width,self.height),interpolation=cv2.INTER_AREA)
    
    def _if_threshold(self, channel, IF_threshold) -> np.ndarray:
        return (channel >= IF_threshold).astype(np.uint8)
    
    def _mask_matrix_to_df(self, mask: np.ndarray) -> pd.DataFrame:
        ys, xs = np.nonzero(mask)
        IF_df = pd.DataFrame({'X': xs.astype(float), 'Y': ys.astype(float)})     
        return IF_df
     
    def Matching_plot_IF_threshold(self, 
                      IF_threshold_list: list, 
                      n_rows: int,
                      which_channel: str = 'NF'):
        if which_channel == 'NF':
            channel = self.nf
        elif which_channel == 'CGRP':
            channel = self.cgrp
        else: raise ValueError(f"channel must be 'NF' or 'CGRP'. Got: {which_channel}")
        
        n_plots = len(IF_threshold_list)
        n_cols = (n_plots + n_rows - 1) // n_rows

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), squeeze=False)
        for i, thresh in enumerate(IF_threshold_list):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            mask = self._if_threshold(channel, thresh)   # shape (h,w), values 0 or 1   
            ys, xs = np.nonzero(mask)    # Only coordinates where pixel == 1 (usually << 112k)
            ax.scatter(xs, ys,
                       color='blue', s=1,  lw=0,
                       label='IF positive')
            ax.scatter(self.df_clust['X'], self.df_clust['Y'],
                       color='red', s=1, alpha=0.3, edgecolor='none',
                       label='Ca²⁺ clusters')        
            ax.set_title(f'Threshold = {thresh}')
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.axis('off')
        # Hide unused panels
        for j in range(i+1, n_rows * n_cols):
            r = j // n_cols
            c = j % n_cols
            axes[r, c].set_visible(False)       
        fig.tight_layout()


    
    def _registrate_linear(self,  IF_bi_df, a,b,  c,d):
        #Input IF_bi_df should be a binary mask
        IF_regi= IF_bi_df.copy()          
        IF_regi["X"]= IF_regi["X"]*a + b
        IF_regi["Y"]= IF_regi["Y"]*c + d
        return IF_regi
    
    def Manual_regi_plot(self, IF_threshold,  a,b, c,d, which_channel: str = 'NF',):
        if which_channel == 'NF':
            channel = self.nf
        elif which_channel == 'CGRP':
            channel = self.cgrp
        else: raise ValueError(f"channel must be 'NF' or 'CGRP'. Got: {which_channel}")
        
        mask = self._if_threshold(channel, IF_threshold)
        IF_df= self._mask_matrix_to_df(mask)
        IF_regi= self._registrate_linear(IF_df, a,b, c,d)
        
        
        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(self.width/10, self.height/20))
        
        ax1.scatter( IF_regi['X'], IF_regi['Y'],  color='blue', s=1)      
        ax1.scatter( self.df_clust["X"], self.df_clust["Y"], color='red',alpha=0.3, s=2)
        ax1.set_title('Post-registration', fontsize=11)
        ax1.set(xlim= (0,self.width-1), ylim= (0,self.height-1))
        ax1.invert_yaxis()
        
        ax2.scatter(IF_df['X'], IF_df['Y'],  color='blue', s=1)
        ax2.scatter( self.df_clust["X"], self.df_clust["Y"], color='red',alpha=0.3, s=2)
        ax2.set_title('Pre-registration', fontsize=11)
        ax2.set(xlim= (0,self.width-1), ylim= (0,self.height-1))
        ax2.invert_yaxis()
   
        plt.tight_layout()

    
    
    def _registration_obj_fun(self, IF_reg_df, cal_df):
        #This is a common algorithm in calculating mask overlapse
        # zip() gives tupples of all combinations of X and Y. set() outputs set format removes duplicates
        IF_set  = set(zip(IF_reg_df['X'].round().astype(int), IF_reg_df['Y'].round().astype(int)))
        Cal_set = set(zip(cal_df['X'].round().astype(int), cal_df['Y'].round().astype(int)))
    
        intersection = len(IF_set & Cal_set)#& is only between sets
        return intersection / len(Cal_set)    
    
    def Registration_optimize(self,          
        IF_threshold,
        a = 1.0,                  # fixed scale in X
        c = 1.0,                  # fixed scale in Y
        step = 1.0,
        start_X = -30.0,
        end_X = 30.0,
        start_Y = -30.0,
        end_Y = 30.0,
        plot_result = True,
        which_channel: str = 'NF'):
        """
        Grid search over translation (b, d) with fixed stretching scale (a, c).
        Returns best (X_shift, Y_shift) and corresponding score.
        """
        if which_channel == 'NF':
           channel = self.nf
        elif which_channel == 'CGRP':
           channel = self.cgrp
        else: raise ValueError(f"channel must be 'NF' or 'CGRP'. Got: {which_channel}")
     
        if step <= 0:
           raise ValueError("step must be positive")
        # Generate empty data containers for shifts, scores, best shift and best score.
        x_shifts = np.arange(start_X, end_X + step/2, step)
        y_shifts = np.arange(start_Y, end_Y + step/2, step)
        scores = np.full((len(y_shifts), len(x_shifts)), np.nan)          # 2D grid for heatmap       
        best_score = -np.inf
        best_shift = (0.0, 0.0)
        shift_record = []
        obj_values = []

        mask = self._if_threshold(channel, IF_threshold)
        IF_df= self._mask_matrix_to_df(mask)
        for i_x, dx in enumerate(x_shifts):#enumerate output both indices and values
           for i_y, dy in enumerate(y_shifts):
                # Apply registration
                IF_reg = self._registrate_linear(IF_df, a, dx, c, dy)

                # Compute score
                score = self._registration_obj_fun(IF_reg, self.df_clust)
                scores[i_y, i_x] = score

                obj_values.append(score)
                shift_record.append((dx, dy))

                if score > best_score:
                    best_score = score
                    best_shift = (dx, dy)

        if plot_result and len(obj_values) > 1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

            # Heatmap of scores
            im = ax1.imshow(
                scores,
                extent=[x_shifts.min(), x_shifts.max(), y_shifts.min(), y_shifts.max()],
                origin='lower',
                aspect='auto',
                cmap='magma',
                interpolation='nearest'
            )
            ax1.plot(best_shift[0], best_shift[1], 'c*', markersize=18, label='Best')
            ax1.set_title('Objective function landscape')
            ax1.set_xlabel('X shift (pixels)')
            ax1.set_ylabel('Y shift (pixels)')
            plt.colorbar(im, ax=ax1, label='Overlap ratio')

            # 1D progression (useful for debugging order)
            ax2.plot(obj_values, '.-', color='teal', lw=1)
            ax2.set_xlabel('Evaluation #')
            ax2.set_ylabel('Score')
            ax2.set_title('Score during grid search')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()


        return best_shift, best_score
    
    
        
        
    
    def _matrix_to_df(self,
                  Coordinates: pd.DataFrame,
                  image: np.ndarray,
                  name: str = 'value') -> pd.DataFrame:
        ys = Coordinates['Y'].astype(int).to_numpy()
        xs = Coordinates['X'].astype(int).to_numpy()
        cluster= Coordinates['Cluster'].to_numpy()
        df = pd.DataFrame({'Y': ys, 'X': xs, "Cluster":cluster, name: image[ys, xs] })
        return df
    
    """The updated pixel should be 
    IF_regi["X1"]= IF_["X"]*a + b
    IF_regi["Y1"]= IF_["Y"]*c + d
    Now we know IF_regi["X1"] and IF_regi["Y1"], for they are the calcium imaging coordinates
    we need to solve IF_["X"] and IF_["Y"] that are the pixels in IF corresponding to the calcium pixel
    """
    
    def Merge_channels_in_df(self, a, b, c, d):

        merged = self.df_clust.copy()

        # calculate the transformed coordinates for indexing
        x_idx = ((merged["X"] - b) / a).clip(0, self.width-1).astype(int).copy()
        y_idx = ((merged["Y"] - d) / c).clip(0, self.height-1).astype(int).copy()

        # add IF values
        merged["NF"] = map_coordinates(self.nf, [y_idx, x_idx], order=1, mode='nearest')
        merged["CGRP"] = map_coordinates(self.cgrp, [y_idx, x_idx], order=1, mode='nearest')

        return merged
    
    


"""---------------------------------------------------------------------------------------------------------
A more efficient way is to do the registration in matrix"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class IFmatchMatrix:

    def __init__(
        self,
        Calcium_imaging_df: pd.DataFrame,
        CGRP_raw: np.ndarray,
        NF_raw: np.ndarray,
        Calcium_height: int = 292,
        Calcium_width: int = 384
    ):

        self.df_clust = Calcium_imaging_df
        self.height = Calcium_height
        self.width = Calcium_width

        # resize IF images
        self.cgrp = cv2.resize(CGRP_raw, (self.width, self.height),
                               interpolation=cv2.INTER_AREA)

        self.nf = cv2.resize(NF_raw, (self.width, self.height),
                             interpolation=cv2.INTER_AREA)

        # build calcium mask once
        self.cal_mask = np.zeros((self.height, self.width), dtype=np.uint8)

        ys = self.df_clust["Y"].astype(int).to_numpy()
        xs = self.df_clust["X"].astype(int).to_numpy()

        self.cal_mask[ys, xs] = 1


    def _if_threshold(self, image, threshold):

        return (image >= threshold).astype(np.uint8)


    def _affine_transform(self, mask, a, b, c, d):

        M = np.array([
            [a, 0, b],
            [0, c, d]
        ], dtype=np.float32)

        transformed = cv2.warpAffine(
            mask,
            M,
            (self.width, self.height),
            flags=cv2.INTER_NEAREST
        )

        return transformed


    def _registration_score(self, IF_mask):

        overlap = np.sum(IF_mask & self.cal_mask)

        return overlap / np.sum(self.cal_mask)


    def Registration_optimize(
        self,
        IF_threshold,
        a=1.0,
        c=1.0,
        step=1,
        start_X=-30,
        end_X=30,
        start_Y=-30,
        end_Y=30,
        which_channel="NF",
        plot_result=True
    ):

        if which_channel == "NF":
            channel = self.nf
        elif which_channel == "CGRP":
            channel = self.cgrp
        else:
            raise ValueError("channel must be NF or CGRP")

        IF_mask = self._if_threshold(channel, IF_threshold)

        x_shifts = np.arange(start_X, end_X + step, step)
        y_shifts = np.arange(start_Y, end_Y + step, step)

        scores = np.zeros((len(y_shifts), len(x_shifts)))

        best_score = -1
        best_shift = (0, 0)

        for ix, dx in enumerate(x_shifts):
            for iy, dy in enumerate(y_shifts):

                IF_trans = self._affine_transform(IF_mask, a, dx, c, dy)

                score = self._registration_score(IF_trans)

                scores[iy, ix] = score

                if score > best_score:
                    best_score = score
                    best_shift = (dx, dy)

        if plot_result:

            plt.figure(figsize=(6,5))
            plt.imshow(
                scores,
                extent=[x_shifts.min(), x_shifts.max(),
                        y_shifts.min(), y_shifts.max()],
                origin="lower",
                cmap="magma"
            )

            plt.scatter(best_shift[0], best_shift[1], c="cyan", s=120)

            plt.xlabel("X shift")
            plt.ylabel("Y shift")
            plt.title("Registration objective")
            plt.colorbar(label="overlap ratio")

            plt.show()

        return best_shift, best_score


    def Merge_channels_in_df(self, a, b, c, d):

        coords = self.df_clust.copy()

        # inverse transform
        coords["X"] = (coords["X"] - b) / a
        coords["Y"] = (coords["Y"] - d) / c

        xs = coords["X"].clip(0, self.width-1).astype(int)
        ys = coords["Y"].clip(0, self.height-1).astype(int)

        merged = self.df_clust.copy()

        merged["NF"] = self.nf[ys, xs]
        merged["CGRP"] = self.cgrp[ys, xs]

        return merged




"""Combine the IF with calcium imaging analysis outcome
A good merge function explaination
https://www.geeksforgeeks.org/python-pandas-merging-joining-and-concatenating/
1.  The first step is an "outer" merge of TU and CGRP, the output is all X,Y that 
    either positive for TU or CGRP, if only positive in TU, in CGRP it will be NA
2.  The second step is to do a "right" merge that calcium (Position) is on the right.
    In this way, only pixels of calcium will remain
    calcium pixels without either TU or CGRP will be NA at corresponding position
3.  The input is already segmented and registerred (regarding calcium)"""
def merge_df_channels(IF_TU_regi, IF_CGRP_regi, Calcium):    
    IF_com =pd.merge(IF_TU_regi.rename(columns={'IF':'TU'})[['X1','Y1','TU']],
                     IF_CGRP_regi.rename(columns={'IF':'CGRP'})[['X1','Y1','CGRP']], 
                     how="outer", on=['X1','Y1'])
    Cal_IF =pd.merge(Calcium[['X','Y','Neuron_indice','Base','Bra','CZP', 'Cap','Kcl']], 
                     IF_com.rename(columns={'X1':'X','Y1':'Y'}),
                     how="left", on=['X','Y'])
    return Cal_IF
