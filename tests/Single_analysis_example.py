# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 12:06:27 2026
Revised on Sun Oct 04 11:36:05 2026

@author: Junxuan Ma

"""
---------------------To install calmono, run in PowerShell:------------------------------

from calmono import imgimport, imgseg, ifmatch, siganalysis
---------------------To import calmono, run in Python:------------------------------

import inspect
import calmono


help(calmono)# show all modules in this package

#Check what classes are included in each modulus
inspect.getmembers(imgimport, inspect.isclass)
inspect.getmembers(imgseg, inspect.isclass)
inspect.getmembers(ifmatch, inspect.isclass)
inspect.getmembers(siganalysis, inspect.isclass)



folder_path =  r"I:\ari\Projects\Neu-DISC\06_Data_RO\03_Data_DCB\Junxuan\Oscar\Method_figures"

#Get all file names in a Folder
calfiles= imgimport.FetchFiles(
    file_path = folder_path,
    file_tag ="Cal_",
    ).Get_filenames()

#Store the calcium imaging file name in czi_name, the Sample name in name.
czi_name=calfiles[0]
name = czi_name.replace("Cal_", "")



#Run the class ImportCZI and import data
importer = imgimport.ImportCZI(
        folder_path_cal=folder_path,
        folder_path_if=folder_path,
        Cal_name=czi_name,
        cal_tag="Cal_",
        if_tag="IF_"
        )

#Check if the IF file corresponding to name exists. This is useful for loop analysis, set if function to avoid breaking the loop.
importer.IF_exists()

#Asign immunofluoresent channels and read the IF data.
CGRP_raw, NF_raw = importer.Read_if(CGRP=1,NF=0)

#Asign calcium imaging data, the spacial size which will be needed in the downstream analysis.
cal_data, shape = importer.Read_cal()#shape: (time,height,width)

"""
Introduction of imgseg.py:

Generate dataframe informing pixel location and which neuron this pixel belong to.
    In this dataframe, each row represent a pixel.
Later this dataframe will be also used to store 
    immunofluorescent information, and
    calcium imaging quantification.
The pixels will be in the same order comparing dataframe with np.array, 
    which make the link between these 2 types of data.


    Raw calcium imaging data as 2D np.ndarray:

        Shape of (time, pixels).
                    axis 1 →  (pixels)
                    ┌──────┬──────┬──────┬──────┐
         axis 0     │ 0,0  │ 0,1  │ 0,2  │ 0,3  │
         ↓ (time)   ├──────┼──────┼──────┼──────┤
                    │ 1,0  │ 1,1  │ 1,2  │ 1,3  │
                    ├──────┼──────┼──────┼──────┤
                    │ 2,0  │ 2,1  │ 2,2  │ 2,3  │
                    └──────┴──────┴──────┴──────┘
    The dataframe output data (pixels, measured features) is in a long format:

                            measured features (columns) →
                ┌──────┬──────┬────────┬─────────┬──────────┬─────────┬──────────┐
                │  Y   │  X   │Cluster │  Ca_PH  │  Ca_freq │ IF_NF   │ IF_CGRP  │
                ├──────┼──────┼────────┼─────────┼──────────┼─────────┼──────────┤
         pixel0 │  10  │  22  │   1    │  0.85   │   4.2    │  120.3  │  88.1    │
         pixel1 │  10  │  23  │   1    │  0.81   │   3.9    │  118.7  │  90.4    │
         pixel2 │  11  │  22  │   1    │  0.79   │   4.1    │  121.0  │  89.2    │
         pixel3 │  45  │  67  │   2    │  1.02   │   5.6    │   95.4  │ 130.8    │
         pixel4 │  45  │  68  │   2    │ _0.98   │   5.2    │  _94.1  │ _128.9   │
          ...   │ ...  │ ...  │...     │   ...   │   ...    │   ...   │ ...      │
         pixelM │ 201  │ 350  │  37    │  0.64   │   3.1    │  105.2  │  77.5    │
                └──────┴──────┴────────┴─────────┴──────────┴─────────┴──────────┘


Sturcuture of the imgseg.py:
    BG_define (Defining background)
        ├── __init__
        ├── _apply_method
        ├── _compute_thre_mask

        ├── Single_curve_plot
        ├── Thre_plot
        ├── Thre_multi_plot
        └── Cut_background (Port to connect other analysis, e.g., ICA)

    PC_PROJ (Principle component analysis)
        ├── __init__
        ├── _get_full_spatial_component

        ├── Get_PCs
        ├── PCA_plot
        ├── PCA_multi_plot
        └── Dimension_reduction (Port to connect other analysis, e.g., ICA)

    Neuron_Cluster (DBSCAN clustering of pixels in spatial principle components)
        ├── __init__
        ├── _mask_matrix_to_df
        ├── _dbscan
        ├── _clusterwise_process
        ├── _pix_size_filter
        ├── _roundness  (static)
        ├── _roundness_filter

        ├── PC_DBSCAN
        ├── PC_DBSCAN_plot
        ├── PC_index_plot

        ├── __define_parameters__
        ├── _subtract_df1_from_df2
        ├── _z_project

        ├── Z_Project
        └── Z_Project_plot

    Neuron_Seg
        ├── __init__
        ├── _output_all_Cal
        ├── _get_border
        ├── _clusterwise_sub_devide

        ├── Output_core_border
        ├── Output_all_Cal
        └── Output_background

        
    The NeuronSeg algorithm contains the following steps:
    1. BG_define: defining background pixels based on threshold defined from the function of 
       Visual_functions.Check_all_threshold.
       
       Inputs (parameters): 
           Calcium_imaging : NDArray[np.float32] or NDArray[np.float64]
               2D array of shape (time, pixels). Can be either ΔF/F₀ or raw fluorescence.

           time_slice_thre1: time window used to threshold based on brightness.
               The time frames within this time window will be
               maximized/meaned/minimized and thresholded to generate mask

           time_slice_thre2_base and time_slice_thre2_peak: used to threshold based on response
               Time windows to define baseline and peak time series
               These two windows of time series will be maximized/meaned/minimized
                   over time to generate single base image and peak image, respectively.
               The mask is generated by thresholding the contrast of peak image - base image
               
           method_1: decide maximized/meaned/minimized of time window defined by time_slice_thre1
           method_2_base and method_2_peak: decide maximized/meaned/minimized of time windows
               defined by time_slice_thre2_base and time_slice_thre2_peak, respectively.

           dim_h and dim_w are the pixel size of calcium imaging.
                It is output of ImportCZI().read_Cal()
           
       Outputs (attributes): 
           Thre_plot() plot the binary mask after thresholding
           Thre_multi_plot() multi-panel plots trying combinations of thresholds
           Cut_background(self,dim_h,dim_w) output:
                A reduced calcium imaging np.array containing only pixels 
                    that are not regarded as background.             
                And a dataframe informing the X and Y position of all selected pixels
                    (pixels that are not regarded as background)
                    corresponding to the reduced calcium imaging data.
"""


# ---------Example of the usage: BG_define-------------------
# ---------Determine parameters------
#Determine the time windows for thresholding. 
#Average all pixels and draw a curve. It is useful to decide on
#    time_slice_thre1, time_slice_thre2_base and time_slice_thre2_peak.
imgseg.BG_define.Single_curve_plot(
    Calcium_imaging= cal_data,
    figsize= (20,5),
    x_interval= 200
    )
# From the displayed curve, we can decide the time windows for thresholding.
plt.close()

# From here enter the manually determined time windows for thresholding.                 
BG_def= imgseg.BG_define(
    Calcium_imaging= cal_data,
    dim_h= shape[1],
    dim_w= shape[2],
    time_slice_thre1= slice(
              12000, 12200
              ),
    time_slice_thre2_base= slice(
              12000, 12200
              ),
    time_slice_thre2_peak= slice(
              12300, 12790
              ),
    method_1 = 'maximum',
    method_2_base = 'mean',
    method_2_peak = 'mean'
    )

# Plot to decide on the thresholds.
BG_def.Thre_multi_plot(thre1_list=range(40,50,2),
                       thre2_list=(6,8,9,10,11,12),
                       figsize_scale= 2.0)
# From the displayed multi-panel plot, decide both thresholds.
plt.close()

# Plot the final thresholding. 
# This is an important quality control in a loop of multiple-files analysis.
BG_def.Thre_plot(thre1=40, thre2=9)





"""----------------Implementation of background removal---------------------------
The mask as a bool vector can be used to delete the background pixels.
However, this results to the calcium np.ndarray data with reduced rows and 
without proper label, we may lose the information about the X and Y
positions of the selected pixels among these reduced rows.

Thus, the idea is to store the information in a dataframe corresponding to
the row-reduced calcium np.ndarray. The rows of the dataframe correspond to the 
rows of the calcium np.narray, whereas the dataframe also has the X and Y
positions
"""
Cal_reduced, df_reduced= BG_def.Cut_background(thre1=40, thre2=9)



"""-----------------------Neuronal segmentation------------------------------------------------


2. PC_PROJ: Spatial SVD (Singular value decomposition) cluster pixels that share the same
            temporal response pattern.
      Also, dimension reduction using spacial PCA (principle component analysis), 
          since the spatial dimenstion (total pixel number) is much larger than time.
   
   Inputs (parameters):
       Calcium_imaging, np.ndarray, it can be original calcium imaging data, 
           or the BG_removal output with background removed.
       time_start, time_end, the starting and end time frame index to cut a piece of time series
           (e.g., the window for a stimulated response) for the downstream analysis.
       n_PC, it is too much to compute all components (total pixel number).
       dim_h, dim_w, the pixel size of calcium imaging, it is output of ImportCZI().read_Cal().
       
  Outputs (attributes):
      PCs, All computed PCs (Principle Components) as np.ndarray; 
          the 0th dimension that is used to select components;
          the 1st dimension that is used to select pixels.
          Notice the PCs are flatterned (one row connected by other rows, not row * columns), 
              it needs to be reshaped to width and height dimensions.
      PC_plot(n), a function to plot the nth PC. This plot can be used for PC choosing for projection.
      PC_proj(PC_indices), A projection of all chosen PCs, 
          PC_indices decide on the PC selection,
              in the form: "list(range(10,19))+list(range(0,2))"
          output an flatterned 1 dimension np.ndarray image.    
"""
# ----------Princle component analysis (PCA) of the calcium imaging data-------------------
"""
    Important that Calcium_imaging should have 0's axis defining time,
    because PCA().fit(input), input is of shape (n_samples, n_features).
    here pixels are features.
    """
#Run PCA on the background removed calcium imaging data
pca_data= imgseg.PC_PROJ(Calcium_imaging= Cal_reduced,
                  df_reduced= df_reduced,
                  time_start= 11127,
                  time_end= 12696,
                  n_PC= 20,
                  dim_h =shape[1], 
                  dim_w =shape[2])

#Run PCA on the raw data without background removal,
pca_data2= imgseg.PC_PROJ(Calcium_imaging= cal_data,
                  time_start= 11127,
                  time_end= 12696,
                  n_PC= 20,
                  dim_h =shape[1], 
                  dim_w =shape[2])

"""
If the data has background pixel removed, one cannot visualize the PCs,
since these PCs also don't have background pixles as well. We need to fill
those background pixels as 0 to restore the full PCs.

This priciple fit for both pca_data.Get_PCs() and the PCA plot functions.
""" 
#Extract the components, PCs do not have background pixels, fullPCs have all pixels
PCs,fullPCs= pca_data.Get_PCs()

# ------Visualize PCs------
# Visualize a specific PC.
pca_data.PCA_plot(3)
# Visualize systematically all PCs
pca_data.PCA_multi_plot(n_panel_row=4, n_panel_col =5, 
                        pc_index_list=range(0,19),
                        figsize_scale= 4.0)
        
"""-------Implementation: pixels to be clustered into corresponding neurons-------

3. Neuron_Cluster: DBSCAN cluster of pixels in spatial principle components.
    
      Inputs (parameters):
          PCs, the second output of PC_PROJ, containing all pixels, np.ndarray
              2D array of shape (height, width)
          pc_binary_threshold, threshold used to change PCs into binary images, usually set as 0 or very small number
          height and width, (int,int), the spatial size of the calcium imaging.
              can be obtained from ImportCZI(...).read_Cal()[1],ImportCZI(...).read_Cal()[2].
          img_process_fun: method to process neuronal image generated from DBSCAN clustering. This is done neuronwise.
  
          ----------------------------Establish function for processing of binary images---------------------------
          #Here is an template doing first a morphological close then fill hole.
          def img_process_fun(img: NDArray[np.floating]):-> np.ndarray:
              from scipy.ndimage import binary_closing, binary_fill_holes       
              return binary_fill_holes(binary_closing(img, structure=np.ones((3,3)) ) )
        
              
      Outputs (attributes) for visualization and parameter learning:
          PC_DBSCAN, a function that perform DBSCAN clustering to segment neurons.
              The function output a dataframe with a neuron column called "Cluster" defining 
              which neuron each pixel belongs to.
          
         PC_DBSCAN_plot, a multi-pannel-plot function that visualize the effect of eps and min_samples
             on neuronal cluster effect for a defined PC
         PC_index_plot, plot all PCs when all parameters are defined.
     
     Output (attributes) for clustering outcome and visualization.
          Z_Project, a function perfrom z projection of all clustered PCs;
              the z-projection is done from the last of the defined list of PCs;
              when going to the previous PC, pixels already clustered in the next PC will be removed from
                  this previous PC. This is to avoid that the same neurons segmented twice;
             
             The output is that output a dataframe, rows representing XY coordinates, 
                  with a neuron column called "Cluster" defining, which neuron each pixel belongs to. 
                  The cluster may not be [1,2,3,...], but [1,3,10...], because some cluster indexes are filtered out.
                  
                  Notise that the output only contains positive pixel rows, without BG_define background, 
                      or DBSCAN defined noise, or size filtered pixels.
          A function that can be used to plot Z_Project 

 Parameter
 pc_index: which PC from the PCs is chosen to be clustered
 
 Parameters to be optimized:
     eps, neighborhood radius, key tuning parameters for DBSCAN. 
     min_samples, minimum points to form a dense region, key tuning parameters for DBSCAN
     min_pixel_size, max_pixel_size, threshold of spatial size to remove particles too small or too
         large to be considered as a neuronal soma.   
     
 Artributes
 A dataframe with XY coordinates of pixels 
     with a Cluster column defining which neuron each pixel belongs to.   
"""  
dbscn_data= imgseg.Neuron_Cluster(
    fullPCs,
    pc_binary_threshold=0,
    core_or_not = False,#If only core, the neurons are smaller, losing boarder pixels
    height = shape[1], 
    width = shape[2]   )

         
         
""""-------Determination of parameters for efficient clustering and neuronal distinction-----"""

"""
Pick a specific PC to do cluster. Visualize the effect of eps and min_samples on clustering.
The goal is to distinguish different neurons while neuronal soma ROI is not too much shrinked
"""
# This plot helps to visualize the influence of eps, on clustering result, the plot have subpannels that represent different combinations of these two parameters. This visualization can be performed under different filter sizes, to evaluate how small/large a particle that needs to be removed
dbscn_data.PC_DBSCAN_plot(pc_index=0, eps_list=[2,3,4,5,6], 
                          min_samples_list= range(6,12,22),
                          min_pixel_size=100, max_pixel_size=10000)

# Determining eps and min_samples, then visulize the clustering on different PCs. 
# This plot gives a pannel of different PCs under a defined parameter setting. n_rows is the number of rows in this pannel, in case the pannels append to be to wide.

dbscn_data.PC_index_plot(pc_list=range(0,20), eps=5, min_samples=36,n_rows=4,
                         min_pixel_size=100, max_pixel_size=10000)    

#Visualize the z-projection of different clusters over all chosen PCs
"""
----------------------z_projection-----------------------------------
1. perform dbscan and size filter on the last (nth) binary pc;
2. remove all positive pixels in nth pc from (n-1)th binary pc;
3. perform dbscan and size filter on the (n-1)th binary pc;
4. keep on the iteration. 
"""  
dbscn_data.Z_Project_plot(
    pc_list=range(0,20), eps=5, min_samples= 22,
    min_pixel_size=80, max_pixel_size=500,
    min_roundness= 0.2, max_roundness= 1
    )


# Implementation: Clustering and z-projection
df_clust=dbscn_data.Z_Project(
    pc_list=range(0,20), eps=5, min_samples= 22,
    min_pixel_size=80, max_pixel_size=500,
    min_roundness= 0.1, max_roundness= 1)


"""---Implementation: separate subcellular region (shell) from the center (core) of neurons
----------------------------------
4. Neuron_Seg: Output a calcium imaging matrix data and a dataframe that inform which pixel belong to which neuron.
    
      Inputs (parameters):
          Cal_raw_data, np.ndarray, it must be original calcium imaging data, 
              not the BG_removal output with background removed!!!
              
          df_clust, pd.DataFrame, rows representing XY coordinates, 
              with a neuron column called "Cluster" defining, which neuron each pixel belongs to.
              Usually the output of Neuron_Cluster().Z_Project()      
              
      Outputs (attributes) for visualization and parameter learning:
          Output_all_Cal, a function that output calcium imaging data with all pixels included in df_clust
              The order of the rows (in the 1st dimension, while the 0th defines space) 
              is the same as df_clust
         
         Output_core_border, a function that output calcium imaging data of core/borders of each neurons,
             and also output a dataframe of XY coordinates of these core/borders pixels, with the same order
"""
#Output the segementation
Nseg = imgseg.Neuron_Seg(
    Cal_raw_data= cal_data,
    df_clust= df_clust,
    height = shape[1], 
    width = shape[2],
    pixel_border_width=3)

df_core, df_border, Cal_core,Cal_border = Nseg.Output_core_border()
df, Cal = Nseg.Output_all_Cal()

# Also extract the background pixel signals. This is useful for background correction in the downstream analysis (siganalysis).
Cal_background = Nseg.Output_background()



"""
The matching to immunofluorescence (IF) is performed through a registration process involving stretching and shifting, while torsion and rotation are not considered. The goal is to align the pixel locations in the calcium imaging image with those in the IF image. In other words, the objective is to maximize the number of co-localized pixels (i.e., pixels with the same XY coordinates) that are positive in both the calcium imaging and IF images. For example, if pixels  identified as cells (positive) in the calcium imaging image correspond to cell pixels in the IF image well, the matching is considered successful.

In this regard, IFmatch is composed of the following steps:
    1. Downsize the IF images if the resolution is different from that of calcium imaging.
    2. Binarize the IF images by optimizing the IF intensity threshold to obtain cell shapes comparable to those in the calcium imaging .
    3. Use manual shifting and stretching and plot to understand the stretching factor.
        This stretching is usually needed because of microscope errors, where the downsizing performed in step one is insufficient to achieve equal scaling.
    4. Use automatic shifting correction of IF image through a simple optimization function. 
        The objective function maximizes the number of shared positive pixels.
    5. Optionally check the matching between calcium imaging and IF.
    6. Output a dataframe of X and Y coordinates of calcium imaging and an additional column of IF intensities.
        
IF matching
Manual adjustment and plotting to understand streching factor
"""

IF_match_data= ifmatch.IFmatch(
    Calcium_imaging_df= df_clust,
    CGRP_raw= CGRP_raw,
    NF_raw= NF_raw,
    Calcium_height = shape[1], 
    Calcium_width = shape[2])

# to determine the binarizing threshold and which IF channel to use
IF_match_data.Matching_plot_IF_threshold(
    IF_threshold_list=range(1000,10000,500), 
    n_rows= 4, which_channel = 'CGRP'
    )
"""--------to determine the stretching correction------------"""
IF_match_data.Manual_regi_plot(
    IF_threshold=2000,a=1.05,b=-7,c=1.02,d=-8
    )


"""
---shift optimization-----

Dataframe-based registration is slow. It is only used for plotting.
It is recommented to use the matrix method instead
"""


# df method which is slow
IF_match_data= ifmatch.IFmatch(
    Calcium_imaging_df= df_clust,
    CGRP_raw= CGRP_raw,
    NF_raw= NF_raw,
    Calcium_height = shape[1], 
    Calcium_width = shape[2])

best_shift, best_score= IF_match_data.Registration_optimize(
    IF_threshold=2000,
    a = 1.05,                  # fixed scale in X
    c = 1.02,                  # fixed scale in Y
    step = 2.0,
    start_X = -10.0,
    end_X = 20.0,
    start_Y = -10.0,
    end_Y = 20.0,
    plot_result = True,
    which_channel = 'NF'
    )

"""--------matrix method which is fast------------
Note: the class IF_match_data_mtr is only used for optimization process,
plotting functions and registration implementation and all only in the
IFmatch class"""
IF_match_data_mtr= ifmatch.IFmatchMatrix(
    Calcium_imaging_df= df_clust,
    CGRP_raw= CGRP_raw,
    NF_raw= NF_raw,
    Calcium_height= shape[1],
    Calcium_width= shape[2])

best_shift, best_score= IF_match_data_mtr.Registration_optimize(
    IF_threshold=2000,
    a = 1.05,                  # fixed scale in X
    c = 1.02,                  # fixed scale in Y
    step = 2.0,
    start_X = -10.0,
    end_X = 20.0,
    start_Y = -10.0,
    end_Y = 20.0,
    plot_result = True,
    which_channel = 'NF'
    )

IF_match_data_mtr= ifmatch.IFmatchMatrix(
    Calcium_imaging_df= df_clust,
    CGRP_raw= CGRP_raw,
    NF_raw= NF_raw,
    Calcium_height= shape[1],
    Calcium_width= shape[2])

df_IF_matrix= IF_match_data_mtr.Merge_channels_in_df( 
    a=1.05,b=-10,c=1.02,d=-6
    )


# Implementation IF matching and output
IF_match_data= ifmatch.IFmatch(
    Calcium_imaging_df= df_clust,
    CGRP_raw= CGRP_raw,
    NF_raw= NF_raw,
    Calcium_height = shape[1], 
    Calcium_width = shape[2])
# Final check of the registration
IF_match_data.Manual_regi_plot(
    IF_threshold=2000, a=1.05, b=best_shift[0], 
    c=1.02, d=best_shift[1],
    )    
df_IF= IF_match_data.Merge_channels_in_df( 
    a=1.05, b=best_shift[0], 
    c=1.02, d=best_shift[1]
    )

#Check if the matrix method generate the same result as df method
df_IF.equals(df_IF_matrix)

"""
---Signal processing and analysis---------

The signal processing is composed of the following steps
SigProcess: class
1. Normalization of the time series. Plotting of the time series curve.

2. Low pass filter to reduce the influence from high frequency noise. Plotting the effect of low pass filter is useful for defining butter filter parameters, namely the cutoff and order.

3. Linear correction to reduce the effect of photobleaching. Plotting the correction can help in understanding if the correction is correctly done. This is espectially important when the calcium imaging lasts very long that the photobleaching effect is no longer linear.
Also, optionally substract the background change due to adding reagent or neutrophile response. 
Output from step 3 is directly used for peak height based quantification.
     
4. Deconvolution. Assuming that intracellular calcium veriations are caused mainly by a summation, i.e., convolution, of action potentials (APs). Deconvolution helps in reconstructing the AP train and then calculate the AP firing rate. (Yaksi, Nat.Methods, 2006)
The outcome of deconvolution will be plot to decide on the kernel representing the morphology of calcium spike. 

Sigquanti: class
    Frequency (i.e., spiking rate) based quantification
5. Calculate the calcium spike frequency with or without re-processing of deconvolution.

    Peak height based quantification
6. Peak height measurement. 

7. Neuronal activity plot and data output
"""


# Plotting the calcium curves, the first 10 pixels of the calcium imaging data
Sigdata= siganalysis.SigProcess(Cal,df)
Sigdata.Pre_curve_plot(temp_base_start=0, temp_base_end=900)#Pick the first 10 pixels to visualize

# Butter filter
# First check by setting certain parameters, what range of frequencies is filtered out (frequency response).
Sigdata.Fre_response_plot(fs=30, order = 4, cutoff=4)

# Check the effect of filter by setting a list of parameters. 
Sigdata.Butter_plot(fs=30, pixel_index=1,
    time_start= 0, time_end= 300, #choose the time frame window to plot the filter's effect
    cut_list= range(8,14,2), order_list= (8,10,12,14), #list of values about cut and order to be plot and tested          
    def_base_start=0, def_base_end=300,
    figsize= (10,5)    )

# Butter filter + Normalization implementation
# Once parameters are determined, perform the filter, output the filtered and unfiltered data. 
Cal_filt,Cal_unfilt= Sigdata.Butter_filter(fs=30, cutoff=12,  order=14,
                  def_base_start=0, def_base_end=12000)

# A final look to see if the filter make sense or not. Show the filtered curve, each curve a neuron.
Sigdata.Curve_plot(Cal_filt,
                time_start= 0, time_end= 900,
                figsize= (15,5),
                x_interval= 50)

# Show the original curve to compare.
Sigdata.Curve_plot(Cal_unfilt,
                time_start= 0, time_end= 900,
                figsize= (15,5),
                x_interval= 50)


"""-----Linear correction of photobleach-----------------------
Notice that if the calcium imaging is too long, the photobleaching effect is no longer linear. The solution is to fit on the window close to the time window that is analyzed as an example here, the stimulated response is analyzed at 12000's time frame, then the 6000-12000 window that is close to the 12000 is used to perform this correction"""
Sigdata= siganalysis.SigProcess(Cal,df)

# Pick one time series (in column) to visualize correction
Sigdata.Substract_linear_pick_column_plot(
    Cal_filt, 
    linear_fit_start= 6000, 
    linear_fit_end= 12000,
    column_index=5, figsize=(30,5))

# mplement correction
corrected, photobleach = Sigdata.Subtract_linear_baseline(
    Cal_filt, 
    linear_fit_start= 6000, 
    linear_fit_end= 12000
    )

"""-----Linear correction + background removal
In addition to removing consistent linear change of the signal, it is also possible to remove non-linear changes due to non-linear changes in background. This refers to the whole image brightness changes when adding reagent during calcium imaging or neutrophile response. 
"""
# Calculate the background change in extracellular region.
# It averages over all background pixels and normalizes the curve.
extr_background= Sigdata.Extract_extr_base(
    Cal_background, 
    def_base_start= 6000,
    def_base_end= 12000)
# Visualize the effect of background removal following linear correction.
Sigdata.Substract_background_plot(
                array= Cal_filt, 
                extr_background= extr_background,
                linear_fit_start= 6000, 
                linear_fit_end= 12000,
                column_index=2 # Which pixel to visualize
                )

#Remove background caused by adding reagent. Correct it according to the extracellular region.
corrected, background = Sigdata.Extr_background_substract(
                array= Cal_filt, 
                extr_background= extr_background,
                linear_fit_start= 6000, 
                linear_fit_end= 12000
                )



"""-----Deconvolution for frequency-based analysis-----"""
Sigdata= siganalysis.SigProcess(corrected, df)

# Plot kernals. 
# When the calcium spike's shape is known, can make decision on the kernal's shape that is the closest to the calcium spike shape.
Sigdata.Kernels_plot(length_list=range(2,20,2), tao_list=(0.01,0.1,0.5,1,2,5,10,15),figsize=(4,2))


Sigdata.Decon_para_plot(array= corrected[12000:12600,:], column_index=0,
                        length_list= range(10,200,20), tao_list=(2,5,10,15,30,60), 
                        figsize=(10,2))

# This function is very slow to run for the whole time series.pick a time window [12000:12600,:] instead.
Cal_decon= Sigdata.Decon(corrected[12000:12600,:], length= 10, tao= 30)

# This is a faster version that if long time period has to be deconvolved.
Cal_decon_fast= Sigdata.Decon_fast(corrected, length= 10, tao= 30)

# The following plot shows the effect of deconvolution. Also it can be seen that Sigdata.Decon and Sigdata.Decon_fast generate the same output.
Sigdata.Curve_plot(corrected,
                time_start= 12000, time_end=12600,
                figsize= (15,5),
                x_interval= 50)

Sigdata.Curve_plot(Cal_decon,
                time_start= 0, time_end=600,
                figsize= (15,5),
                x_interval= 50)

Sigdata.Curve_plot(Cal_decon_fast,
                time_start= 12000, time_end= 12600,
                figsize= (15,5),
                x_interval= 50)


"""
----Signal quantification-----------
Plot a curve to define spike sorting cutoff

Note: Since photobleaching correction is performed within the time window 11000:12600, frequency analysis should be restricted to this interval or at least remain close to it. Analyzing outside this range may introduce artifacts due to baseline drift in long calcium imaging time series.
"""
Sig_quant= siganalysis.Sigquanti(Cal_decon_fast, df, shape[1], shape[2])
Sig_quant.Get_spike_plot(pixel_index= 5005,
                         spike_h_cutoff=0.08,
                         time_start= 10000, time_end= 12700, #which time window to show
                         figsize= (30,5) )

Sig_quant= siganalysis.Sigquanti(corrected, df, shape[1], shape[2])
Sig_quant.Get_spike_plot(pixel_index= 5005,
                         spike_h_cutoff=0.08,
                         time_start= 10000, time_end= 12700, #which time window to show
                         figsize= (30,5) )

"""
----Plot spiking rate and peak height heatmap-----
Also very useful in defining spike_h_cutoff.
"""
Sig_quant= siganalysis.Sigquanti(Cal_decon_fast, df_IF, shape[1], shape[2])
Sig_quant.Freq_heatmap(spike_h_cutoff= 0.08, #peak height cutoff used to define calcium spikes
                          seg_start= 10000, seg_end= 11000, #pick a time window to calculate spike frequency
                          fs= 28.57 #sample frequency of calcium imaging, useful for spike frequency to be in Hz
                          )
# Plot to check which IF marker positive neurons have larger spiking frequency. Make sure that the input df_clust have columns of IF staining value. These values are added in the IFmatch or IFmatchMatrix class"""
Sig_quant= siganalysis.Sigquanti(Cal_decon_fast, df_IF, shape[1], shape[2])
Sig_quant.IF_freq_heatmap(spike_h_cutoff= 0.08, #peak height cutoff used to define calcium spikes
                          seg_start= 10000, seg_end= 11000, #pick a time window to calculate spike frequency
                          fs= 28.57 #sample frequency of calcium imaging, useful for spike frequency to be in Hz
                          )

Sig_quant= siganalysis.Sigquanti(corrected, df_IF, shape[1], shape[2])
Sig_quant.Peak_height_heatmap(base_start= 11800, base_end=12200,
                  sti_start=12200, sti_end=12300)

Sig_quant= siganalysis.Sigquanti(corrected, df_IF, shape[1], shape[2])
Sig_quant.Peak_height_heatmap(base_start= 11800, base_end=12200,
                  sti_start=12200, sti_end=12400)

Sig_quant= siganalysis.Sigquanti(corrected, df_IF, shape[1], shape[2])
Sig_quant.Peak_height_heatmap(base_start= 11800, base_end=12200,
                  sti_start=12200, sti_end=12500)

Sig_quant= siganalysis.Sigquanti(corrected, df_IF, shape[1], shape[2])
Sig_quant.Peak_height_heatmap(base_start= 11800, base_end=12200,
                  sti_start=12200, sti_end=12600)

Sig_quant= siganalysis.Sigquanti(corrected, df_IF, shape[1], shape[2])
Sig_quant.Peak_height_heatmap(base_start= 11800, base_end=12200,
                  sti_start=12200, sti_end=12700)

# Output spiking rate and peak height data
output_df= df_IF.copy()

Sig_quant= siganalysis.Sigquanti(Cal_decon_fast, df_IF, shape[1], shape[2])
output_df=Sig_quant.Freq_data_output( output_df,
                  colname='Freq_10k-11k',#how do you call this data? e.g., spikeing rate (Hz))
                  spike_h_cutoff=0.08, #peak height cutoff used to define calcium spikes
                  seg_start=10000, seg_end=11000, #pick a time window to calculate spike frequency
                  fs= 28.57)

output_df=Sig_quant.Peak_height_output( output_df,
                  colname='PH_10k-11k',#how do you call this data? e.g., spikeing rate (Hz))
                  base_start= 11800, base_end=12200,
                  sti_start=12200, sti_end=12700)


# Aggregate DataFrame from pixel level to neuron level
df_aggre= output_df.groupby("Cluster").agg(
    freq_mean=("Freq_10k-11k", "mean"),
    freq_median=("Freq_10k-11k", "median"),
    peak_mean=("PH_10k-11k", "mean"),
    peak_max=("PH_10k-11k", "max"),
    n_pixels=("Cluster", "size"))

output_df.to_csv("Pixelwise_output.csv", index=False)
df_aggre.to_csv("Neuronwise_output.csv", index=False)
