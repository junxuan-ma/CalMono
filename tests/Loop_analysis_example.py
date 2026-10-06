
# Revised on Sun Oct 04 11:36:05 2026

# @author: Junxuan Ma

# Implementation of the analysis in loop automatically for all the CZI files in a folder. 


from calmono import imgimport, imgseg, ifmatch, siganalysis

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes   
from skimage.measure import find_contours

import pandas as pd
from roifile import ImagejRoi, roiwrite

# Set your folder path here
folder_path_if = r"I:\ari\Projects\Neu-DISC\06_Data_RO\03_Data_DCB\Junxuan\Oscar\2nd visit\Python auto\Python combined"

folder_path = r"I:\ari\Projects\Neu-DISC\06_Data_RO\03_Data_DCB\Junxuan\Oscar\2nd visit\Python auto\Python combined\Calcium"



Thresholds= pd.read_csv(Path(folder_path_if)/'Diagnose.csv')

#Preprocessing clusters before doing Z-projection.
def img_process_fun(img: np.ndarray) -> np.ndarray:
    return binary_fill_holes(binary_closing(img, structure=np.ones((3,3)) ) )


# Get all calcium imaging CZI file names in the folder
calfiles= imgimport.FetchFiles(
    file_path= folder_path,
    file_tag="Donor"
    ).Get_filenames()


# Loop through each file
for idx, czi_name in enumerate(calfiles):
    #Parameters
    ifname = czi_name.replace(".czi", "_IF")
    name= ifname.replace("_IF", "")


    #Import the CZI file
    importer = imgimport.ImportCZI(
        folder_path_cal=folder_path,
        folder_path_if=folder_path_if,
        Cal_name=czi_name,
        cal_tag=".czi",
        if_tag="_IF.czi"
        )
    
    # If there is no corresponding IF file, skip this calcium imaging file.
    # This is to not breaking the loop.   
    if not importer.IF_exists():
        continue

    #Read the IF and calcium imaging data.
    CGRP_raw, NF_raw = importer.Read_if(CGRP=1,NF=0)
    cal_data, shape = importer.Read_cal() #shape: (time,height,width)

    # Threshold for the segmentation is stored in Diagnose.csv, based on manual decision.
    Threshold=Thresholds[Thresholds['Name']==ifname]

    #Substract background
    BG_def= imgseg.BG_define(Calcium_imaging= cal_data,
          dim_h= shape[1],
          dim_w= shape[2],
          time_slice_thre1= slice(
              Threshold["sti_end3"].values[0]-200,
              Threshold["sti_end3"].values[0]-100
              ),
          time_slice_thre2_base= slice(
              Threshold["sti_end3"].values[0]-100,
              Threshold["sti_end3"].values[0]),
          time_slice_thre2_peak= slice(Threshold["sti_end3"].values[0],
                                       12790),
          method_1 = 'maximum',
          method_2_base = 'mean',
          method_2_peak = 'mean',)
    
    Cal_reduced, df_reduced= BG_def.Cut_background(thre1=2, thre2=4)

    #Clustering to different neurons
    pca_data= imgseg.PC_PROJ(
                                Calcium_imaging= Cal_reduced,
                                df_reduced= df_reduced,
                                time_start= Threshold['sti_end3'].values[0]-100,
                                time_end= 12790,
                                n_PC= 20,
                                dim_h =shape[1], 
                                dim_w =shape[2]
                                )
    
    PCs,fullPCs= pca_data.Get_PCs()
    dbscn_data= imgseg.Neuron_Cluster(
                                fullPCs,
                                pc_binary_threshold=0,
                                core_or_not = True,#If only core, the neurons are smaller, losing boarder pixels
                                height = shape[1], 
                                width = shape[2],
                                img_process_fun = img_process_fun
                                )



    #save image at higher dpi than the dpi used to build the figure, so that the saved image is higher resolution than the figure on screen, although the ROI won't fit in the image well.
    dbscn_data.Z_Project_plot(pc_list=range(0,20), eps=5, min_samples= 22,
                             min_pixel_size=80, max_pixel_size=500,
                             min_roundness= 0.2, max_roundness= 1,
                             dpi=20 )
    plt.savefig(name + "seg.png", dpi=200, bbox_inches='tight')
    plt.close()



    
    df_clust=dbscn_data.Z_Project(pc_list=range(0,20),
                                eps=5,
                                min_samples= 22,
                                min_pixel_size=80, max_pixel_size=500,
                                min_roundness = 0.2, max_roundness = 1
                                )
    
    #If there is no neuron segmented at all, to not break the loop:
    if df_clust.empty:
        continue



    #Write the roi that can be opened in ImageJ Fiji
    rois = []
    for roi_id, group in df_clust.groupby('Cluster'):
        xmin, xmax = group.X.min(), group.X.max()
        ymin, ymax = group.Y.min(), group.Y.max()

        # rasterize this cluster's pixels into a small binary mask
        mask = np.zeros((ymax - ymin + 3, xmax - xmin + 3), dtype=np.uint8)  # +3 padding so edge pixels aren't clipped
        mask[group.Y - ymin + 1, group.X - xmin + 1] = 1

        # trace the outer contour(s) of the mask
        contours = find_contours(mask, level=0.5)
        if not contours:
            continue

        # take the largest contour (main outline; skip tiny artifacts/holes)
        contour = max(contours, key=len)

        # find_contours returns (row, col) = (Y, X); ImagejRoi wants (X, Y),
        # and shift back by the padding + crop offset
        coords = np.column_stack([
            contour[:, 1] - 1 + xmin,   # X
            contour[:, 0] - 1 + ymin,   # Y
        ]).astype('float32')

        roi = ImagejRoi.frompoints(coords)
        roi.name = f'roi_{roi_id}'
        rois.append(roi)

    roiwrite(name + '_RoiSet.zip', rois, mode='w')
    




    # Preparing calcium imaging data
    N_seg= imgseg.Neuron_Seg(
        Cal_raw_data= cal_data,
        df_clust= df_clust,
        height = shape[1], 
        width = shape[2],
        pixel_border_width=3
        )

    Cal_background = N_seg.Output_background() 
    df, Cal= N_seg.Output_all_Cal()

    #IF matching finding the shift using the full neurons in calcium imaging, notice the input is df_clust representing the full neurons, not df_border representing only the border of neurons
    IF_match_data_mtr= ifmatch.IFmatchMatrix(
                                Calcium_imaging_df= df,
                                CGRP_raw= CGRP_raw,
                                NF_raw= NF_raw,
                                Calcium_height= shape[1],
                                Calcium_width= shape[2])
    best_shift, best_score= IF_match_data_mtr.Registration_optimize(
                                IF_threshold=2000,
                                a = 1.05, # fixed scale in X
                                c = 1.02, # fixed scale in Y
                                step = 2.0,
                                start_X = -10.0,
                                end_X = 20.0,
                                start_Y = -10.0,
                                end_Y = 20.0,
                                plot_result = False,
                                which_channel = 'NF')

    """
    # Now map neuron border calcium imaging data to IF data. Notice the output is df_border representing only the border of neurons, not df_clust representing the full neurons.
    IF_match_data_mtr= ifmatch.IFmatchMatrix(
                                    Calcium_imaging_df= df_border,
                                    CGRP_raw= CGRP_raw,
                                    NF_raw= NF_raw,
                                    Calcium_height= shape[1],
                                    Calcium_width= shape[2])
                                    """
    df_IF = IF_match_data_mtr.Merge_channels_in_df( 
                                a=1.05,
                                b=best_shift[0],
                                c=1.02,
                                d=best_shift[1]
                                )


    
    #Signal processing, Notice the data analyzed is df_IF_border corresponding to Cal_border.
    Cal= Cal.astype(np.float32, copy=False)
    #Truncated the time series to leave only the capsaicin response part.
    Cal= Cal[Threshold["sti_end2"].values[0]-500 : Threshold["sti_end3"].values[0] , :]
    Cal_background= Cal_background[Threshold["sti_end2"].values[0]-500 : Threshold["sti_end3"].values[0] , :]

    time= Threshold["sti_end3"].values[0] - Threshold["sti_end2"].values[0] + 500


    Sigdata= siganalysis.SigProcess(Cal,df_IF)
    Cal_filt,_= Sigdata.Butter_filter(fs=30, cutoff=10,  order=2,
                  def_base_start=0, def_base_end=400)

    #The baseline level calculated by the extracellular region.
    extr_background= Sigdata.Extract_extr_base(Cal_background, def_base_start= 0,def_base_end= 400)
    
    #Remove background caused by adding reagent. Correct it according to the extracellular region.
    corrected, _ = Sigdata.Extr_background_substract(
                array= Cal_filt, 
                extr_background= extr_background,
                linear_fit_start= 0, 
                linear_fit_end= 400
                )

    Sigdata.Substract_background_plot(
                array= Cal_filt, 
                extr_background= extr_background,
                linear_fit_start= 0, 
                linear_fit_end= 400,
                column_index=2
                )
    plt.savefig(name + "_corr.png", dpi=200, bbox_inches='tight')
    plt.close()


    Sigdata.Curve_plot(corrected,
                time_start= 0, time_end= 3000,
                figsize= (15,5),
                x_interval= 50)
    plt.savefig(name + "_curve.png", dpi=200, bbox_inches='tight')
    plt.close()


    

    #Only for freqency based method
    Cal_decon_fast= Sigdata.Decon_fast(
        corrected, 
        length= 5, 
        tao= 21
        )
    
    #Signal quantification frequency based method
    output_df= df_IF.copy()
    Sig_quant= siganalysis.Sigquanti(Cal_decon_fast, df_IF, shape[1], shape[2])
    output_df=Sig_quant.Freq_data_output( output_df,
                  colname='Freq_Cap_0.16',
                  spike_h_cutoff=0.16, 
                  seg_start=500, 
                  seg_end=time, 
                  fs= 28.57)
    output_df=Sig_quant.Freq_data_output( output_df,
                      colname='Freq_Cap_0.18',
                      spike_h_cutoff=0.18, 
                      seg_start=500, 
                      seg_end=time, 
                      fs= 28.57)
    output_df=Sig_quant.Freq_data_output( output_df,
                          colname='Freq_Cap_0.20',
                          spike_h_cutoff=0.20, 
                          seg_start=500, 
                          seg_end=time, 
                          fs= 28.57)
    output_df=Sig_quant.Freq_data_output( output_df,
                              colname='Freq_Cap_0.22',
                              spike_h_cutoff=0.22, 
                              seg_start=500, 
                              seg_end=time, 
                              fs= 28.57)
    #Signal quantification peak height based method
    Sig_quant= siganalysis.Sigquanti(corrected, df_clust, shape[1], shape[2])
    output_df=Sig_quant.Peak_height_output( output_df,
                    colname='PH_Cap',
                    base_start= 0, 
                    base_end= 500,
                    sti_start=500, 
                    sti_end= time)
    df_aggre= output_df.groupby("Cluster").agg(
                    freq_mean_0_16 = ("Freq_Cap_0.16", "mean"),
                    freq_mean_0_18 = ("Freq_Cap_0.18", "mean"),
                    freq_mean_0_20 = ("Freq_Cap_0.20", "mean"),
                    freq_mean_0_22 = ("Freq_Cap_0.22", "mean"),
                    peak_mean=("PH_Cap", "mean"),
                    peak_median=("PH_Cap", "median"),
                    peak_max=("PH_Cap", "max"),
                    n_pixels=("Cluster", "size"),
                    NF = ("NF" , "mean"),
                    CGRP = ("CGRP" , "mean")
                    )
    output_df.to_csv("Pixelwise_"+name+"output.csv", index=False)
    df_aggre.to_csv("Neuronwise_"+name+"output.csv", index=False)
