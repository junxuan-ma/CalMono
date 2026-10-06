# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:57:14 2026

@author: Junxuan Ma
"""

#Processing and extracting neuronal signals to evaluate neuronal activities
"""
SigProcess: class
The signal processing is composed of the following steps
1. Normalization of the time series. Plotting of the time series curve is useful for defining the baseline time window.
2. Low pass filter to reduce the influence from high frequency noise. Plotting the effect of low pass filter is useful for defining butter filter parameters, namely the cutoff and order.  
3. Linear correction to reduce the effect of photobleaching. Plotting the correction can help in understanding if the correction is correctly done. This is espectially important when the calcium imaging lasts very long that the photobleaching effect is no longer linear.
Output from step 3 is directlyused for peak height based quantification.

Also, optionally correct the background change due to adding stimulus or neutrophil response. 

  
4. Deconvolution. Assuming that intracellular calcium veriations are caused mainly by
    a summation, i.e., convolution, of action potentials (APs). Deconvolution helps in 
    reconstructing the AP train and then calculate the AP firing rate. 
    (Yaksi, Nat.Methods, 2006)
Step 4 is optionally used for frequency based quantification
"""
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
from scipy.signal import find_peaks,butter,sosfreqz,sosfiltfilt,deconvolve

"""------------------------------Signal processing-----------------------------------------"""
class SigProcess:
    def __init__(self, 
                 Calcium_imaging: np.ndarray,
                 df_cluster: pd.DataFrame):
        """
        Parameters (inputs)
            Calcium_imaging: 2D array with time series in rows (time along 0th dimension)
            df_cluster gives information about which pixel belongs to which neuron.
        
            Notice that the count of rows in df_cluster 
                should be corresponding to the 1st dimension of Calcium_imaging data,
                namely the pixel number should be the same. 
                This is risky when messing up core, border and whole neuron.
        
        Attributes (outputs)
            Pre_curve_plot, a curve plotting function to define baseline window 
                for normalization.
            
            Fre_response_plot and Butter_plot, Butter filter related plotting functions.
            Butter_filter, a function to perform Butter filtering on time series.
            
            Subtract_linear_baseline, a function to perform photobleaching correction.
            Substract_linear_pick_column_plot, a curve plotting function to confirm
                effect of Subtract_linear_baseline

            Extract_extr_base, a function that first do photobleach correction and then extract extracellular background from a 2D array of time series.
            Extr_background_substract, a function to perform photobleach correction first, then remove extracellular background (background also photobleach-corrected).
                
            Kernels_plot and Decon_para_plot, deconvolution related plotting functions
            Decon, scipy-based deconvolution; Decon_fast, fft based deconvolution (faster).
        """
        if Calcium_imaging.ndim != 2:
            raise ValueError("Calcium_imaging must be a 2D NumPy array (time x features).")
        if Calcium_imaging.shape[1] != df_cluster.shape[0]:
                raise ValueError(
                    f"Row mismatch: Calcium_imaging has {Calcium_imaging.shape[1]} columns, "
                    f"but df_cluster has {df_cluster.shape[0]} rows."
                    f"The calcium imaging is not aligned with the DataFrame of neuronal segmentation"
                )
        if "Cluster" not in df_cluster.columns:
                raise ValueError("df_cluster must contain a column named 'Cluster' that contains the information of neuronal segmentation")
        
        self.cal= Calcium_imaging
        self.df= df_cluster
 
    
    """---------------------------Defining baseline window---------------------------------"""
    def _average_by_cluster(self, calcium_imaging: np.ndarray, reference_df: pd.DataFrame):
        """Average the columns of calcium_imaging based on cluster labels in reference_df.
        Parameters:
            calcium_imaging: np.ndarray of shape (time_frames, pixels)
            reference_df: pd.DataFrame with a column 'Cluster' of length equal to number of pixels per neuron
        Returns:
            np.ndarray of shape (time_frames, num_clusters) with averaged signals
            cluster_labels: list of cluster labels in the same order as the columns
            """
        clusters = reference_df["Cluster"].unique()
        averaged_array = np.zeros((calcium_imaging.shape[0], len(clusters)))
        reference_df = reference_df.reset_index(drop=True)
        for i, cluster in enumerate(clusters):
            indices = reference_df.index[reference_df["Cluster"] == cluster].tolist()
            # Take mean across selected columns (axis=1 = time dimension)
            averaged_array[:, i] = calcium_imaging[:, indices].mean(axis=1)
        
        return averaged_array, clusters 
         
    def _normalization(self, array, def_base_start, def_base_end):
        baseline = array[def_base_start:def_base_end, :].mean(axis=0, keepdims=True)
        return (array - baseline) / baseline   
    
    def Pre_curve_plot(self, 
                    temp_base_start: int, temp_base_end: int,
                    figsize: list= (30,5),
                    x_interval: int= 200):
        """Due to the fact that the stimulation cannot be precisely controlled 
            at the millisecond level, the exact starting point of the stimulation 
            and the baseline time window preceding the stimulation are not known.
        This algorithm average pixels within the same neuron and output a neuron signal.
        Notice that the influence of noise is influenced by number of pixels.
            This averaging among pixels is not suitable for signal extraction 
            unless all neurons are of the same size.
        
        temp_base_start and temp_base_end represent a temporary baseline time window used 
            for normalization. They are typically set to a short interval from 0 to 10 
            in order to avoid including time frames that may occur after stimulation.
        After plotting the data, these parameters can be adjusted based on the abrupt 
            change in fluorescence caused by fluid injection during neuronal stimulation.
            figsize and x_interval can be played to show clearer when the stimulation was given.
        """
        per_neuron, cluster_labels= self._average_by_cluster(self.cal, self.df)
        array = self._normalization(per_neuron, temp_base_start, temp_base_end)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(array)
        ax.xaxis.set_major_locator(MultipleLocator(x_interval))
        plt.xticks(rotation='vertical')
        ax.set_xlabel("Time frame")
        ax.set_ylabel("ΔF/F0")
        ax.set_title("Normalized Fluorescence Curves")
        plt.tight_layout()

    
    """This function is to plot array of curves that pixels belonging to the same
            neuron is averaged first and each curve represent one neuron.
        The input Cal_filt can be processed (normalized, filtered or deconvolved) 
            or non-processed curves
        The time_start and time_end is the time window you want to display in the plot"""
    def Curve_plot(self, Cal_filt,
                    time_start: int, time_end: int, #which time window to show
                    figsize: list= (30,5),
                    x_interval: int= 200):
        
        per_neuron, cluster_labels= self._average_by_cluster(Cal_filt[time_start:time_end,:], self.df)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(per_neuron)#transpose for array is [y,x], but plot is (x,y)
        ax.xaxis.set_major_locator(MultipleLocator(x_interval))
        plt.xticks(rotation='vertical')
        ax.set_xlabel("Time frame")
        ax.set_ylabel("ΔF/F0")
        ax.set_title("Fluorescence Curves")
        plt.tight_layout()
       
            
    
    """------------------------------Butter filter--------------------------------------"""
    def _butter_lowpass(self, cutoff, fs, order=4):
        """Return SOS coefficients for a lowpass Butterworth filter."""
        sos = butter(order, cutoff, btype='low', fs=fs, output='sos')
        return sos

    def _butter_lowpass_filter(self, data, fs, cutoff, order=4, axis=0):
        """
        Apply a zero-phase Butterworth low-pass filter to 1D or 2D data.

        Parameters:
            data (np.ndarray): Input signal (1D or 2D).
            cutoff (float): Cutoff frequency (Hz).
            fs (float): Sampling frequency (Hz).
            order (int): Filter order.
            axis (int): Axis along which to apply the filter (time axis).

        Returns:
            np.ndarray: Filtered signal (same shape as input).
        """
        sos = self._butter_lowpass(cutoff, fs, order,)
        return sosfiltfilt(sos, data, axis=axis)
    
    def Fre_response_plot(self, fs:int=30, order:int = 4, cutoff:int=4):
        """fs is the sampling frequency of the signal.

        order determines the shape or smoothness of the filter cutoff. 
            A higher order produces a steeper transition around the cutoff frequency, 
            making the filter behave more like an ideal (sharp) cutoff.

        cutoff defines the cutoff frequency of the filter. 
            A larger cutoff value allows more high-frequency components of the signal 
            to pass through the filter.
        
        The butter filter works with the normalized frequency cutoff which is the real 
            frequency divided by fs*0.5. 
        The cutoff cannot be lower than fs*0.5 which is called the Nyquist frequency.
        
        Form this plot, make sure that the cutoff frequency makes sense."""
        #cutoff = 8 # desired cutoff frequency of the filter, Hz
        #
        sos = self._butter_lowpass(cutoff, fs, order)# Get the filter coefficients so we can check its frequency response.
        w, h = sosfreqz(sos, fs=fs)# from coefficients to frequency response.
        """The function freqz use parameters generated from the filter design to 
        generate a frequency response.
        w is the frequencies, h is the frequency response as complex numbers. 
        That's why to plot it, np.sqrt is needed.
        When the designed filter's output is sos use sosfreqz which compute the frequency response of a digital filter in SOS format.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfreqz.html#scipy.signal.sosfreqz
        """
        plt.figure(figsize=(8,4))
        plt.plot(w, np.abs(h), 'b')
        plt.axvline(cutoff, color='k', linestyle='--')
        plt.title("Lowpass Filter Frequency Response")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")
        plt.xlim(0, 0.5*fs)
        plt.grid(True)
        plt.tight_layout()
        
    
    def Butter_plot(self, fs, pixel_index: int,#choose a single time s
        time_start: int, time_end: int,#choose the time frame window to plot the filter's effect
        cut_list: list, order_list: list,#list of values about cut and order to be plot and tested          
        def_base_start: int, def_base_end: int,
        figsize: list= (10,5)    
        ):
        time= np.arange(time_start, time_end)
        data= self._normalization(self.cal[time_start:time_end, :],
                           def_base_start,def_base_end)
        fig,axs=plt.subplots(int(len(cut_list)),int(len(order_list)),
                             figsize=(len(order_list)*figsize[0],
                                      len(cut_list)*figsize[1]))
        axs = np.atleast_2d(axs)
        for i, cut in enumerate(cut_list):
            for j, order in enumerate(order_list):
                ax= axs[i,j]
                y = self._butter_lowpass_filter(data[:,pixel_index],  fs, cut, order)
                ax.plot(time, data[:,pixel_index], 'b-', label='data')
                ax.plot(time, y, 'r-', linewidth=2, label='filtered data')
                ax.set_title(f'order={order}, cutoff={cut}')
                ax.set_xlabel("Time frame")
                ax.set_ylabel("Signal")
                ax.legend()
        plt.tight_layout()
        
    
    def Butter_filter(self, fs: int, cutoff: int,  order: int,
                      def_base_start: int, def_base_end: int):
        array= self._normalization(self.cal, def_base_start,def_base_end)
        filt_array = self._butter_lowpass_filter(array, fs,cutoff,order)
        
        return filt_array, array
    
    
        
    """-------------------------------Photobleaching correction------------------------------""" 
    def Subtract_linear_baseline(self, 
                                 array: np.ndarray,
                                 linear_fit_start: int, 
                                 linear_fit_end: int
                                 ):
        """
        Subtract linear baseline from each column (neuron) in a 2D array (time x neurons)
        fully vectorized: each neuron gets its own linear fit.

        Parameters:
            array : np.ndarray
                1D array of shape (time,) or 2D array of shape (time, neurons)
            linear_fit_start : int
                Start index of baseline window
            linear_fit_end : int
                End index of baseline window

        Returns:
            corrected : np.ndarray
                Baseline-subtracted array (same shape)
            background : np.ndarray
                Linear background array (same shape)
                """
        was_1d = (array.ndim == 1)
        if was_1d:
            array = array[:, None]
        time = np.arange(array.shape[0])[:, None]  # shape (time,1)
        fit_time = np.arange(linear_fit_start, linear_fit_end)  # baseline indices
        # Extract baseline segment
        base_segment = array[linear_fit_start:linear_fit_end, :]  # shape (baseline_len, neurons)
        # Compute slope and intercept per neuron
        x_dev = fit_time - fit_time.mean()  # shape (baseline_len,)
        # The following is the least square equation.
        slope = (x_dev[:, None] * (base_segment - base_segment.mean(axis=0))).sum(axis=0) / (x_dev**2).sum()
        intercept = base_segment.mean(axis=0) - slope * fit_time.mean()
        # Compute background for all time points
        background = slope[None, :] * time + intercept[None, :]
        corrected = array - background
        if was_1d:
            corrected = corrected[:, 0]
            background = background[:, 0]

        return corrected, background
    
    def Substract_linear_pick_column_plot(self, array, linear_fit_start, linear_fit_end,
                                       column_index, figsize=(30,5)):
        
        corrected, background= self.Subtract_linear_baseline(array, linear_fit_start, linear_fit_end)
        time = np.arange(array.shape[0])
        plt.figure(figsize=figsize)
        plt.plot(time, array[:, column_index], label="Original Signal", color="blue", alpha=0.6)
        plt.plot(time, background[:, column_index], label="Linear Background", color="orange", linewidth=2)
        plt.plot(time, corrected[:, column_index], label="Corrected Signal", color="red", linewidth=1)
        plt.axhline(0, color='green', linestyle='--', alpha=0.5)

        plt.title(f"Column {column_index} Linear Baseline Correction")
        plt.xlabel("Time frame")
        plt.ylabel("Signal (ΔF/F0)")
        plt.legend()
        plt.tight_layout()



    """-------------------Extracellular background correction--------------------------
    Remove the extracellular/neuropil background contamination from each neuron's trace.
    Notice that even the extracellular/neuropil background need a linear correction for it has photobleach as well
    """
    def Extract_extr_base(self, Cal_background:np.ndarray,
                                 def_base_start: int, 
                                 def_base_end: int):
         #averaging over pixels
         base= Cal_background.mean(axis=1)
         #normalization
         baseline = base[def_base_start:def_base_end].mean()
         return (base - baseline) / baseline   


    def Extr_background_substract(self, 
                                  array: np.ndarray,
                                  extr_background:np.ndarray,#e.g., (4200,)
                                  linear_fit_start: int, 
                                  linear_fit_end: int
                                  ):
         corrected_array,_ =self.Subtract_linear_baseline(array, linear_fit_start, linear_fit_end)
         corrected_base,_= self.Subtract_linear_baseline(extr_background, linear_fit_start, linear_fit_end)

         # (4200,) -> (4200, 1) for broadcasting
         # (4200,1191)-(4200, 1) is correct, while (4200,1191)-(4200,) will be infact (1191)-(4200)
         corrected_base = corrected_base.reshape(-1, 1)  
         return corrected_array-corrected_base, corrected_base

    def Substract_background_plot(self, array, 
                             extr_background,
                             linear_fit_start, 
                             linear_fit_end,
                             column_index, 
                             figsize=(30,5)):
            
            corrected,background = self.Extr_background_substract(
                array, extr_background,
                linear_fit_start, linear_fit_end
                )
            
            time = np.arange(array.shape[0])
            plt.figure(figsize=figsize)
            plt.plot(time, array[:, column_index], label="Original Signal", color="blue", alpha=0.6)
            plt.plot(time, background, label="Extracellular Background", color="orange", linewidth=2)
            plt.plot(time, corrected[:, column_index], label="Corrected Signal", color="red", linewidth=1)
            plt.axhline(0, color='green', linestyle='--', alpha=0.5)
    
            plt.title(f"Column {column_index} Linear Baseline Correction")
            plt.xlabel("Time frame")
            plt.ylabel("Signal (ΔF/F0)")
            plt.legend()
            plt.tight_layout()
         



         
   
    """------------Deconvolution---------------------"""
    def _get_exp_kernel(self, length:float, tao:float):
        time = np.arange(length)
        kernel = np.exp(-time / tao)
        return time, kernel

    def Kernels_plot(self, length_list:list, tao_list:list,figsize:list=(20,5)):
        _, axs = plt.subplots(len(length_list), len(tao_list), 
                        figsize=(len(tao_list)*figsize[0], len(length_list)*figsize[1] ))
        axs = np.atleast_2d(axs)
        for i, kernel_length in enumerate(length_list):
            for j, kernel_tao in enumerate(tao_list):
                time, kernel = self._get_exp_kernel(kernel_length, kernel_tao)
                axs[i, j].plot(time, kernel)
                axs[i, j].set_title(f'tao:{kernel_tao}_len:{kernel_length}')
        plt.tight_layout()
        
        

    def Decon_para_plot(self, array:np.ndarray, column_index:int,
                            length_list:list, tao_list:list, 
                            figsize:list=(10,5)):
        
        curve = array[:,column_index]
        time= np.arange(len(curve))
        
        fig, axs = plt.subplots( len(length_list), len(tao_list), 
            figsize = (len(tao_list)*figsize[0], len(length_list)*figsize[1])  )        
        axs = np.atleast_2d(axs)
        
        for i, kernel_length in enumerate(length_list):
            for j, kernel_tao in enumerate(tao_list):
                _, kernel = self._get_exp_kernel(kernel_length, kernel_tao)
                deconv, remainder = deconvolve(curve, kernel)
                axs[i,j].plot(time[:len(deconv)], curve[:len(deconv)], 'b', label='data')
                axs[i,j].plot(time[:len(deconv)], deconv, 'r', label='deconv')
                axs[i,j].plot(time[:len(deconv)], remainder[:len(deconv)], 'g', label='remainder')
                axs[i,j].set_title(f'tao:{kernel_tao}_len:{kernel_length}')
        plt.tight_layout()
        
        
    def Decon(self, array:np.ndarray, length:int, tao:float):
        _, kernel = self._get_exp_kernel(length, tao)
        # Define a helper to deconvolve a single 1D array
        def _decon1d(col_signal):
            deconv, _ = deconvolve(col_signal, kernel)
            return deconv

        # Apply along axis=0 (columns)
        deconv_array = np.apply_along_axis(_decon1d, axis=0, arr=array)
        return deconv_array
 
    """Applying scipy.signal.deconvolve columnwise is slow when the data is large
    use the following function to speed up.
    
    Still Decon_fast consume a long periord of time. It is important to segment the
        time window of interest to perform deconvolution, not the whole time frames."""   
    def Decon_fast(self, array: np.ndarray, length:int, tao:float):
        _, kernel = self._get_exp_kernel(length, tao)
        
        n_time = array.shape[0]
    
        # Zero-pad kernel to match signal length
        kernel_pad = np.zeros(n_time)
        kernel_pad[:len(kernel)] = kernel
    
        # FFT along time axis
        fft_y = np.fft.fft(array, axis=0)
        fft_k = np.fft.fft(kernel_pad)[:, None]  # shape (time, 1)
    
        # Avoid division by zero
        eps = 1e-12
        fft_k = np.where(np.abs(fft_k) < eps, eps, fft_k)
    
        # Deconvolve in frequency domain
        deconv = np.fft.ifft(fft_y / fft_k, axis=0).real
    
        return deconv



"""----Signal quantification--------------------------------
Sigquanti: class

Frequency (i.e., spiking rate) based quantification is composed of the following steps
5. Calculate the calcium spike frequency with or without pre-processing of deconvolution.

Peak height based quantification
6. Peak height measurement. 

7. Neuronal activity plot and data output"""        
      
      
class Sigquanti:
    def __init__(self, 
                 Processed_cal: np.ndarray,
                 df_cluster: pd.DataFrame,
                 height: int,
                 width: int
                 ):
        """
        Parameters (inputs)
            Processed_cal, 2D array with time series in rows (time along 0th dimension)
                
            df_cluster, gives information about which pixel belongs to which neuron.
        
            Notice that the count of rows in df_cluster 
                should be corresponding to the 1st dimension of Calcium_imaging data,
                namely the pixel number should be the same. 
                This is risky when messing up core, border and whole neuron.
        
        Attributes (outputs)
            
        """
        if Processed_cal.ndim != 2:
            raise ValueError("Calcium_imaging must be a 2D NumPy array (time x features).")
        if Processed_cal.shape[1] != df_cluster.shape[0]:
                raise ValueError(
                    f"Row mismatch: Calcium_imaging has {Processed_cal.shape[0]} rows, "
                    f"but df_cluster has {df_cluster.shape[0]} rows."
                    f"The calcium imaging is not aligned with the DataFrame of neuronal segmentation"
                )
        if "Cluster" not in df_cluster.columns:
                raise ValueError("df_cluster must contain a column named 'Cluster' that contains the information of neuronal segmentation")
        
        self.cal= Processed_cal
        self.df=  df_cluster
        self.height= height
        self.width= width
    
    
    """---------------------------------Spiking rate quantification and plot------------------------
    A function to plot a pixel's curve, check spike sorted by setting a cutoff"""
    def Get_spike_plot(self, pixel_index: int,
                    spike_h_cutoff: float,#peak height cutoff used to define calcium spikes
                    time_start: int, time_end: int, #which time window to show
                    figsize: list= (30,5) ):
        time = np.arange(time_start, time_end)
        curve= self.cal[time_start:time_end, pixel_index]
        peaks, _= find_peaks(curve, height= spike_h_cutoff)
        
        _, ax = plt.subplots(figsize=figsize)
        ax.plot(time, curve)
        ax.plot(time[peaks], curve[peaks],"x",color = "purple", label="Spikes")
        ax.axhline(spike_h_cutoff, color='red', linestyle='--', alpha=0.5, label="Threshold")
        
        plt.xticks(rotation='vertical')
        ax.set_xlabel("Time frame")
        ax.set_ylabel("ΔF/F0")
        ax.set_title("Fluorescence Curves")
        plt.tight_layout()
        
    
    def _freq_calculator_array(self, array,
                              spike_h_cutoff: float, #peak height cutoff used to define calcium spikes
                              seg_start: int, seg_end: int, #pick a time window to calculate spike frequency
                              fs:float #sample frequency of calcium imaging, useful for spike frequency to be in Hz
                              ):
        """
        Fully vectorized version of spike sorting.
        Much faster than columnwise scipy.find_peak
        """
        y = array[seg_start:seg_end, :]  # shape (time, pixels)

        # Detect peaks column-wise
        peaks_mask = (y[1:-1, :] > y[:-2, :]) & \
                     (y[1:-1, :] > y[2:, :]) & \
                     (y[1:-1, :] > spike_h_cutoff)

        return peaks_mask.sum(axis=0) / (y.shape[0] / fs)
    
    def _heatmap_matrix_from_df(self, value):
        v = np.asarray(value).ravel()
        if self.df.shape[0] != len(v):
            raise ValueError(
                f"Row mismatch: the quantified values per pixel shows {len(v)} pixels, "
                f"different from the DataFrame's row number: {self.df.shape[0]}."
                f"The quantified value is not aligned with the DataFrame. Then should represent the same pixel number"
            )
            
        x= self.df['X'].to_numpy()
        y= self.df['Y'].to_numpy()
        img = np.zeros((self.height , self.width), dtype=v.dtype)
        img[y, x]= v
        return img
        
    
    def Freq_heatmap(self, 
                      spike_h_cutoff: float, #peak height cutoff used to define calcium spikes
                      seg_start: int, seg_end: int, #pick a time window to calculate spike frequency
                      fs:float):
        freq= self._freq_calculator_array(self.cal, spike_h_cutoff, seg_start, seg_end, fs)
        img = self._heatmap_matrix_from_df(freq)
        plt.figure(figsize=(self.height/5, self.width/5))
        plt.imshow(img, cmap='hot', origin='upper', interpolation='none', aspect='equal')

        
    def IF_freq_heatmap(self, 
                      spike_h_cutoff: float, #peak height cutoff used to define calcium spikes
                      seg_start: int, seg_end: int, #pick a time window to calculate spike frequency
                      fs:float):
        if "CGRP" not in self.df:
                raise ValueError("df_cluster does not contain the column 'CGRP' ")
        if "NF" not in self.df:
                raise ValueError("df_cluster does not contain the column 'NF' ")
        cgrp= self._heatmap_matrix_from_df(self.df['CGRP'])
        nf= self._heatmap_matrix_from_df(self.df['NF'])
        
        fig, axes = plt.subplots(1,3, figsize= (self.height/5,self.width*3/5) )
        ax=axes[0]
        freq= self._freq_calculator_array(self.cal, spike_h_cutoff, seg_start, seg_end, fs)
        img = self._heatmap_matrix_from_df(freq)
        ax.imshow(img, cmap='hot', origin='upper', interpolation='none', aspect='equal')
        
        ax.set_title("Spiking frequncy", fontsize=60)
        ax.axis('off')
        
        ax=axes[1]
        ax.imshow(cgrp, origin='upper', interpolation='none', aspect='equal')
        ax.set_title("CGRP", fontsize=60)
        ax.axis('off')
        
        ax=axes[2]
        ax.imshow(nf, origin='upper', interpolation='none', aspect='equal')
        ax.set_title("NF", fontsize=60)
        ax.axis('off')
        
        plt.tight_layout()
        

    def Freq_data_output(
            self, 
            df: pd.DataFrame,
            colname:str,#how do you call this data? e.g., spikeing rate (Hz))
            spike_h_cutoff: float, #peak height cutoff used to define calcium spikes
            seg_start: int, seg_end: int, #pick a time window to calculate spike frequency
            fs:float):
        """The Dataframe with XY pixel infor you use to store the data.
           Usually make a copy of _IF_match output.
           Notice that this df's row number have to be the same as the class input df_clust
           """
        if self.cal.shape[1] != df.shape[0]:
                raise ValueError(
                    f"Row mismatch: Calcium_imaging has {self.cal.shape[0]} rows (pixels), "
                    f"but df has {df.shape[0]} rows (pixels). In Freq_data_output."
                    f"The Dataframe you use to store the quantification does not align with the calcium imaging data analyzed"
                )
        freq= self._freq_calculator_array(self.cal, spike_h_cutoff, seg_start, seg_end, fs)
        merged = df
        merged[colname]=freq
        
        return merged

    """------Peak height based quantification--------------------------
    
        Important notes:      
            1. These window definitions can be learnt from Sigprocess(...).Pre_curve_plot(...)
            2. One way to intuitively show the response is:
                Sigprocess(Calcium_imaging: np.ndarray,
                df_cluster: pd.DataFrame).Curve_plot(self, Cal_filt,
                                time_start: int, time_end: int, #which time window to show
                                figsize: list= (30,5),
                                x_interval: int= 200)
                                                     """    
    
    def _peak_height_array(self, array,
                      base_start: int, base_end: int,
                      sti_start: int, sti_end: int):
        """Calculate peak height for all pixels.

        Parameters
            base_start, base_end : int Baseline window
            sti_start, sti_end : int Stimulation window

        Returns
            np.ndarray, Peak height for each pixel (shape: pixels,)
        """
        baseline = np.mean(array[base_start:base_end, :], axis=0)
        stim_max = np.max(array[sti_start:sti_end, :], axis=0)
        return stim_max - baseline
    
    
    def Peak_height_heatmap(self, 
                      base_start: int, base_end: int,
                      sti_start: int, sti_end: int):
        ph= self._peak_height_array(self.cal, base_start, base_end, sti_start, sti_end)
        img = self._heatmap_matrix_from_df(ph)
        plt.figure(figsize=(self.height/5, self.width/5))
        im = plt.imshow(img, cmap='hot', origin='upper', interpolation='none', aspect='equal')
        
        cbar= plt.colorbar(im, shrink=0.4)
        cbar.set_label(label='Peak height (a.u.)',fontsize=70)
        cbar.ax.tick_params(labelsize=50)
        
        
        
    def Peak_height_output(
            self, 
            df: pd.DataFrame,
            colname:str,#how do you call this data? e.g., peak height (a.u.))
            base_start: int, base_end: int,
            sti_start: int, sti_end: int):
        """df is the Dataframe with XY pixel infor you use to store the data.
          Usually make a copy of _IF_match output.
          Notice that this df's row number have to be the same as the class input df_clust
          """
        if self.cal.shape[1] != df.shape[0]:
               raise ValueError(
                   f"Row mismatch: Calcium_imaging has {self.cal.shape[0]} rows (pixels), "
                   f"but df has {df.shape[0]} rows (pixels). in Peak_height_output"
                   f"The Dataframe you use to store the quantification does not align with the calcium imaging data analyzed"
               )
        ph= self._peak_height_array(self.cal, base_start, base_end, sti_start, sti_end)
        merged = df
        merged[colname]=ph
       
        return merged 
    
    
    """
    Example to aggregate DataFrame from pixel level to neuron level
    df_aggre= df.groupby("Cluster").agg(
        freq_mean=("freq", "mean"),
        freq_median=("freq", "median"),
        peak_mean=("peak_height", "mean"),
        peak_max=("peak_height", "max"),
        n_pixels=("Cluster", "size"))"""
        
    
    
    
        

       
       
       
       