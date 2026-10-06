# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 20:14:33 2026
Revised on Sun Oct 04 13:02:40 2026

@author: Junxuan Ma
"""

import czifile
import numpy as np
import os

#Read confocal files
"""
imgimport.FetchFiles:
    First fetch all the CZI file names in the folder, they should be calcium imaging files.
        Import one of the file out of the file names, e.g., a calcium imaging file.

    Then replace the calcium imaging tag with the IF tag to get the corresponding IF file name.
        Import the IF file in the corresponding folder storing IF files.

Example:
    Specifically: IF is in a form of "IF_sample_name.czi".
    And Calcium imaging is in a form of "Cal_sample_name.czi".
    The sample name would be "sample_name.czi" .
    ***Note that the tags are case-insensitive in windows. Avoid another tag with different case (capitalization: avoid "if_Cal.czi" for both IF_ and Cal_ will fetch this file).

Storage of the files:
    Calcium imaging and IF files can be stored in the same folder or different folders. 
    If they are in the same folder, then folder_path_cal and folder_path_if can be the same.

General analysis pipeline design:
    Since the files are very large and there is no memory to store all simutaneously, it has to be read one and analyze, then read the next.

Detailed instruction of use:
Thus the Auto_ImportCZI contains 2 steps:
    Fetch_Files: list all file names representing all Samples

        Inputs (parameters): 
            file_path: the local directory of the image files (can be either calcium imaging or IF files).
            file_tag: 
                If calcium imaging and IF files are in the same folder, then file_tag should be "Cal_" or "IF_" to distinguish the two types of files.
                If calcium imaging and IF files are in different folders, then file_tag can be ".czi" to fetch all files in the folder, since they are already separated by folder.

        
        Outputs (attributes):
            Get_filenames: a function to call all the calcium imaging/IF file names.
        
    
    ImportCZI: choose one to import and make it easy for iteration
        Inputs (parameters): 
            folder_path_cal: the local directory of the calcium image file names
            folder_path_if: the local directory of the IF image file names.
            Cal_name: the filename of one calcium imaging chosen for analysis.
            Cal_tag: the string pattern in the file name to distinguish Calcium imaging from IF
            IF_tag: the string pattern specific for IF file

        
        Outputs (attributes):
            Read_if: 
                Import the IF file and output the CGRP and NF image file in np.ndarray, respectively.
                Inputs: which channel index is "CGRP" and which is "NF". 
                Output: the CGRP and NF image file in np.ndarray, respectively.
                    
                    Example: CGRP_raw, NF_raw= ImportCZI.Read_if(CGRP= 1, NF= 0)

            Read_cal: 
                Output: the raw calcium imaging data as np.ndarray
                and the pixel size of the image in a list of (time, height, width).
    
    
Note the read_image function needs to be adapted based on the format of image
For the following analysis the imported calcium imaging data should be np.array and need to have time in 0th dimention
    
""" 
#Step 1, list all files representing all samples
from pathlib import Path
class FetchFiles:
    def __init__(
        self,
        file_path: str,
        file_tag: str = "Cal",
    ) -> None:
        self._path = Path(file_path)
        self._tag = file_tag

        if not self._path.exists():
            raise FileNotFoundError(f"Path does not exist: {self._path}")

    # ---------- Internal logic ----------
    def _get_filenames(self, tag: str):
        return [f.name for f in self._path.glob(f"*{tag}*.czi")]

    # ---------- Public API ----------
    def Get_filenames(self):
        return self._get_filenames(self._tag)


    

# Step 2, choose one to import and make it easy for iteration
class ImportCZI():    
    def __init__(self, 
                 folder_path_cal: str,
                 folder_path_if: str,
                 Cal_name: str = 'Cal_Donor1_Day2_G5_b5_2.czi',
                 cal_tag: str = "Cal",
                 if_tag: str = "IF"): 
        IF_name = Cal_name.replace(cal_tag, if_tag)
        self.Cal_name = Path(folder_path_cal) / Cal_name #Full path object, not just file name
        self.IF_name = Path(folder_path_if) / IF_name #Full path object, not just file name

        
        
        
    # ---------- Internal logic ----------
    def _read_IF(self, CGRP: int, NF: int) ->tuple[np.ndarray,np.ndarray]:
        if CGRP >= self._if_data.shape[0] or NF >= self._if_data.shape[0]:
            raise IndexError("CGRP or NF index out of bounds of channels")
        return self._if_data[CGRP,:,:], self._if_data[NF,:,:]
    
    def _read_Cal(self):
        """The same as:
            return np.reshape(self._cal_data,(self._cal_data.shape[0],
               self._cal_data.shape[1]*self._cal_data.shape[2]))"""
        return np.reshape(self._cal_data,(self._cal_data.shape[0],-1))
    
    def _cal_shape(self):
        return self._cal_data.shape
    
    # ---------- Public API ----------
    def IF_exists(self) -> bool:
        return os.path.exists(self.IF_name)

    def Read_if(self, CGRP: int, NF: int)->tuple[np.ndarray,np.ndarray]:
        self._if_data = np.squeeze(czifile.imread(self.IF_name))
        return self._read_IF(CGRP,NF)
    
    def Read_cal(self) -> tuple[np.ndarray, tuple[int, ...]]:
        self._cal_data = np.squeeze(czifile.imread(self.Cal_name))
        return self._read_Cal(), self._cal_shape()
            
            
            
        
        
