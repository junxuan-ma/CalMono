# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 20:14:33 2026

@author: Junxuan Ma
"""

import glob
import czifile
import numpy as np
import cv2
import os

#Read confocal files
"""
To automate import, match IF (immunofluorescence) with calcium imaging with the same Sample Name.
Specifically: IF is in a form of "IF_sample_name.czi".
And Calcium imaging is in a form of "Cal_sample_name.czi".
The sample name would be "sample_name.czi" .
Note that the IF_ or Cal_ tag has to be before the same sample name.

All the calcium imaging files and IF files are stored in the folder defined by the input Local_path
Since the files are very large and there is no memory to store all simutaneously, it has to be read one and analyze, then read the next.

Thus the Auto_ImportCZI contains 2 steps:
    1. Fetch_Files: list all files representing all samples
    Inputs (parameters): 
    --------
    Local_path: the local directory of the image files
    Cal_tag: the string pattern in the file name that can be used to distinguish Calcium imaging from IF
    IF_tag: the string pattern specific for IF file
        Note: calcium imaging file name can be switched to IF file name
        by changing Cal_tag to IF_tag 
    
    Outputs (attributes):
    ---------
    Get_cal_files: a function to call all the calcium imaging file names.
    Get_if_files: a function to call all the IF file names
        
    
    2. ImportCZI: choose one to import and make it easy for iteration
    Notice that the IF and calcium imaging file names have to be well matched.
    The only difference is their tags before the sample name.
    Inputs (parameters): 
    --------
    local_path: the local directory of the image files
    Cal_name: the filename of one calcium imaging chosen for analysis
    Cal_tag: the string pattern in the file name that can be used to distinguish Calcium imaging from IF
    IF_tag: the string pattern specific for IF file
        Note: calcium imaging file name can be switched to IF file name
        by changing Cal_tag to IF_tag 
    
    Outputs (attributes):
    ---------
    Read_if: input which channel index is "CGRP" and which is "NF". 
        output the CGRP and NF image file in np.ndarray, respectively.
    Read_cal: output the raw calcium imaging data as np.ndarray
        and the pixel size of the image in a list of (height, width).
    
    
Note the read_image function needs to be adapted based on the format of image
For the following analysis the imported calcium imaging data should be np.array and need to have time in 0th dimention
    
""" 
#Step 1, list all files representing all samples
from pathlib import Path
class FetchFiles:
    def __init__(
        self,
        local_path: str,
        cal_tag: str = "Cal",
        if_tag: str = "IF",
    ) -> None:
        self._path = Path(local_path)
        self._cal_tag = cal_tag
        self._if_tag = if_tag

        if not self._path.exists():
            raise FileNotFoundError(f"Path does not exist: {self._path}")

    # ---------- Internal logic ----------
    def _get_filenames(self, tag: str):
        return [f.name for f in self._path.glob(f"{tag}*.czi")]

    # ---------- Public API ----------
    def Get_cal_files(self):
        return self._get_filenames(self._cal_tag)

    def Get_if_files(self):
        return self._get_filenames(self._if_tag)
    

#Step 2, choose one to import and make it easy for iteration
class ImportCZI():    
    def __init__(self, 
                 local_path: str,
                 Cal_name: str = 'Cal_Donor1_Day2_G5_b5_2.czi',
                 cal_tag: str = "Cal",
                 if_tag: str = "IF"):
        self._path = Path(local_path)
        self.Cal_name = self._path / Cal_name #Full path object, not just file name
        self._cal_tag = cal_tag
        self._if_tag = if_tag
        
        #To ensure correspondence, IF path is derived from Cal by changing tag
        self.IF_name = self.Cal_name.with_name(#Full path object
            self.Cal_name.name.replace(self._cal_tag, self._if_tag)
        )#Full path object, not just file name
        
        self._if_data = np.squeeze(czifile.imread(self.IF_name))
        self._cal_data = np.squeeze(czifile.imread(self.Cal_name))
        
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
    
    def Read_if(self, CGRP: int, NF: int)->tuple[np.ndarray,np.ndarray]:
        return self._read_IF(CGRP,NF)
    
    def Read_cal(self) -> np.ndarray:
        return self._read_Cal(), self._cal_shape()
            
            
            
        
        
