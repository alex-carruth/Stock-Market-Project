# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 20:08:12 2021

@author: acarr
"""
import pandas as pd
from illumima_dataorg import ilmn
from ilmn_insider_2 import Insider

ilmn_final = pd.merge(left=ilmn,right=Insider, how='left', left_index=True,right_index=True)
ilmn_final['Number of Securities Transacted'] = ilmn_final['Number of Securities Transacted'].fillna(0)
ilmn_final['Number of Securities Owned'] = ilmn_final['Number of Securities Owned'].fillna(0)
ilmn_final['Amount'] = ilmn_final['Amount'].fillna(0)

