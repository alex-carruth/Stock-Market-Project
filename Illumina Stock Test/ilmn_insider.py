# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 15:47:27 2021

@author: acarr
"""

import json
import urllib.request
from datetime import date
from dateutil.relativedelta import relativedelta

#API Key
TOKEN = "9f800bf8345337d8af4c144a6067c1940f2b41e921e84f2761e3698a5a5abe94"

#API endpoint
API = "https://api.sec-api.io?token=" + TOKEN

today = date.today()
prev = today - relativedelta(years=5)
dt = today.strftime("%Y-%m-%d")
pdt = prev.strftime("%Y-%m-%d")
#Filter parameters
filt = "formType:\"4\" AND formType:(NOT \"4/A\") AND filedAt[" + pdt + " TO " + dt + "]"

start = 0

#Return 10,000 filings per API call
size = 10000

#Sort by filedAt
sort = [{ "filedAt": { "order": "desc"} }]

payload = {
        "query": { "query_string": {"query": filt}},
        "from": start,
        "size": size,
        "sort": sort    
}