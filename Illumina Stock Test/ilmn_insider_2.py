# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 20:34:48 2021

@author: acarr
"""
from datetime import date
from dateutil.relativedelta import relativedelta
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import pandas_datareader.data as web

def to_soup(url):
    url_response = requests.get(url)
    webpage = url_response.content
    soup = BeautifulSoup(webpage, 'html.parser')
    return soup

#Picks up the Insider Trades data
today = date.today()
prev = today - relativedelta(years=5)
start_yahoo = prev.strftime("%d/%m/%Y")
end_yahoo = today.strftime("%d/%m/%Y")

end = prev.strftime("%Y-%m-%d")
symbols = ['ILMN']

dfs = []
for i in range(len(symbols)):
    lst = [symbols[i]]
    cik = ['1110803']
    page = 0
    beg_url = 'https://www.sec.gov/cgi-bin/own-disp?action=getissuer&CIK='+cik[0]+'&type=&dateb=&owner=include&start='+str(page*80)
    urls = [beg_url]
    df_data = []
    for url in urls:
        soup = to_soup(url)
        transaction_report = soup.find('table', {'id':'transaction-report'})

        t_chil = [i for i in transaction_report.children]
        t_cont = [i for i in t_chil if i != '\n']

        headers = [ i for i in t_cont[0].get_text().split('\n') if i != '']
        data_rough = [i for lst in t_cont[1:] for i in lst.get_text().split('\n') if i != '' ]
        data = [data_rough[i:i+12] for i in range(0,len(data_rough), 12)]
        last_line = data[-1]
        for i in data:
            if (end > i[1]):
                break
            else:
                if (i != last_line):
                    df_data.append(i)
                else:
                    df_data.append(i)
                    page += 1
                    urls.append('https://www.sec.gov/cgi-bin/own-disp?action=getissuer&CIK='+cik[0]+'&type=&dateb=&owner=include&start='+str(page*80))

    df = pd.DataFrame(df_data,columns = headers)                
    df['Purch'] = pd.to_numeric(df['Acquistion or Disposition'].apply(lambda x: 1 if x == 'A' else (0))
                   *df['Number of Securities Transacted'])
    df['Sale'] = pd.to_numeric(df['Acquistion or Disposition'].apply(lambda x: 1 if x == 'D' else 0)
                   *df['Number of Securities Transacted'])
    df['Amount']=df['Purch'].fillna(df['Sale']*-1)
    
    
    
    dfLabels = ['Sale','Purch','Reporting Owner', 'Acquistion or Disposition','Deemed Execution Date', 'Form', 'Transaction Type','Direct or Indirect Ownership','Line Number', 'Owner CIK','Security Name']
    df=df.drop(columns=dfLabels, axis=1)
    
dfKeepLabels = ['Number of Securities Transacted', 'Number of Securities Owned','Amount']

df['Number of Securities Transacted'] = df['Number of Securities Transacted'].astype(float)
df['Number of Securities Owned'] = df['Number of Securities Owned'].astype(float)

groupedT_vals = pd.DataFrame(df['Number of Securities Transacted'].groupby(df['Transaction Date']).sum())
groupedO_vals = pd.DataFrame(df['Number of Securities Owned'].groupby(df['Transaction Date']).sum())
groupedA_vals = pd.DataFrame(df['Amount'].groupby(df['Transaction Date']).sum())    
Insider = pd.merge(groupedT_vals, groupedO_vals, how='outer', on = 'Transaction Date')
Insider = pd.merge(Insider, groupedA_vals, how='outer', on = 'Transaction Date')