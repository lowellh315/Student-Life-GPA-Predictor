#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import *
import glob
import scipy 
from datetime import datetime as dt
import sklearn
sns.style = 'darkgrid'
import ruptures as rpt


# In[2]:


# read the audio file skipping three out of every 4 rows
audio = pd.read_csv('tables/audio/audio.csv', skiprows = [i if i%60 != 0 else 1 for i in range(100000)])


# In[3]:


audio


# In[4]:


def location_intervals(indoor_locations): 
    """
    given indoor_locations dataframe, returns a list of tuples [(start timestamp, end timestamp)] that represents intervals
    where the person was in those locations continuously (sensors within 30 minutes of each other)
    """
    # the following code creates a list of tuples which represent the ranges of timestamps where users were inside
    start_timestamp = None
    loc_intervals = []
    for i in indoor_locations.index:
        
        current_timestamp = indoor_locations.loc[i]['timestamp']
        
        if start_timestamp is None: 
            start_timestamp = current_timestamp
            
        # don't consider locations continuous when sensed more than 1/2 hour apart
        if current_timestamp > start_timestamp + 1800:
            loc_intervals.append((start_timestamp, current_timestamp))
            start_timestamp = None
            continue
        try: 
            # if the next index is not inside, we go to the except loop, otherwise continue
            indoor_locations.loc[i + 1]
            continue
        except: 
            # in the event the next index is not inside, this is the end of the time range so we append that to the intervals
            if current_timestamp - start_timestamp >= 1800: 
                loc_intervals.append((start_timestamp, current_timestamp))
            start_timestamp = None
            continue
    

    return loc_intervals


# In[5]:


def party_intervals(audio, loc_intervals): 
    """
    given input audio dataframe and loc_intervals containing timestamp location intervals
    """
    loc_index = 0
    # this will hold the intervals to return in format (start, end, avg audio inference)
    total_audio_intervals = []
    # this will hold the intermediate steps for a single interval in the total list above
    # loop through all audio inferences
    count = 0 
    for inter in loc_intervals: 
        
        num_silent_labels = 0
        num_current_labels = 0
    
        int_start= inter[0]
        int_end = inter[1]
        
        done = False
        
        while done is False: 
            try: 
                time = audio.iloc[count]['timestamp']
            except: 
                done = True 
                continue
                
            if time < int_start: 
                count += 1
            elif time > int_end: 
                if num_current_labels > 0: 
                    
                    silent_labels = num_silent_labels/num_current_labels  
                    
                    if silent_labels < .4: 
                        total_audio_intervals.append((int_start, int_end, 1-silent_labels))
                        
                done = True
            else: 
                if audio.iloc[count][' audio inference'] == 0: 
                    num_silent_labels += 1
                num_current_labels += 1
                count += 1
                continue
    
    
    """
    for i in audio.index: 
        time = audio.loc[i]['timestamp']
        done = False
        while done is False: 
            # find the overlap between audio files and the location intervals
            try: 
                int_start = loc_intervals[loc_index][0]
                int_end = loc_intervals[loc_index][1]
            except: 
                done = True
                continue
            # if the audio time is before
            if time < int_start: 
                done = True
                continue
            elif time > int_end: 
                loc_index += 1
                if num_current_labels > 0: 
                    silent_labels = num_silent_labels/num_current_labels  
                    
                    if int_end - int_start >= 1200 and silent_labels < 0.4: 
                        total_audio_intervals.append((int_start, int_end, 1-silent_labels))
                continue
            else: 
                if audio.loc[i][' audio inference'] == 0: 
                    num_silent_labels += 1
                num_current_labels += 1
                done = True
                continue
        """
    return total_audio_intervals


# In[9]:


def find_partying(audio, wifi_locations, party_locs): 
    """
    inputs: audio dataframe, wifi_locations dataframe, and potential party locations
    outputs: the intervals of wifi location where the corresponding audio inference was >= .4 where the user was also in
    a party location. 
    """
    
    party_loc_data = wifi_locations[wifi_locations['location'].isin(party_locs)]
    
    total_party_df = pd.DataFrame()
    
    for uid in wifi_locations['uid'].unique(): 
        
        print(uid)
        
        location_ints = location_intervals(party_loc_data[party_loc_data['uid'] == uid])
    
        audio_ints = np.asarray(party_intervals(audio[audio['uid'] == uid], location_ints))
    
        try: 
            party_df = pd.DataFrame({'start time': audio_ints[:, 0],
                                     'end time': audio_ints[:, 1], 
                                     'proportion loud labels': audio_ints[:, 2],
                                     'uid': uid})
        except: 
            continue
        party_df['duration'] = party_df['end time'] - party_df['start time']
        
        total_party_df = total_party_df.append(party_df, ignore_index = True)
        
    return total_party_df


# In[10]:


party_locs = [
    'in[mclaughlin]', 'in[steele]', 'in[east-wheelock]', 'in[north-park]', 'in[north-main]', 'in[robinson]',
    'in[maxwell]', 'in[fahey-mclane]', 'in[fayerweather]', 'in[massrow]', 'in[ripley]', 'in[woodward]', 'in[butterfield]', 
    'in[cutter-north]', 'in[french]', 'in[hallgarten]', 'in[gile]', 'in[newhamp]', 'in[hitchcock]', 'in[smith]', 
    'in[channing-cox]', 'in[Cohen]', 'in[whittemore]', 'in[tllc-raether]', 'in[tllc]', 'in[richardson]', 'in[judge]', 
    'in[bissell]'
]
# the party_locs list includes all dorms and frat houses in the unique locations set
wifi_locs = pd.read_csv('tables/wifi_location/wifi_location.csv')


# In[11]:


parties = find_partying(audio, wifi_locs, party_locs)


# In[12]:


parties.to_csv('dataset/sensing/partying/partying.csv', index = False)


# In[17]:


parties['day of week'] = pd.to_datetime(parties['start time'], unit = 's').dt.dayofweek
parties['day of year'] = pd.to_datetime(parties['start time'], unit = 's').dt.dayofyear
parties

