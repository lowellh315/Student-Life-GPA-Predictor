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


def find_intervals(activity_df, locations_df): 
    """
    given an activity_dataframe and a locations dataframe takes the first step for calculating
    the indoor mobility, or average activity inference for times where the user was inside.
    This function assumes that locations_df and activity_df are already ordered by date. 
    This initial function outputs two lists: 
        loc_intervals containing the time intervals where locations are inside
        activities_intervals contianing the time intervals for each activity inference so the activities can be 
        processed with a reasonable runtime. 
    """
    # create a dataframe of all indoor locations
    indoor_booleans = locations_df['location'].apply(lambda l: l[:2]) == 'in'
    indoor_locations = locations_df[indoor_booleans]
    
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
            loc_intervals.append((start_timestamp, current_timestamp))
            start_timestamp = None
            continue

    # the next block of code reduces the activity intervals down to a list of tuples, each tuple containing a start
    # and end timestamp, along with the activity inference for that range of time. This will vastly reduce the number of
    # iterations we have to do to loop through activities in the future. 
    activities_intervals = [] 
    start_timestamp = None 
    act = activity_df.iloc[0]
    
    for i in activity_df.index:
        # this try except loop takes care of the last element
        try: 
            next_act = activity_df.loc[i+1]
        except: 
            continue
        
        # the current and next activity inferenecs
        inf = act[' activity inference']
        next_inf = next_act[' activity inference']
        
        # if the start timestamp is none, we set it to the current timestamp
        if start_timestamp is None: 
            start_timestamp = act['timestamp']

        # if the activity inference changes, make the interval break
        # also, if the timestamps are more than a half hour apart, don't consider this a continuous interval
        if inf != next_inf or next_act['timestamp'] > act['timestamp'] + 1800: 
            end_timestamp = act['timestamp']
            activities_intervals.append((start_timestamp, end_timestamp, inf))
            start_timestamp = next_act['timestamp']
        act = next_act

    """
    at this point we have two sets of intervals, loc_intervals containing the time intervals where locations are inside
    and activities_intervals contianing the time intervals for each activity inference so the activities can be 
    processed with a reasonable runtime. Now, the goal is to match up the intervals where the timestamps for activities
    overlap with timestamps from indoor locations
    """
               
    return loc_intervals, activities_intervals


# In[3]:


def indoor_mobility(loc_intervals, activity_intervals): 
    """
    given the wifi and activity intervals as described above, calculates the indoor mobility for the user given those
    intervals
    """
    # counter tracks the index of activity_intervals we are currently on
    counter = 0 
    indoor_mobility = pd.DataFrame()
    
    for inter in loc_intervals: 
        
        loc_inter_start = inter[0]
        loc_inter_end = inter[1]
        
        if loc_inter_start is None: 
            print(inter)
        done = False
        
        while done is False: 
            try: 
                act_inter_start = activity_intervals[counter][0]
            except: 
                break
            act_inter_end = activity_intervals[counter][1]

            if act_inter_end < loc_inter_start: 
                counter += 1
                continue
                
            # if the start of the activity interval is greater than that of the location interval, we're done with this loc.
            elif act_inter_start > loc_inter_end: 
                done = True
                continue

            im_start = max(act_inter_start, loc_inter_start)
            
            if act_inter_end <= loc_inter_end: 
                im_end = act_inter_end
                
                temp = pd.DataFrame({'start timestamp': im_start, 
                                 'end timestamp': im_end, 
                                 'duration (s)': [im_end - im_start],
                                 'activity inference': [activity_intervals[counter][2]]})
        
                indoor_mobility = indoor_mobility.append(temp, ignore_index = True)
                counter += 1
                continue
                
                
            if act_inter_end > loc_inter_end: 
                im_end = loc_inter_end
                done = True
                
            temp = pd.DataFrame({'start timestamp': im_start, 
                                 'end timestamp': im_end, 
                                 'duration (s)': [im_end - im_start],
                                 'activity inference': [activity_intervals[counter][2]]})
        
            indoor_mobility = indoor_mobility.append(temp, ignore_index = True)
    
    return indoor_mobility


# In[4]:


def indoor_mobility_by_day(indoor_mob, act_col): 
    """
    given an indoor mobility dataframe, calculate the average indoor mobility per day the dataframe.
    returns a dataframe contianing avg indoor mobility per day and uid
    """
    agg_df = pd.DataFrame()
    for day in indoor_mob['day'].unique(): 
        total_duration = 0
        aggregate_act = 0
        # average activity inference per day = duration*activity inference/total duration
        for row in indoor_mob[['duration (s)', act_col]][indoor_mob['day'] == day].values:
            total_duration += row[0]
            aggregate_act += row[0]*row[1]
        aggregate_act = aggregate_act/total_duration

        agg_df = agg_df.append(pd.DataFrame({'average indoor mobility': [aggregate_act], 
                                    'day': [day]}))
    agg_df['uid'] = indoor_mob.uid.unique()[0]
    
    return agg_df


# In[5]:


def final_indoor_mobility_process(activities, wifi_locations): 
    """
    combines the three functions above to process indoor mobility for all users given the activitites and wifi locations
    dataframes. Returns the indoor mobility aggregated for all users
    """
    # initialize indoor mobility dataframes
    indoor_mob = pd.DataFrame() 
    day_im = pd.DataFrame()
    evening_im = pd.DataFrame()
    night_im = pd.DataFrame()
    
    # take the intersection of the uids from the two dataframes
    uids = set(activities.uid.unique()) | set(wifi_locations.uid.unique())
    # loop through every uid
    for uid in uids: 
        print(uid)
        # find activity and location intervals for the uid
        loc_int, act_int = find_intervals(activities[activities['uid'] == uid],
                                          wifi_locations[wifi_locations['uid'] == uid])
        
        # find the overlapping indoor mobility dataframe (containing duration inside + activity inference as columns)
        im = indoor_mobility(loc_int, act_int)
        im['uid'] = uid
        im['time'] = pd.to_datetime(im['start timestamp'], unit = 's') 
        im['date'] = im['time'].apply(lambda x: dt.strftime(x, '%Y-%m-%d'))
        
        # reuse the activity function to make epochs and reformat the indoor mobility grouping by day
        tot, day, evening, night = aggregate_activities(activity_epochs(im))
        
        # apply the formula 
        indoor_mob = indoor_mob.append(indoor_mobility_by_day(tot, 'activity inference'), ignore_index = True)
        day_im = day_im.append(indoor_mobility_by_day(day, 'activity inference'), ignore_index = True)
        evening_im = evening_im.append(indoor_mobility_by_day(evening, 'activity inference'), ignore_index = True)
        night_im = night_im.append(indoor_mobility_by_day(night, 'activity inference'), ignore_index = True)
        
    indoor_mob = indoor_mob.rename(columns = {' activity inference': 'indoor mobility'})
    day_im = day_im.rename(columns = {'day activity inference': 'day indoor mobility'})
    evening_im = evening_im.rename(columns = {'evening activity inference': 'evening indoor mobility'})
    night_im = night_im.rename(columns = {'night activity inference': 'night indoor mobility'})
        
    return indoor_mob, day_im, evening_im, night_im


# In[6]:


def aggregate_activities(activity): 
    """
    given input activity dataframe this function organizes that dataframe by day and uid, then returns a total dataframe 
    along with a dataframe for each epoch - day, evening, and night. 
    """
    # This code is a little clunky, but I thought formatting it in a function would take a longer runtime because it would 
    # require more for loops, and the activity data already takes a long time to process since it is a large dataset. 

    # the goal of this function is to create a new dataframe with the average activity inference for each day for each user. 
    # it will also make dataframes 

    # I will aggregate the data in the folloing data frames
    aggreg = pd.DataFrame()
    day_aggreg = pd.DataFrame()
    evening_aggreg = pd.DataFrame()
    night_aggreg = pd.DataFrame()

    # loop through each uid
    for uid in activity.uid.unique(): 

        # take the uid specific data, group it by day, find the mean, and then append each dataframe to its respective aggregate.
        uid_data = activity[activity['uid'] == uid]
        day = uid_data[uid_data['epoch'] == 'Day']
        evening = uid_data[uid_data['epoch'] == 'Evening']
        night = uid_data[uid_data['epoch'] == 'Night']

        uid_data['uid'] = uid
        day['uid'] = uid
        evening['uid'] = uid
        night['uid'] = uid

        aggreg = aggreg.append(uid_data)
        day_aggreg = day_aggreg.append(day)
        evening_aggreg = evening_aggreg.append(evening)
        night_aggreg = night_aggreg.append(night)
        
    # make the columns in each dataframe more descriptive
    activity = aggreg
    
    # make day activities dataframe
    day_activity = day_aggreg
    day_activity = day_activity.rename(columns = {' activity inference': 'day activity inference'})

    # make evening activities dataframe
    evening_activity = evening_aggreg
    evening_activity = evening_activity.rename(columns = {' activity inference': 'evening activity inference'})

    # make night activities dataframe
    night_activity = night_aggreg
    night_activity = night_activity.rename(columns ={' activity inference': 'night activity inference'})
    
    return activity, day_activity, evening_activity, night_activity


# In[7]:


def epoch(times_tuple):
    """
    input: tuple containing start and end times (in hours on 24 hour scale)
    output: the epoch that corresponds to the timestamps
    note: we chose to only return timestamps that had both start and end time within a single epoch. An alternative would
    be splitting conversations that span multiple epochs into two conversations, one in each epoch, but we decided not to 
    because that would double count conversations for each user.  
    """
    start = times_tuple[0]
    end = times_tuple[1]
    
    # Day epoch: hours 10 am -6 pm
    if (start and end) >= 10 and (start and end) <18 :
        return 'Day'
    # Night epoch: 12 am - 10 am
    elif (start and end)<10:
        return 'Night'
    # evening epoch: 6 pm - 12 am
    elif (start and end) >= 18:
        return 'Evening'


# In[8]:


def activity_epochs(activity):
    # these lines find the day and hour of activity inference
    activity['day'] = activity['date'].apply(lambda x: dt.strptime(x, '%Y-%m-%d')).dt.dayofyear

    activity['epoch'] = pd.to_datetime(activity['time']).dt.hour

    # next, apply the epoch function to find the epoch of each activity
    activity['epoch'] = list(zip(activity['epoch'], activity['epoch']))

    activity['epoch'] = activity['epoch'].apply(epoch)
    
    return activity


# In[9]:


activities = pd.read_csv('tables/activity/activity.csv')


# In[10]:


wifi_locations = pd.read_csv('tables/wifi_location/wifi_location.csv')


# In[11]:


indoor_mob, day_im, evening_im, night_im = final_indoor_mobility_process(activities, wifi_locations)


# In[14]:


indoor_mob.to_csv('dataset/sensing/indoor_mobility/indoor_mobility.csv', index = False)
day_im.to_csv('dataset/sensing/indoor_mobility/day_indoor_mobility.csv', index = False)
evening_im.to_csv('dataset/sensing/indoor_mobility/evening_indoor_mobility.csv', index = False)
night_im.to_csv('dataset/sensing/indoor_mobility/night_indoor_mobility.csv', index = False)


# In[19]:


indoor_mob['average indoor mobility'].mean()


# In[17]:


day_im['average indoor mobility'].max()


# In[ ]:




