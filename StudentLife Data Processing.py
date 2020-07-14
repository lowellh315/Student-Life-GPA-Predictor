#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib
import gc, glob, os, datetime, dateutil


# In[6]:


#Adapted from https://github.com/karzak/StudentLife


# In[7]:


#main path to downloaded StudentLife dataset
main_path = "Omkar:/" #change this
ds_path = "dataset"
subfolders = os.listdir(ds_path)
subpaths = dict((i,ds_path+'/'+i) for i in subfolders if i != '.DS_Store')
subfiles = dict((i,os.listdir(subpaths[i])) for i in subpaths)


# In[8]:


# paths to EMA files
response_paths = dict((i,subpaths['EMA']+'/response/'+i) for i in os.listdir(subpaths['EMA']+'/'+subfiles['EMA'][1]) if i != '.DS_Store')
response_files = dict((i,os.listdir(response_paths[i])) for i in response_paths)


# In[9]:


# make a directory for tables
os.chdir(ds_path)
if not os.path.exists('tables'):
    os.mkdir('tables')
    
# make subdirectories (survey, ema, sensors) to save processed data to 
if not os.path.exists('tables/survey'):
    os.mkdir('tables/survey')
    
if not os.path.exists('tables/ema'):
    os.mkdir('tables/ema')
    
if not os.path.exists('tables/sensors'):
    os.mkdir('tables/sensors')


# In[10]:


# paths to ema tables
ema_paths = dict((''.join(i.replace('.csv','')),subpaths['tables']+'/ema/'+i) for i in os.listdir(subpaths['tables']+'/ema') if i != '.DS_Store')


# In[11]:


# paths to survey tables
survey_paths = dict((''.join(i.replace('.csv','')),subpaths['tables']+'/survey/'+i) for i in os.listdir(subpaths['tables']+'/survey') if i != '.DS_Store')


# In[12]:


pd.options.mode.chained_assignment = None


# # Activity EMA

# In[13]:


#Loading all participants data onto one dataframe
os.chdir(main_path + '/dataset/EMA/response/Activity')
filelist = []
for files in glob.glob("*.json"):
    filelist.append(files)
idlist = []
for i in filelist:
    x = (i.split("_")[1]).split(".")[0]
    idlist.append(x)
dflist = []
for x in filelist:
    df = pd.read_json(x, date_unit = 's')
    dflist.append(df)
for x in range(len(dflist)):
    dflist[x]['id'] = idlist[x]
activity_ema = pd.concat(dflist)

#filter by date
activity_ema = activity_ema[activity_ema['resp_time'] > dateutil.parser.parse("2013-04-08")]

#Summarize data to get mean survey response values
activity_ema_summary = activity_ema.groupby('id').mean()
activity_ema_summary.reset_index(inplace = True)
activity_ema_summary = activity_ema_summary.drop(['Social2', 'null'],1)
activity_ema_summary.columns = ('uid', 'relaxing_with_others', 'working_with_others', 'relaxing_alone', 'working_alone')
activity_ema_summary.to_csv(main_path + '/dataset/tables/ema/ActivityEMA.csv', index = False)


# # Exercise EMA responses

# In[ ]:


#Loading all participants data onto one dataframe
os.chdir(main_path + '/dataset/EMA/response/Exercise')
filelist = []
for files in glob.glob("*.json"):
    filelist.append(files)
idlist = []
for i in filelist:
    x = (i.split("_")[1]).split(".")[0]
    idlist.append(x)

dflist = []
for x in filelist:
    df = pd.read_json(x)
    dflist.append(df)
for x in range(len(dflist)):
    dflist[x]['id'] = idlist[x]
exercise_ema = pd.concat(dflist)


# In[ ]:


exercise_intent = exercise_ema[['id', 'schedule']]
#Convert strings to numberic
exercise_intent = exercise_intent.loc[exercise_intent['schedule'].isin([1, '1'])]
exercise_intent = exercise_intent[pd.notnull(exercise_intent['schedule'])]
exercise_intent['schedule'] = exercise_intent['schedule'].astype(int)
#Calculate days where respondants said they planned to work out and skipped
exercise_intent_summary = exercise_intent.groupby('id').sum()
exercise_intent_summary.reset_index(inplace = True)
exercise_intent_summary.columns = ('uid', 'days_skipped_exercise')
#Calculate days exercised
exercise_days = exercise_ema[['id', 'have']]
exercise_days = exercise_days.loc[exercise_days['have'].isin([1, '1'])]
exercise_days = exercise_days[pd.notnull(exercise_days['have'])]
exercise_days['have'] = exercise_days['have'].astype(int)
exercise_days_summary = exercise_days.groupby('id').sum()
exercise_days_summary.reset_index(inplace = True)
exercise_days_summary.columns = ('uid', 'days_exercise')
#Survey responses for walking and exercising
exercise_summary = exercise_ema[['id','exercise', 'walk']]
exercise_summary = exercise_summary.groupby('id').mean()
exercise_summary.reset_index(inplace = True)
exercise_summary.columns = ('uid', 'time_exercise_factor', 'time_walking_factor')
#Merge the data
exercise_summary = exercise_summary.merge(exercise_days_summary, on = 'uid').merge(
exercise_intent_summary, on = 'uid')
exercise_summary.to_csv(main_path + '/dataset/tables/ema/Exercise_EMA.csv', index = False)


# # Mood EMA responses

# In[ ]:


#Loading all participants data onto one dataframe
os.chdir(main_path + '/dataset/EMA/response/Mood')
filelist = []
for files in glob.glob("*.json"):
    filelist.append(files)
idlist = []
for i in filelist:
    x = (i.split("_")[1]).split(".")[0]
    idlist.append(x)

dflist = []
for x in filelist:
    df = pd.read_json(x)
    dflist.append(df)
for x in range(len(dflist)):
    dflist[x]['id'] = idlist[x]
mood_ema = pd.concat(dflist)
mood_ema['mood'] = ""


# In[ ]:


#Re factoring and replacing nulls
def factor(c):
    if c['happyornot'] == 1:
        if c['happy'] == 1:
            return 5
        elif c['happy'] == 2:
            return 6
        elif c['happy'] == 3:
            return 7
        elif c['happy'] == 4:
            return 8
    elif c['sadornot'] == 1:
        if c['sad'] == 1:
            return 4
        elif c['sad'] == 2:
            return 3
        elif c['sad'] == 3:
            return 2
        elif c['sad'] == 4:
            return 1
    elif (c['happyornot'] == 0 and c['sadornot'] == 0):
        return np.nan


mood_ema['happyornot'] = mood_ema['happyornot'].replace('2', 2)
mood_ema['happyornot'] = mood_ema['happyornot'].replace('1', 1)
mood_ema['sadornot'] = mood_ema['sadornot'].replace('1', 1)
mood_ema['sadornot'] = mood_ema['sadornot'].replace('2', 2)
mood_ema['mood'] = mood_ema.apply(factor, axis = 1)


# In[ ]:


mood_response = mood_ema[['id', 'mood']]
mood_response = mood_response.groupby('id').mean()
mood_response.reset_index(inplace = True)
mood_response.columns = ('uid', 'mood')
mood_response.to_csv(main_path + '/dataset/tables/ema/Mood_EMA.csv', index = False)


# # Sleep EMA responses

# In[ ]:


#Loading all participants data onto one dataframe
os.chdir(main_path + '/dataset/EMA/response/Sleep')
filelist = []
for files in glob.glob("*.json"):
    filelist.append(files)
idlist = []
for i in filelist:
    x = (i.split("_")[1]).split(".")[0]
    idlist.append(x)
dflist = []
for x in filelist:
    df = pd.read_json(x)
    dflist.append(df)
for x in range(len(dflist)):
    dflist[x]['id'] = idlist[x]
sleep_ema = pd.concat(dflist)


# In[ ]:


#Refactoring
sleep_ema = sleep_ema[sleep_ema['resp_time'] > dateutil.parser.parse("2013-04-02")]
def rev_factor(c):
    if c['rate'] == 1:
        return 4
    elif c['rate'] == 2:
        return 3
    elif c['rate'] == 3:
        return 2
    elif c['rate'] == 4:
        return 1

def get_hours(c):
    if c['hour'] != np.nan:
        return c['hour']/2+2.5
        
sleep_ema['rate'] = sleep_ema.apply(rev_factor, axis = 1)
sleep_ema['hour'] = sleep_ema.apply(get_hours, axis = 1)

sleep_ema_summary = sleep_ema.groupby('id').mean()
sleep_ema_summary.reset_index(inplace = True)
sleep_ema_summary.columns = ('uid', 'sleep_hours', 'sleep_quality', 'social')
sleep_ema_summary.to_csv(main_path + '/dataset/tables/ema/SleepEMA.csv', index = False)


# # Socializing EMA responses

# In[ ]:


#Loading all participants data onto one dataframe
os.chdir(main_path + '/dataset/EMA/response/Social')
filelist = []
for files in glob.glob("*.json"):
    filelist.append(files)
idlist = []
for i in filelist:
    x = (i.split("_")[1]).split(".")[0]
    idlist.append(x)

dflist = []
for x in filelist:
    df = pd.read_json(x)
    dflist.append(df)
for x in range(len(dflist)):
    dflist[x]['id'] = idlist[x]
social_ema = pd.concat(dflist)

#Filter date
social_ema = social_ema[social_ema['resp_time'] > dateutil.parser.parse("2013-03-25")]
social_interactions_summary = social_ema[['id', 'number']]
social_interactions_summary = social_interactions_summary.groupby('id').mean()
social_interactions_summary.reset_index(inplace = True)
social_interactions_summary.columns = ('uid', 'people_interacted_with')
social_interactions_summary.to_csv(main_path + '/dataset/tables/ema/social_ema_response.csv', index = False)


# # Stress EMA responses

# In[ ]:


#Loading all participants data onto one dataframe
os.chdir(main_path + '/dataset/EMA/response/Stress')
filelist = []
for files in glob.glob("*.json"):
    filelist.append(files)
idlist = []
for i in filelist:
    x = (i.split("_")[1]).split(".")[0]
    idlist.append(x)

dflist = []
for x in filelist:
    df = pd.read_json(x)
    dflist.append(df)
for x in range(len(dflist)):
    dflist[x]['id'] = idlist[x]
stress_ema = pd.concat(dflist)


# In[ ]:


#Filter date
stress_ema = stress_ema[stress_ema['resp_time'] > dateutil.parser.parse("2013-03-26")]
#Relevel factors
def stress_relevel(c):
    if c['level'] == 1:
        return 12
    elif c['level'] == 2:
        return 11
    elif c['level'] == 3:
        return 10
    elif c['level'] == 4:
        return 13
    elif c['level'] == 5:
        return 14
def stress_renumber(c):
    if c['level'] == 10:
        return 5
    elif c['level'] == 11:
        return 4
    elif c['level'] == 12:
        return 3
    elif c['level'] == 13:
        return 2
    elif c['level'] == 14:
        return 1
stress_summary = stress_ema[['id','level']]
stress_summary = stress_summary[pd.notnull(stress_summary['level'])]

#need to relevel and renumber stress levels because the response and their corresponding
#number were not linear
stress_summary['level'] = stress_summary.apply(stress_relevel, axis =1)
stress_summary['level'] = stress_summary.apply(stress_renumber, axis =1)


# In[ ]:


stress_summary = stress_summary.groupby('id').mean()
stress_summary.reset_index(inplace = True)
stress_summary.columns = ('uid', 'stress_level')
stress_summary.to_csv(main_path + '/dataset/tables/ema/Stress_EMA.csv', index = False)


# # PHQ-9 Depression Survey

# In[ ]:


phq = pd.read_csv(main_path + '/dataset/survey/PHQ-9.csv', index_col = False)
phq.columns = ('uid','type','interest', 'depression', 'sleep','energy', 'appetite', 'self_image', 'concentration', 'manic_depressive', 'suicidal', 'response_difficulty')


# In[ ]:


#Turning factors in to scores
def factor_to_score_interest(c):
    if c['interest'] == 'Not at all':
        return 0
    elif c['interest'] == 'Several days':
        return 1
    elif c['interest'] == 'More than half the days':
        return 2
    elif c['interest'] == 'Nearly every day':
        return 3
def factor_to_score_depression(c):
    if c['depression'] == 'Not at all':
        return 0
    elif c['depression'] == 'Several days':
        return 1
    elif c['depression'] == 'More than half the days':
        return 2
    elif c['depression'] == 'Nearly every day':
        return 3
def factor_to_score_sleep(c):
    if c['sleep'] == 'Not at all':
        return 0
    elif c['sleep'] == 'Several days':
        return 1
    elif c['sleep'] == 'More than half the days':
        return 2
    elif c['sleep'] == 'Nearly every day':
        return 3
def factor_to_score_energy(c):
    if c['energy'] == 'Not at all':
        return 0
    elif c['energy'] == 'Several days':
        return 1
    elif c['energy'] == 'More than half the days':
        return 2
    elif c['energy'] == 'Nearly every day':
        return 3
def factor_to_score_appetite(c):
    if c['appetite'] == 'Not at all':
        return 0
    elif c['appetite'] == 'Several days':
        return 1
    elif c['appetite'] == 'More than half the days':
        return 2
    elif c['appetite'] == 'Nearly every day':
        return 3
def factor_to_score_self_image(c):
    if c['self_image'] == 'Not at all':
        return 0
    elif c['self_image'] == 'Several days':
        return 1
    elif c['self_image'] == 'More than half the days':
        return 2
    elif c['self_image'] == 'Nearly every day':
        return 3
def factor_to_score_concentration(c):
    if c['concentration'] == 'Not at all':
        return 0
    elif c['concentration'] == 'Several days':
        return 1
    elif c['concentration'] == 'More than half the days':
        return 2
    elif c['concentration'] == 'Nearly every day':
        return 3
def factor_to_score_manic_depressive(c):
    if c['manic_depressive'] == 'Not at all':
        return 0
    elif c['manic_depressive'] == 'Several days':
        return 1
    elif c['manic_depressive'] == 'More than half the days':
        return 2
    elif c['manic_depressive'] == 'Nearly every day':
        return 3
def factor_to_score_suicidal(c):
    if c['suicidal'] == 'Not at all':
        return 0
    elif c['suicidal'] == 'Several days':
        return 1
    elif c['suicidal'] == 'More than half the days':
        return 2
    elif c['suicidal'] == 'Nearly every day':
        return 3

#Calculate overall score
phq['interest_score'] = phq.apply(factor_to_score_interest, axis = 1)
phq['depression_score'] = phq.apply(factor_to_score_depression, axis = 1)
phq['sleep_score'] = phq.apply(factor_to_score_sleep, axis = 1)
phq['energy_score'] = phq.apply(factor_to_score_energy, axis = 1)
phq['appetite_score'] = phq.apply(factor_to_score_appetite, axis = 1)
phq['self_image_score'] = phq.apply(factor_to_score_self_image, axis = 1)
phq['concentration_score'] = phq.apply(factor_to_score_concentration, axis = 1)
phq['manic_depressive_score'] = phq.apply(factor_to_score_manic_depressive, axis = 1)
phq['suicidal_score'] = phq.apply(factor_to_score_suicidal, axis = 1)


# In[ ]:


#split data into pre- and post- study dataframes
phq_pre = phq.loc[phq.type == 'pre']
phq_post = phq.loc[phq.type == 'post']

phq_pre.to_csv(main_path + '/dataset/tables/survey/phq_pre.csv', index = False)
phq_post.to_csv(main_path + '/dataset/tables/survey/phq_post.csv', index = False)


# # Perceived Stress Survey

# In[ ]:


pss = pd.read_csv(main_path + '/dataset/survey/PerceivedStressScale.csv', index_col = False)
pss.columns = ('uid','type','upset', 'not_control', 'nervous','confident', 'going_your_way', 'not_cope', 'control', 'in_control', 'out_of_control', 'too_much')


# In[ ]:


#Turning factors into scores
#for positive question, lower score means feel statement more often 
def pos_pss(c,x):
    if c[x] == 'Very Often':
        return 0
    elif c[x] == 'Fairly Often':
        return 1
    elif c[x] == 'Sometime':
        return 2
    elif c[x] == 'Almost Never':
        return 3
    
#for negative question, higher score means feel statement more often    
def neg_pss(c,x):
    if c[x] == 'Very Often':
        return 3
    elif c[x] == 'Fairly Often':
        return 2
    elif c[x] == 'Sometime':
        return 1
    elif c[x] == 'Almost Never':
        return 0

#grouped questions into positive or negative questions 
#positive question example: 'In the last month, how often have you felt that things were going your way?'
#negative question example: ' In the last month, how often have you felt that you were unable to control the important things in your life?'
pos_q = ['confident','going_your_way','control','in_control']
neg_q = ['upset', 'not_control', 'nervous', 'not_cope', 'out_of_control', 'too_much']

for x in pos_q:
    pss[x] = pss.apply(lambda y: pos_pss(y,x), axis = 1)

for x in neg_q:
    pss[x] = pss.apply(lambda y: neg_pss(y,x), axis = 1)


# In[ ]:


#took sum of all scores for each question to get overall loneliness score
pss['perceived_stress'] = pss.sum(axis = 1)
pss_total = pss[['uid','type','perceived_stress']]
pss_pre = pss_total.loc[pss_total.type == 'pre']
pss_post = pss_total.loc[pss_total.type == 'post']
pss_pre.to_csv(main_path + '/dataset/tables/survey/pss_pre.csv', index = False)
pss_post.to_csv(main_path + '/dataset/tables/survey/pss_post.csv', index = False)


# # Loneliness Survey

# In[ ]:


lonely = pd.read_csv('E:/dataset/survey/LonelinessScale.csv', index_col = False)
lonely.columns = ('uid','type','pos1', 'neg1', 'neg2','pos2', 'pos3', 'pos4', 'neg3', 'neg4', 'outgoing', 'pos5', 'neg5', 'neg6', 'neg7', 'neg8', 'pos6', 'pos7', 'neg9', 'neg10', 'pos8', 'pos9')


# In[ ]:


#Turning factors into scores
#for positive question, lower score means feel statement more often    
def pos_lonely(c,x):
    if c[x] == 'Often':
        return 0
    elif c[x] == 'Sometimes':
        return 1
    elif c[x] == 'Rarely':
        return 2
    elif c[x] == 'Never':
        return 3

#for negative question, higher score means feel statement more often    
def neg_lonely(c,x):
    if c[x] == 'Often':
        return 3
    elif c[x] == 'Sometimes':
        return 2
    elif c[x] == 'Rarely':
        return 1
    elif c[x] == 'Never':
        return 0
    
#grouped questions into positive or negative questions 
#positive question example: 'I feel part of a group of friends'
#negative question example: 'There is no one I can turn to'
pos_q = ['pos1','pos2','pos3','pos4','pos5','pos6','pos7','pos8','pos9']
neg_q = ['neg1', 'neg2', 'neg3', 'neg4', 'neg5', 'neg6', 'neg7', 'neg8', 'neg9', 'neg10']


for x in pos_q:
    lonely[x] = lonely.apply(lambda y: pos_lonely(y,x), axis = 1)

for x in neg_q:
    lonely[x] = lonely.apply(lambda y: neg_lonely(y,x), axis = 1)


# In[ ]:


#took sum of all scores for each question to get overall loneliness score
lonely['loneliness'] = lonely.sum(axis = 1)
lonely_total = lonely[['uid','type','loneliness']]
lonely_pre = lonely_total.loc[lonely_total.type == 'pre']
lonely_post = lonely_total.loc[lonely_total.type == 'post']
lonely_pre.to_csv(main_path + '/dataset/tables/survey/loneliness_pre.csv', index = False)
lonely_post.to_csv(main_path + '/dataset/tables/survey/loneliness_post.csv', index = False)


# # Grades

# In[ ]:


os.chdir(main_path + '/dataset/education')
grades = pd.read_csv('grades.csv', index_col = None, header = 0)
grades.columns = ('uid', 'overall_gpa', 'spring_gpa', 'class_gpa')
grades.to_csv(main_path + '/dataset/tables/ema/grades.csv', index = False)


# # Merging Tables

# In[ ]:


#merging EMA tables together
ema = dict((i,pd.read_csv(ema_paths[i])) for i in ema_paths)
EMA = ema.values()
EMA = [x.set_index('uid') for x in EMA]
k_ema = ema.keys()
merged_data = pd.concat(EMA,axis = 1, join = 'outer', sort = True, keys = k_ema)
merged_data.to_csv(main_path + '/dataset/tables/emaData.csv', index = True)


# # Sensor Data Processing

# ## Activity Data

# In[ ]:


def activity():
    pd.options.mode.chained_assignment = None
    os.chdir(main_path + '/dataset/sensing/activity')
    #place all csv files and their corresponding ids in a list
    filelist = []
    for files in glob.glob("*.csv"):
        filelist.append(files)
    idlist = []
    for i in filelist:
        x = (i.split("_")[1]).split(".")[0]
        idlist.append(x)
    #read each csv into a pandas dataframe and add the id as a column
    dflist = []
    for x in range(len(filelist)):
        iter_csv = pd.read_csv(filelist[x], index_col=None, header = 0, iterator = True, chunksize = 1000)
        df = pd.concat([chunk[chunk.iloc[:,1] < 3] for chunk in iter_csv])
        df['uid'] = idlist[x]
        df['timestamp'] = df['timestamp'] - 14400
        df['time'] =  pd.to_datetime(df['timestamp'], unit = 's')
        df['date'] = pd.DatetimeIndex(df['time']).date
        dflist.append(df)
    #merge the dataframes into the output dataframe
    activity = pd.concat(dflist)
    del dflist
    gc.collect()
    activity.to_csv(main_path + '/dataset/tables/sensors/activity.csv', index = False)
    del activity
    gc.collect()


# ## Audio Data

# In[ ]:


def audio():
    os.chdir(main_path + '/dataset/sensing/audio')
    filelist = []
    for files in glob.glob("*.csv"):
        filelist.append(files)
    idlist = []
    for i in filelist:
        x = (i.split("_")[1]).split(".")[0]
        idlist.append(x)
    dflist = []
    for x in range(len(filelist)):
        iter_csv = pd.read_csv(filelist[x], index_col=None, header = 0, iterator = True, chunksize = 1000)
        df = pd.concat([chunk[chunk.iloc[:,1] < 3] for chunk in iter_csv])
        df['uid'] = idlist[x]
        df['timestamp'] = df['timestamp'] - 14400
        df['time'] =  pd.to_datetime(df['timestamp'], unit = 's')
        df['date'] = pd.DatetimeIndex(df['time']).date
        dflist.append(df)
    audio = pd.concat(dflist)
    del dflist
    gc.collect()
    audio.to_csv(main_path + '/dataset/tables/sensors/audio.csv', index = False)
    del audio
    gc.collect()


# ## Bluetooth Data

# In[ ]:


def bluetooth():
    pd.options.mode.chained_assignment = None
    os.chdir(main_path + '/dataset/sensing/bluetooth')
    filelist = []
    #place all csv files and their corresponding ids in a list
    for files in glob.glob("*.csv"):
        filelist.append(files)
        idlist = []
    for i in filelist:
        x = (i.split("_")[1]).split(".")[0]
        idlist.append(x)
    dflist = []
    for x in filelist:
        df = pd.read_csv(x, index_col=None, header = 0)
        dflist.append(df)
    for x in range(len(dflist)):
        dflist[x]['uid'] = idlist[x]
    bluetooth = pd.concat(dflist)
    bluetooth.to_csv(main_path + '/dataset/tables/sensors/bluetooth.csv', index = False)


# ## Conversation Data

# In[ ]:


def conversation():
    os.chdir(main_path + '/dataset/sensing/conversation')
    filelist = []
    for files in glob.glob("*.csv"):
        filelist.append(files)
        idlist = []
    for i in filelist:
        x = (i.split("_")[1]).split(".")[0]
        idlist.append(x)
    dflist = []
    for x in filelist:
        df = pd.read_csv(x, index_col=None, header = 0)
        dflist.append(df)
    for x in range(len(dflist)):
        dflist[x]['uid'] = idlist[x]
    conversation = pd.concat(dflist)
    conversation['start'] = pd.to_datetime(conversation['start_timestamp'], unit = 's')
    conversation['end'] = pd.to_datetime(conversation[' end_timestamp'], unit = 's')
    conversation['duration'] = (conversation['end']- conversation['start'])/np.timedelta64(1,'s')
    conversation['date'] = pd.DatetimeIndex(conversation['start']).date
    conversation['start_hour'] = pd.DatetimeIndex(conversation['start']).hour
    conversation.to_csv(main_path + '/dataset/tables/sensors/conversation.csv', index =False)

    #Filter conversation to 'daytime' epoch
    day_talk = conversation.loc[((conversation.start_hour >= 9) | (conversation.start_hour < 18))]
    day_talk.to_csv(main_path + '/dataset/tables/sensors/day_talk.csv', index = False)


# ## Dark Data

# In[ ]:


def dark():
    os.chdir(main_path + '/dataset/sensing/dark')
    filelist = []
    for files in glob.glob("*.csv"):
        filelist.append(files)
    idlist = []
    for i in filelist:
        x = (i.split("_")[1]).split(".")[0]
        idlist.append(x)
    
    dflist = []
    for x in filelist:
        df = pd.read_csv(x, index_col=None, header = 0)
        dflist.append(df)
    for x in range(len(dflist)):
        dflist[x]['uid'] = idlist[x]
    dark = pd.concat(dflist)
    dark['start'] = dark['start'] - 14400
    dark['end'] = dark['end'] - 14400
    dark['start_time'] = pd.to_datetime(dark['start'], unit = 's')
    dark['end_time'] = pd.to_datetime(dark['end'], unit = 's')
    
    #Calculate duration of sleep
    dark['duration'] = (dark['end_time'] - dark['start_time'])/np.timedelta64(1,'m')
    dark['start_hour'] = pd.DatetimeIndex(dark['start_time']).hour
    dark['date'] = pd.DatetimeIndex(dark['start_time']).date
    dark.to_csv(main_path + '/dataset/tables/sensors/dark.csv', index = False)
    
    #Filter dark to 'nighttime' epoch and filter for events longer than 180 min (sleeping)
    bedtime = dark.loc[((dark.start_hour >= 20) | (dark.start_hour < 6)) & (dark.duration >= 180.0)]
    
    #Create a factor variable based on when students go to bed
    def early_to_bed(c):
        if c['start_hour'] == 20:
            return 1
        elif c['start_hour'] == 21:
            return 2
        elif c['start_hour'] == 22:
            return 3
        elif c['start_hour'] == 23:
            return 4
        elif c['start_hour'] == 24:
            return 5
        elif c['start_hour'] == 0:
            return 6
        elif c['start_hour'] == 1:
            return 7
        elif c['start_hour'] == 2:
            return 8
        elif c['start_hour'] == 3:
            return 9
        elif c['start_hour'] == 4:
            return 10
        elif c['start_hour'] == 5:
            return 11
    bedtime['bedtime_early'] = bedtime.apply(early_to_bed, axis = 1)
    bedtime = bedtime.rename(columns = {'duration': 'night_duration'})
    bedtime.to_csv(main_path + '/dataset/tables/sensors/bedtime.csv')


# ## GPS Data

# In[ ]:


def gps():
    os.chdir(main_path + '/dataset/sensing/gps')
    filelist = []
    for files in glob.glob("*.csv"):
        filelist.append(files)
    idlist = []
    for i in filelist:
        x = (i.split("_")[1]).split(".")[0]
        idlist.append(x)
    dflist = []
    for x in filelist:
        df = pd.read_csv(x, index_col=None, header = 0)
        dflist.append(df)
    for x in range(len(dflist)):
        dflist[x]['id'] = idlist[x]
    gps = pd.concat(dflist)
    gps.reset_index(inplace = True)
    gps.columns = ('timestamp', 'provider', 'network_type', 'accuracy', 'lat',
    'lon', 'altitude', 'bearing' ,'speed', 'travelstate', 'null', 'uid')
    gps = gps.drop("null", 1)
    gps.to_csv(main_path + '/dataset/tables/sensors/gps.csv', index = False)
    del gps


# ## Phone Charge Data

# In[ ]:


def phone_charge():
    os.chdir(main_path + '/dataset/sensing/phonecharge')
    filelist = []
    for files in glob.glob("*.csv"):
        filelist.append(files)
    idlist = []
    for i in filelist:
        x = (i.split("_")[1]).split(".")[0]
        idlist.append(x)
    dflist = []
    for x in filelist:
        df = pd.read_csv(x, index_col=None, header = 0)
        dflist.append(df)
    for x in range(len(dflist)):
        dflist[x]['uid'] = idlist[x]
    phonecharge = pd.concat(dflist)
    phonecharge['start'] = phonecharge['start'] - 14400
    phonecharge['end'] = phonecharge['end'] - 14400
    phonecharge['start_time'] = pd.to_datetime(phonecharge['start'], unit = 's')
    phonecharge['end_time'] = pd.to_datetime(phonecharge['end'], unit = 's')
    phonecharge['duration'] = (phonecharge['end_time'] - phonecharge['start_time'])/np.timedelta64(1,'s')
    phonecharge['date'] = pd.DatetimeIndex(phonecharge['start_time']).date
    phonecharge.to_csv(main_path + '/dataset/tables/sensors/phonecharge.csv', index = False)


# ## Phone Lock Data

# In[ ]:


def phonelock():
    os.chdir(main_path + '/dataset/sensing/phonelock')
    filelist = []
    for files in glob.glob("*.csv"):
        filelist.append(files)
    idlist = []
    for i in filelist:
        x = (i.split("_")[1]).split(".")[0]
        idlist.append(x)
    dflist = []
    for x in filelist:
        df = pd.read_csv(x, index_col=None, header = 0)
        dflist.append(df)
    for x in range(len(dflist)):
        dflist[x]['uid'] = idlist[x]
    phonelock = pd.concat(dflist)
    phonelock['start'] = phonelock['start'] - 14400
    phonelock['end'] = phonelock['end'] - 14400
    phonelock['start_time'] = pd.to_datetime(phonelock["start"], unit = 's')
    phonelock['end_time'] = pd.to_datetime(phonelock["end"], unit = 's')
    phonelock['duration'] = (phonelock['end_time'] - phonelock.start_time)/np.timedelta64(1,'s')
    phonelock['date'] = pd.DatetimeIndex(phonelock['start_time']).date
    phonelock.to_csv(main_path + '/dataset/tables/sensors/phonelock.csv', index = False)


# ## WiFi Data

# In[ ]:


def wifi():
    os.chdir(main_path + '/dataset/sensing/wifi')
    filelist = []
    for files in glob.glob("*.csv"):
        filelist.append(files)
    idlist = []
    for i in filelist:
        x = (i.split("_")[1]).split(".")[0]
        idlist.append(x)
    dflist = []
    for x in range(len(filelist)):
        iter_csv = pd.read_csv(filelist[x], index_col=None, header = 0, iterator = True, chunksize = 1000)
        df = pd.concat([chunk for chunk in iter_csv])
        df['uid'] = idlist[x]
        dflist.append(df)
    wifi = pd.concat(dflist)
    wifi.to_csv(main_path + '/dataset/tables/sensors/wifi.csv', index = False)
    del wifi


# ## WiFi Location and Study Events Data

# In[5]:


def wifi_location():
    os.chdir(main_path + '/dataset/sensing/wifi_location')
    filelist = []
    for files in glob.glob("*.csv"):
        filelist.append(files)
    idlist = []
    for i in filelist:
        x = (i.split("_")[2]).split(".")[0]
        idlist.append(x)
    dflist = []
    for x in filelist:
        df = pd.read_csv(x, index_col=None, header = 0)
        dflist.append(df)
    for x in range(len(dflist)):
        dflist[x]['uid'] = idlist[x]
    wifi_location = pd.concat(dflist)
    wifi_location.reset_index(inplace = True)
    wifi_location.columns = ("timestamp", "location", "null", "uid")
    wifi_location = wifi_location.drop("null", 1)
    wifi_location['timestamp'] = wifi_location['timestamp'] - 14400
    wifi_location['time'] = pd.to_datetime(wifi_location['timestamp'], unit = 's')
    wifi_location.to_csv(main_path + '/dataset/tables/sensors/wifi_location.csv', index = False)
    
    study_locs = ['in[baker-berry]', 'in[dana-library]', 'in[feldberg_library]',
    'in[sanborn]', 'in[dartmouth_hall]', 'in[silsby-rocky]']
    df_study = wifi_location.loc[wifi_location['location'].isin(study_locs)]
    #del wifi_location
    #Calculate time spent continuously in study location
    df_study['delta'] = (df_study['time']-df_study['time'].shift())
    #df_study['delta'] = df_study['delta'].replace(to_replace = np.nan, value = int(0))
    def study_events(c):
        global i
        if c['location'] == c['shift'] and c['delta'] < datetime.timedelta(minutes= 20):
            try:
                return i
            except:
                i = 1
                return i
        else:
            try:
                i += 1
                return i
            except:
                i = 1
                return i
    df_study['shift'] = df_study['location'].shift().fillna(df_study['location'])
    df_study['study_event'] = df_study.apply(study_events, axis= 1)
    #filter any weird values
    def event_delta(c):
        if c['delta'] < datetime.timedelta(minutes = 20):
            return c['delta']
        else:
            return datetime.timedelta(seconds = 0)
    df_study['event_delta'] = df_study.apply(event_delta, axis = 1)
    df_study['event_delta'] = (df_study['event_delta']/np.timedelta64(1, 'm'))
    df_study['date'] = pd.DatetimeIndex(df_study['time']).date

    dropcols = ['timestamp','time', 'delta', 'shift']
    cols = [c for c in df_study.columns.tolist() if c not in dropcols]
    df_study_events = df_study[cols]
    df_study_events = df_study_events.groupby(['study_event', 'date', 'uid']).sum().reset_index()
    df_study_events["event_start"] = np.nan
    df_study_events["event_end"] = np.nan
    
    #Get the start time and end time of each study session
    for i in range(1,len(df_study_events['study_event'])):
        df_study_events['event_start'][i-1] = df_study.loc[df_study['study_event'] == i]['time'].min()
        df_study_events['event_end'][i-1] = df_study.loc[df_study['study_event'] == i]['time'].max()
    df_study_events['event_delta'] = np.round(df_study_events['event_delta'], 0)
    
    #Filter for study events lasting 20 or more minuets
    df_study_events = df_study_events.loc[df_study_events['event_delta'] >= 20]
    df_study_events.to_csv(main_path + '/dataset/tables/sensors/study_events.csv', index = False)
    del df_study_events


# ## Run All Sensor Data

# In[ ]:


#Uncomment below to run

print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
activity()
print('activity done')
print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
audio()
print('audio done')
print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
bluetooth()
print('bluetooth done')
print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
conversation()
print('conversation done')
print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
dark()
print('dark done')
print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
gps()
print('gps done')
print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
phone_charge()
print('phone_charge done')
print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
phonelock()
print('phonelock done')
print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
wifi()
print('wifi done')
print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
wifi_location()
print('wifi_location done')
print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))


# In[ ]:





# In[ ]:




