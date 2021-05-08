#!/usr/bin/env python
# coding: utf-8

# Importing the prerequisities

# In[802]:


import pandas as pd
import os
import numpy as np
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
import re
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import tweepy as tw
import pandas as pd
import yaml
from tqdm import tqdm
from datetime import datetime


# Declare the input values

# In[806]:


class Params:
    
    #service = "Plasma Donors"
    City = "Ghaziabad"
    #search_words =  ["verified+noida+Plasma+-'needed'+-'need'+-'needs'+-'required'+-'require'+-'requires'+-'requirement'+-'requirements'+-'rqrd'"] #[#Remdesivir available ", "#Plasma available Bangalore", "#Oxygen available Bangalore"]
    search_words = ["verified+" + City + "+bed+OR+beds+OR+icu+OR+oxygen+OR+ventilator+OR+ventilators+OR+remdesivir+OR+favipiravir+OR+tocilizumab+OR+plasma+OR+tiffin+OR+food+OR+ambulance+-'not+verified'+-'unverified'+-'needed'+-'need'+-'needs'+-'required'+-'require'+-'requires'+-'requirement'+-'requirements'+-'rqrd'"]
    #search_words = ["verified+" + City + "+plasma+OR+oxygen+OR+concentrator+OR+oxygen can+OR+Oxygen Refill+-'not+verified'+-'unverified'+-'needed'+-'need'+-'needs'+-'required'+-'require'+-'requires'+-'requirement'+-'requirements'"]
    date_since = "2021-05-08"
    date_until = "2021-05-09"
    time_now = datetime.strptime(str(datetime.now().strftime("%H:%M:%S")), '%H:%M:%S')
    total_tweets = 100
    twitter_access_token=""
    twitter_access_token_secret=""
    twitter_consumer_key=""
    twitter_consumer_secret=""
    dir_path = "" #'/Users/abc/Downloads/Stanford_NER/'
    file_path = 'stanford-ner-2020-11-17/classifiers/english.conll.4class.distsim.crf.ser.gz'
    jar_file_path = 'stanford-ner-2020-11-17/stanford-ner.jar'
    


# If reuqired User could update the below list for Citie and run the below cell

# In[782]:


Cities = ["""Vizianagaram -> AP -> Vizianagaram
Vijayawada -> AP -> Vijayawada
Visakhapatnam -> AP -> Visakhapatnam
Jaipur -> RJ -> Jaipur
Tirupathi -> AP -> Tirupati
Tirunalveli -> TN -> Tirunelveli
Srinagar -> JK -> Srinagar
Rajkot -> GJ -> Rajkot
Rajahmundry -> AP -> Rajahmundry
Patna -> BR -> Patna
Ongole -> AP -> Ongole
Nellore -> AP -> Nellore
Nagpur -> MH -> Nagpur
Mysore -> KA -> Mysore
Meerut -> UP -> Meerut
kanpur -> UP -> Kanpur
Jamshedpur -> WB -> Jamshedpur
Faridabad -> UP -> Faridabad
Guntur -> AP -> Guntur
Hyderabad -> AP -> Hyderabad
Coimbatore -> TN -> Coimbatore
Chennai -> TN -> Chennai
Bongaigaon -> AS -> Bongaigaon
aligarh -> UP -> Aligarh
Agra -> UP -> Agra
Ghaziabad -> UP -> Ghaziabad
Badaun -> UP -> Badaun
Khorda -> OR -> Khorda
Barabanki -> UP -> Barabanki
Mhow -> MP -> Indore
South Tukoganj -> MP -> Indore
AB Road -> MP -> Indore
Lucknow -> UP -> Lucknow
Mumbai -> MH -> Mumbai
Pune -> MH -> Pune
Jalpaiguri -> WB -> Jalpaiguri
Palakkad -> KL -> Palakkad
Kannur -> KL -> Kannur
Kochi -> KL -> Kochi
Indore -> MP -> Indore
Ahmedabad -> GJ -> Ahmedabad
Bengaluru -> KA -> Bengaluru
Varanasi -> UP -> Varanasi
Kolkata -> WB -> Kolkata
Delhi -> DL -> Delhi
Bangalore -> KA -> Bangalore
Bengaluru -> KA -> Bengaluru
Noida -> UP -> Noida
Gurugram -> HR -> Gurugram
Gurgaon -> HR -> Gurgaon"""]


spl = Cities[0].split('\n')
new_loc = []
for i in spl:
    
    new_loc.append(list(set(i.split(' '))))

#new_loc = []
new_loc = [j for i in new_loc for j in i]
#new_loc = [", ".join(i) for i in new_loc]
new_loc = list(set(new_loc))
new_loc.remove('->')


# In[783]:


Cities_State_mapping = {'Aligarh': "Uttar Pradesh",
 'Bongaigaon': "Assam",
 'Srinagar': "Jammu & Kashmir",
 'Palakkad' : "Kerela",
 'Guntur' : "Andhra Pradesh",
 'Indore' : "Madhya Pradesh",
 'Coimbatore' : "Tamil Nadu",
 'Meerut' : "Uttar Pradesh",
 'Jamshedpur' : "Jharkhand",
 'Khorda' : "Orrisa",
 'Tirunalveli' : "Tamil Nadu",
 'Barabanki' : "Uttar Pradesh",
 'Noida': "Uttar Pradesh",
 'Chennai' :"Tamil Nadu" ,
 'Mysore' : "Karnataka",
 'Badaun' : "Uttar Pradesh",
 'Nellore': "Andhra Pradesh",
 'Gurgaon': "Haryana",
 'Gurugram': "Haryana",
 'Hyderabad': "Telangana",
 'Delhi' : "Delhi",
 'Nagpur' : "Maharashtra",
 'Ongole': "Andhra Pradesh",
 'Kanpur' : "Uttar Pradesh",
 'Mumbai': "Maharashtra",
 'Kochi': "Kerela",
 'Ahmedabad': "Gujrat",
 'Faridabad': "Uttar Pradesh",
 'Bengaluru': "Karnataka",
 'Varanasi': "Uttar Pradesh",
 'Tirupati': "Andhra Pradesh",
 'Rajahmundry': "Andhra Pradesh",
 'Rajkot': "Gujrat",
 'Kannur': "kerela",
 'Patna': "Bihar",
 'Ghaziabad' : "Uttar Pradesh",
 'Jalpaiguri': "West Bengal",
 'Agra': "Uttar Pradesh",
 'Visakhapatnam': "Andhra Pradesh",
 'Vizianagaram': "Andhra Pradesh",
 'Lucknow': "Uttar Pradesh",
 'Vijayawada': "Andhra Pradesh",
 'Bangalore': "Karnataka",
 'Pune' : "Maharashtra",
'Kolkata' : "West Bengal",
'Jaipur': "Rajasthan"
}


# If reuqired User could update the below list for categories in the same format and run the below cell

# In[784]:


services = { "Plasma Donors": ["A-", "A+", "AB-", "AB+","O-","O+", "B+", "B-"],
           "Hospital Beds" : ["Ventilator", "Isolation Bed", "Covid Bed", "Oxygen Bed", "ICU Bed"],
           "Oxygen Suppliers" : ["Oxygen Cylinder Refill", "Oxygen Cylinder", "Oxygen Concentrator", "Portable Oxygen Can", "Oxygen Cylinder Equipment"],
            "Testing Centres" : ["RT-PCR", "TrueNat", "CBNAAT", "Rapid Antigen"],
           "Ambulance" : ["With Oxygen", "Without Oxygen"],
           "Food" : ["Lunch", "Breakfast", "Dinner"],
           "Medicine" : ["Medrol", "Remdesivir", "Remdisivir", "Doxycycline", "Dexamethasone", "Oximeter", "Ivermectin", "Favipiravir", "Tocilizumab"],
            "Remdisivir": ["Remdisivir"]
}


# In[785]:


categories = ["""Plasma Donors => A-
        Oxygen Suppliers => Oxygen Cylinder Refill
        Medicine => Medrol
        Hospital Beds => Ventilator
        Hospital Beds => Isolation Bed
        Oxygen Suppliers => Oxygen Cylinder
        Medicine => Remdesivir
        Medicine => Remdisivir
        Hospital Beds => Covid Bed
        Hospital Beds => Oxygen Bed
        Medicine => Doxycycline
        Testing centers => RT-PCR
        Medicine => Dexamethasone
        Hospital Beds => ICU Bed
        Testing centers => Rapid Antigen
        Medicine => Oximeter
        Ambulance => With Oxygen
        Plasma Donors => AB-
        Plasma Donors => O-
        Food => Dinner
        Plasma Donors => B+
        Food => Lunch
        Oxygen Suppliers => Oxygen Concentrator
        Oxygen Suppliers => Portable Oxygen Can
        Plasma Donors => A+
        Medicine => Ivermectin
        Oxygen Suppliers => Oxygen Cylinder Equipment
        Testing centers => TrueNat
        Plasma Donors => O+
        Medicine => Favipiravir
        Food => Breakfast
        Plasma Donors => AB+
        Medicine => Tocilizumab
        Testing centers => CBNAAT
        Plasma Donors => B-
        Ambulance => Without Oxygen"""]

spl_c = categories[0].split('\n')
new_cat = []
for i in spl_c:
    
    new_cat.append(list(set(i.split(' '))))

#new_loc = []
new_cat = [j for i in new_cat for j in i]
#new_loc = [", ".join(i) for i in new_loc]
new_cat = list(set(new_cat))


# Function to fetch tweets based on the params defined by user

# In[808]:


access_token= Params.twitter_access_token
access_token_secret= Params.twitter_access_token_secret
consumer_key= Params.twitter_consumer_key
consumer_secret= Params.twitter_consumer_secret

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

def get_tweets(search_words, date_since, date_until, size = 5):
    # Define the search term and the date_since date as variables
    
    new_search = search_words + " -filter:retweets"

    tweets = tw.Cursor(api.search,
                q=new_search,
                lang="en",
                since=date_since,until=date_until, tweet_mode='extended').items(size)

    #tweets = api.search(q=search_words, count=size, tweet_mode='extended')

    # Iterate and print tweets
    res = []
    time_created = []
    for tweet in tweets:
        if not tweet.retweeted:
            res.append(tweet.full_text)
            time_created.append(tweet.created_at)
    
    return res, time_created


# Converting tweets to text format

# In[809]:


def get_tweet_data(search_words):
    text_dump = [] #OXYGEN REFILL - Lucknow. Register at \n \n\nCylinders to be refilled as per oxygen level.…'
    for i in search_words:
        
        print(" ==== TWEET EXTRACTION IN PROCESS ===")
        tweet_dump, time_created = get_tweets(search_words = i, 
                                                date_since = Params.date_since, date_until = Params.date_until, size = Params.total_tweets)
        # text = 'New Delhi is having a lot of cases. Help needed.'


        print(" ==== TWEETS CLEANING IN PROCESS ===")
         
        for i in tweet_dump:
            #print(i)
            #print("===============")
            text_dump.append(i)
    
    return text_dump, time_created


# Creating instance of POS tagger

# In[790]:


#the below mentioned paths should be based on user's local path variable.
#in order to run the below code, download the classifier from the folloring location:
#https://nlp.stanford.edu/software/CRF-NER.html#Download
#under the "Downloads" section with file name "Download Stanford Named Entity Recognizer version 4.2.0"

st = StanfordNERTagger(Params.dir_path + Params.file_path,
					   Params.dir_path + Params.jar_file_path,
					   encoding='utf-8')

def get_ner(text):
    tokenized_text = word_tokenize(text)
    tokens_without_sw = [word for word in tokenized_text if not word in stopwords.words()]
    classified_text = st.tag(tokens_without_sw)
    return classified_text


# Function Generating the dataset

# In[791]:


def generate_dataset(text_dump, city_1):
    idx = 0
    data = pd.DataFrame()
    for text in tqdm(text_dump, desc = "progress of tweets churned"):
        text = " ".join(filter(lambda x:x[0]!='@', text.split()))
        
        loc = []
        num = []
        org = []
        cat = []
        other = []
        sub_ser = []
        address = []
        person = []
        
        description = " ".join(filter(lambda x:x[0]!='@' and x[0]!='#' and x[-1]!=':', text.split()))
        description = re.sub(r"\d{10}|\d{5}[-\.\s]??\d{5}", "", description)
        
        classified_text = get_ner(text)
        services_keys = list(services.keys())
        #print(classified_text)
#         tt = text.replace("\n", " ")
#         tt = tt.split(" ")
        
#         sub_ser = [sub for sub in services[service] if sub in tt]
#         if len(sub_ser) == 0:
#             sub_ser = services[service]
        
        loc = [str(i[0]).capitalize() for i in classified_text if i[1] == 'LOCATION']
        org = [str(i[0]).capitalize() for i in classified_text if i[1] == 'ORGANIZATION']
        other = [str(i[0]).capitalize() for i in classified_text if i[1] == 'O']
        num = re.findall(r"\d{10}|\d{5}[-\.\s]??\d{5}", text)
        person = [str(i[0]).capitalize() for i in classified_text if i[1] == 'PERSON']
        #print(person)
        #####Removing keywords that do not belong under organization
        org_remove = []
        
        
        #print(org)
        for i in org:
            #print(i)
            if i.lower() in map(str.lower,new_loc):
                loc.append(i)
                org_remove.append(i)
            elif i.lower() in map(str.lower,new_cat):
                cat.append(i)
                org_remove.append(i)


        if  len(org_remove)>0:
            for j in org_remove:
                org.remove(j)


         #####Removing keywords that do not belong under Location
        loc_remove = []
        for i in loc:
            if i.lower() in map(str.lower, new_cat):
                cat.append(i)
                loc_remove.append(i)


        if  len(loc_remove)>0:
            for j in loc_remove:
                loc.remove(j)


         ####cleaning others

        for k in other:
            #print(i)
            if k.lower() in map(str.lower, new_loc):
                loc.append(k)
            elif k.lower() in map(str.lower, new_cat):
                cat.append(k)
                
        for m in person:
            #print(i)
            if m.lower() in map(str.lower, new_loc):
                loc.append(m)
            elif m.lower() in map(str.lower, new_cat):
                cat.append(m)

        city = [i for i in loc if i in new_loc]
        address = [i for i in loc if i not in city]       
        ser = [i for i in list(services.keys()) if len(set(i.split(" ")) & set(cat)) > 0]
        city = [i for i in city if i.capitalize() == city_1]
        
        state = [Cities_State_mapping[i] for i in city if i in list(Cities_State_mapping.keys())]
        
        num = list(set(num))
        num = [i.replace(" ", "") for i in list(set(num))]
        org = " ".join(org)
        len_org = len(org)
        #print(num)
        for i in ser:
            
            sub_ser = [k for k in services[i] if len(set(map(str.lower,k.split(" "))) & set(map(str.lower,cat))) > 0]
            sub_ser = sub_ser if len(sub_ser) == 1 else ""
            data.loc[idx+1, "{name}"] = i if len_org ==0 else org
            data.loc[idx+1, "{service_name}"] = "Remdesivir" if i == "Remdisivir" else i                                #" ".join(list(set(cat)))
            data.loc[idx+1, "{sub_service_name}"] = ", ".join(list(set(sub_ser)))
            data.loc[idx+1, "{state}"] = " ".join(list(set(state)))
            data.loc[idx+1, "{city}"] = " ".join(list(set(city)))
            data.loc[idx+1, "{area}"] = " ".join(list(set(city)))
            data.loc[idx+1, "{mobile}"] = ", ".join(list(num))
            data.loc[idx+1, "{address}"] = " ".join(list(address))
            data.loc[idx+1, "{description}"] = description
            data.loc[idx+1, "{pincode}"] = ""
            data.loc[idx+1, "{gmb_gmc}"] = ""
            data.loc[idx+1, "{verification_status}"] = "verified"
            idx += 1
   
    return data
        


# Run the function to generate the dataset and save in csv file

# In[810]:


text_dump, time_created = get_tweet_data(Params.search_words)
len(text_dump)


# In[811]:


# In order to get tweets from previous 5 hours run the below code and then generate the dataset
def check_for_frequent_running(text_dump,time_created, time_now,freq_ind = False, recency_hour = 0):
    if freq_ind:
        time_created1 = [datetime.strptime(str(i.time()), '%H:%M:%S') for i in time_created]
        idx_recency = [i for i, n in enumerate(time_created1) if (time_now-n).seconds/60/60 < recency_hour]
        text_dump1 =  [text_dump[i] for i in idx_recency]
    else:
        text_dump1 =  text_dump
    
    return text_dump1
        


# In[817]:


text_dump_new = check_for_frequent_running(text_dump,time_created, Params.time_now,freq_ind = True, recency_hour = 11)


# In[819]:


#len(text_dump_new)


# In[798]:


df = generate_dataset(text_dump_new, Params.City)


# In[800]:


df_copy = df
df_copy['{mobile}'].replace('', np.nan, inplace=True)
df_copy.dropna(subset=['{mobile}'], inplace=True)
df_copy = df_copy.drop_duplicates(subset=['{mobile}', '{service_name}'])


# In[779]:


df_copy.to_excel(Params.dir_path + "covidSoS_data_" + Params.City + ".xlsx", index = False)

