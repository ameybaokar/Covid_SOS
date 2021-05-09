#!/usr/bin/env python
# coding: utf-8

# Importing the prerequisities

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
from datetime import datetime, timedelta
from constants import Constants


# Declare the input values

class TwitterScrapper:
    def __init__(self) :
            #service = "Plasma Donors"
        self.date_until = (datetime.utcnow() + timedelta(days=1)).date()
        self.date_since = datetime.utcnow().date() 
        self.total_tweets = 10000
        self.access_token= Constants.Access_token
        self.access_token_secret= Constants.Access_token_secret
        self.consumer_key= Constants.Consumer_key
        self.consumer_secret= Constants.Consumer_secret
        self.dir_path = Constants.Dir #'/Users/xyz/Downloads/Stanford_NER/'
        self.file_path = 'stanford-ner-2020-11-17/classifiers/english.conll.4class.distsim.crf.ser.gz'
        self.jar_file_path = 'stanford-ner-2020-11-17/stanford-ner.jar'

        with open(Constants.Dir + 'config.json') as f:
            widgets = json.load(f)
        
        self.cities = [i['name'] for i in widgets]
        self.states = [i['state'] for i in widgets]
        self.citi_to_state = {}
        for i in widgets:
            self.citi_to_state[i['name']] = i['state']

        self.services = { "Plasma Donors": ["A-", "A+", "AB-", "AB+","O-","O+", "B+", "B-"],
           "Hospital Beds" : ["Ventilator", "Isolation Bed", "Covid Bed", "Oxygen Bed", "ICU Bed"],
           "Oxygen Suppliers" : ["Oxygen Cylinder Refill", "Oxygen Cylinder", "Oxygen Concentrator", "Portable Oxygen Can", "Oxygen Cylinder Equipment"],
            "Testing Centres" : ["RT-PCR", "TrueNat", "CBNAAT", "Rapid Antigen"],
           "Ambulance" : ["With Oxygen", "Without Oxygen"],
           "Food" : ["Lunch", "Breakfast", "Dinner"],
           "Medicine" : ["Medrol", "Remdesivir", "Remdisivir", "Doxycycline", "Dexamethasone", "Oximeter", "Ivermectin", "Favipiravir", "Tocilizumab"],
            "Remdisivir": ["Remdisivir"]
            }



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
        self.new_cat = []
        for i in spl_c:
            
            self.new_cat.append(list(set(i.split(' '))))

        #new_loc = []
        self.new_cat = [j for i in self.new_cat for j in i]
        #new_loc = [", ".join(i) for i in new_loc]
        self.new_cat = list(set(self.new_cat))


        # Function to fetch tweets based on the params defined by user

        auth = tw.OAuthHandler(self.consumer_key, self.consumer_secret)
        auth.set_access_token(self.access_token, self.access_token_secret)
        self.api = tw.API(auth, wait_on_rate_limit=True)

        self.st = StanfordNERTagger("./" + self.file_path,
                    "./" + self.jar_file_path,
                    encoding='utf-8')


    def get_tweets(self, search_words, date_since, date_until, size = 5, hours = 5):
        # Define the search term and the date_since date as variables
        new_search = search_words + " -filter:retweets"
        tweets = tw.Cursor(self.api.search,
                    q=new_search,
                    lang="en",
                    since=date_since, until=date_until, tweet_mode='extended').items(size)

        #tweets = api.search(q=search_words, count=size, tweet_mode='extended')

        # Iterate and print tweets
        res = []
        time_limit = datetime.utcnow() - timedelta(hours=hours)
        for tweet in tweets:
            print(tweet.created_at, time_limit, tweet.created_at > time_limit)
            if not tweet.retweeted :
                if tweet.created_at > time_limit:
                    res.append(tweet.full_text)
                else :
                    break;
        
        return res


    # Converting tweets to text format
    def get_tweet_data(self, search_words, date_since, date_until, hours):
        text_dump = [] #OXYGEN REFILL - Lucknow. Register at \n \n\nCylinders to be refilled as per oxygen level.…'
        for i in search_words:
            print(" ==== TWEET EXTRACTION IN PROCESS ===")
            tweet_dump = self.get_tweets(search_words = i, 
                                                    date_since = date_since, date_until=date_until, size = self.total_tweets, hours=hours)
            # text = 'New Delhi is having a lot of cases. Help needed.'
            print(" ==== TWEETS CLEANING IN PROCESS ===")
            
            for i in tweet_dump:
                text_dump.append(i)
        
        return text_dump


    # Creating instance of POS tagger

    # In[12]:


    #the below mentioned paths should be based on user's local path variable.
    #in order to run the below code, download the classifier from the folloring location:
    #https://nlp.stanford.edu/software/CRF-NER.html#Download
    #under the "Downloads" section with file name "Download Stanford Named Entity Recognizer version 4.2.0"



    def get_ner(self, text):
        tokenized_text = word_tokenize(text)
        tokens_without_sw = [word for word in tokenized_text if not word in stopwords.words()]
        classified_text = self.st.tag(tokens_without_sw)
        return classified_text


    # Function Generating the dataset

    # In[698]:


    def generate_dataset(self, text_dump, city_1):
        idx = 0
        data = pd.DataFrame()
        for text in tqdm(text_dump, desc = "progress of tweets churned " + city_1):
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
            
            classified_text = self.get_ner(text)
            services_keys = list(self.services.keys())
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
                if i.lower() in map(str.lower, self.cities):
                    loc.append(i)
                    org_remove.append(i)
                elif i.lower() in map(str.lower, self.new_cat):
                    cat.append(i)
                    org_remove.append(i)


            if  len(org_remove)>0:
                for j in org_remove:
                    org.remove(j)


            #####Removing keywords that do not belong under Location
            loc_remove = []
            for i in loc:
                if i.lower() in map(str.lower, self.new_cat):
                    cat.append(i)
                    loc_remove.append(i)


            if  len(loc_remove)>0:
                for j in loc_remove:
                    loc.remove(j)


            ####cleaning others

            for k in other:
                #print(i)
                if k.lower() in map(str.lower, self.cities):
                    loc.append(k)
                elif k.lower() in map(str.lower, self.new_cat):
                    cat.append(k)
                    
            for m in person:
                #print(i)
                if m.lower() in map(str.lower, self.cities):
                    loc.append(m)
                elif m.lower() in map(str.lower, self.new_cat):
                    cat.append(m)

            city = [i for i in loc if i in self.cities]
            address = [i for i in loc if i not in city]       
            ser = [i for i in list(self.services.keys()) if len(set(i.split(" ")) & set(cat)) > 0]
            city = [i for i in city if i.capitalize() == city_1]
            state = [self.citi_to_state[i] for i in city if i in list(self.citi_to_state.keys())]
            
            num = list(set(num))
            num = [i.replace(" ", "") for i in list(set(num))]
            org = " ".join(org)
            len_org = len(org)
            for i in ser:
                sub_ser = [k for k in self.services[i] if len(set(map(str.lower,k.split(" "))) & set(map(str.lower, cat))) > 0]
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

    # In[774]:

    def get_xls_for_city(self, city="Noida", hours=5, date_since = None,  date_until = None):
        #search_words =  ["verified+noida+Plasma+-'needed'+-'need'+-'needs'+-'required'+-'require'+-'requires'+-'requirement'+-'requirements'+-'rqrd'"] #[#Remdesivir available ", "#Plasma available Bangalore", "#Oxygen available Bangalore"]
        search_words = ["verified+" + city + "+bed+OR+beds+OR+icu+OR+oxygen+OR+ventilator+OR+ventilators+OR+remdesivir+OR+favipiravir+OR+tocilizumab+OR+plasma+OR+tiffin+OR+food+OR+ambulance+-'not+verified'+-'unverified'+-'needed'+-'need'+-'needs'+-'required'+-'require'+-'requires'+-'requirement'+-'requirements'+-'rqrd'"]
        #search_words = ["verified+" + City + "+plasma+OR+oxygen+OR+concentrator+OR+oxygen can+OR+Oxygen Refill+-'not+verified'+-'unverified'+-'needed'+-'need'+-'needs'+-'required'+-'require'+-'requires'+-'requirement'+-'requirements'"]
        if not date_since:
            date_since = self.date_since
        if not date_until:
            date_until = self.date_until
        text_dump = self.get_tweet_data(search_words, date_since, date_until, hours)

        if len(text_dump) > 0 :
            df = self.generate_dataset(text_dump, city)
            if not df.empty:
                df_copy = df
                df_copy['{mobile}'].replace('', np.nan, inplace=True)
                df_copy.dropna(subset=['{mobile}'], inplace=True)
                df_copy = df_copy.drop_duplicates(subset=['{mobile}', '{service_name}'])
                if not df_copy.empty:
                    path = self.dir_path + "to-be-pushed/covidSoS_data_" + city + ".xlsx"
                    df_copy.to_excel(path, index = False)
                    return path
        return False
