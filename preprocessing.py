

import pandas as pd
import opendatasets as od
import random



# od.download("https://www.kaggle.com/competitions/spaceship-titanic")




raw_data=pd.read_csv("/Users/Tyreek ALEXANDER/Documents/kaggle_titanic/spaceship-titanic/train.csv")


ytrain=raw_data['Transported']



def clean_dt(raw_data):
    extract_cabin=raw_data.Cabin.str.split('/',expand=True)
    return (raw_data
    .drop(columns=['Name','Cabin','PassengerId'])
    .assign(
        Deck=extract_cabin[0].fillna(random.choice(['A','B','C','D','E','F','G','T'])),
        Starboard=extract_cabin[2].fillna(random.choice(['S','P'])),

        CryoSleep=raw_data.CryoSleep.fillna(random.choice([True,False])),
        VIP=raw_data.VIP.fillna(random.choice([True,False])),
        
        Destination=raw_data.Destination.fillna(random.choice(random.choice(['TRAPPIST-1e','55 Cancri e','PSO J318.5-22']))),
        HomePlanet=raw_data.HomePlanet.fillna(random.choice(random.choice(['Earth','Europa','Mars']))),
        
        ShoppingMall=raw_data.ShoppingMall.fillna(0).replace('nan',0),
        Spa=raw_data.Spa.fillna(0).replace('nan',0),
        VRDeck=raw_data.VRDeck.fillna(0).replace('nan',0),
        FoodCourt=raw_data.FoodCourt.fillna(0).replace('nan',0),
        RoomService=raw_data.RoomService.fillna(0).replace('nan',0),
        
        Age=raw_data.Age.fillna(raw_data.Age.median())
        )
     
    )


# encoding boolean labels
from sklearn import preprocessing

def transform_step(raw_data):
   label_encoder=preprocessing.LabelEncoder()

   raw_data.CryoSleep=label_encoder.fit_transform(raw_data.CryoSleep).astype(int)
   raw_data.Starboard=label_encoder.fit_transform(raw_data.Starboard).astype(int)
   raw_data.VIP=label_encoder.fit_transform(raw_data.VIP)
   
   raw_data=pd.get_dummies(raw_data,columns=['Destination','HomePlanet'])
   raw_data=pd.get_dummies(raw_data,columns=['Deck'])

   return raw_data


res_encoder=preprocessing.LabelEncoder()
ytrain=res_encoder.fit_transform(ytrain)


Xtrain=clean_dt(raw_data)
Xtrain=transform_step(Xtrain)
Xtrain.drop(columns=['Transported'],inplace=True)   




raw_test=pd.read_csv('/Users/Tyreek ALEXANDER/Documents/kaggle_titanic/spaceship-titanic/test.csv')

Xtest=clean_dt(raw_test)
Xtest=transform_step(Xtest)


from sklearn import model_selection

kag_X_train,kag_X_test,kag_y_train,kag_y_test=model_selection.train_test_split(Xtrain,ytrain,random_state=69,stratify=ytrain)


# python -m jupyter nbconvert --to python full_wrangling.ipynb 


