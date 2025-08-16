import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            # Data preprocessing

            df=pd.read_csv('artifacts/data.csv')
            
            # Below, converting relevant columns to timedelta to more accurately extract info
            df['duration_timedelta'] = pd.to_timedelta(df['Duration']) #create column of timedelta

            df['Dept full time'] = df['Date_of_Journey'] + ' ' + df['Dep_Time']
            df['timestamp_dept_time'] = pd.to_datetime(df['Dept full time'], format="%d/%m/%Y %H:%M")
            
            df['timestamp_arrival_time'] = df['timestamp_dept_time'] + df['duration_timedelta']
            df.drop(columns=['Date_of_Journey', 'Dep_Time', 'Arrival_Time', 'Dept full time'], axis=1, inplace=True)

            df.dropna(inplace=True) # Remove NA values

            df = df.drop_duplicates() # there are duplicate rows, keep the first occurance

            # Below, extracting day, month, hour and minute
            df['dept_day'] = df['timestamp_dept_time'].dt.day
            df['dept_month'] = df['timestamp_dept_time'].dt.month
            df['dept_hour'] = df['timestamp_dept_time'].dt.hour
            df['dept_minute'] = df['timestamp_dept_time'].dt.minute

            df['arrival_day'] = df['timestamp_arrival_time'].dt.day
            df['arrival_month'] = df['timestamp_arrival_time'].dt.month
            df['arrival_hour'] = df['timestamp_arrival_time'].dt.hour
            df['arrival_minute'] = df['timestamp_arrival_time'].dt.minute

            df['duration_hours'] = df['duration_timedelta'].dt.components.hours
            df['duration_minutes'] = df['duration_timedelta'].dt.components.minutes

            # Drop unneeded columns
            df.drop(columns=['Duration', 'timestamp_dept_time', 'timestamp_arrival_time', 'duration_timedelta'], axis=1, inplace=True)

            categorical_features=[column for column in df.columns if df[column].dtype=='object']

            numerical_features=[column for column in df.columns if df[column].dtype!='object']

            # one hot encoding Airline column
            Airline=pd.get_dummies(df['Airline'],drop_first=True, dtype=int)

            # one hot encoding source column
            Source=pd.get_dummies(df['Source'],drop_first=True, dtype=int)

            # one hot encoding destination column
            Destination=pd.get_dummies(df['Destination'],drop_first=True, dtype=int)

            # split each route into seperate column
            df['Route1']=df['Route'].str.split('→').str[0]
            df['Route2']=df['Route'].str.split('→').str[1]
            df['Route3']=df['Route'].str.split('→').str[2]
            df['Route4']=df['Route'].str.split('→').str[3]
            df['Route5']=df['Route'].str.split('→').str[4]

            # drop route column
            df.drop(columns=['Route'], axis=1, inplace=True)

            for i in ['Route3', 'Route4', 'Route5']: # fill NA with None
                df[i].fillna('None',inplace=True)

            encoder = LabelEncoder()
            for i in ['Route1', 'Route2', 'Route3', 'Route4', 'Route5']: # encode values into numbers
                df[i]=encoder.fit_transform(df[i])

            # since ordinal data, encode as number
            df['Total_Stops']=df['Total_Stops'].map({'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4})

            # drop unneeded columns
            df.drop(columns=['Airline', 'Source', 'Destination', 'Additional_Info'], axis=1, inplace=True)

            # concat back airline, source, destination
            df=pd.concat([df,Airline,Source,Destination],axis=1)

            # Calculate percentile for outliers
            q1, q3 = np.percentile(df['Price'], [25, 75])
            #print(q1)
            #print(q3)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # replace outliers with median
            df['Price']=np.where(df['Price']>upper_bound,df['Price'].median(),df['Price'])

            # Data does not seem to follow distribution so MinMaxScaler
            scaler = MinMaxScaler()
            scaled_df = scaler.fit_transform(df)
            final_df = pd.DataFrame(scaled_df, columns=df.columns)

            logging.info(f"Categorical features: {categorical_features}")
            logging.info(f"Numerical features: {numerical_features}")

            return final_df
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self):

        try:
            logging.info("Obtaining preprocessing object")
            final_df=self.get_data_transformer_object()

            target_column_name="Price"

            X=final_df.drop(target_column_name,axis=1)
            y=final_df[target_column_name]

            logging.info("Train test split initiated")

            input_feature_train_df,input_feature_test_df,target_feature_train_df,target_feature_test_df = train_test_split(X,y,test_size=0.20,random_state=42)

            logging.info("Split train and test data completed")

            logging.info(f"Create train_df and test_df")

            train_df = pd.concat([input_feature_train_df, target_feature_train_df], axis=1)
            test_df = pd.concat([input_feature_test_df, target_feature_test_df], axis=1)



            logging.info(
                f"Convert dataframe to array."
        
            )
            train_arr = np.c_[
                np.array(input_feature_train_df), np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                np.array(input_feature_test_df), np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=final_df

            )

            return (
                train_arr,
                test_arr,
                train_df,
                test_df,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)