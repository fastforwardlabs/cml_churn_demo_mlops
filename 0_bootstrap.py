# # Part 0: Bootstrap File
# You need to at the start of the project. It will install the requirements, creates the 
# STORAGE environment variable and copy the data from 
# raw/WA_Fn-UseC_-Telco-Customer-Churn-.csv into /datalake/data/churn of the STORAGE 
# location.

# The STORAGE environment variable is the Cloud Storage location used by the DataLake 
# to store hive data. On AWS it will s3a://[something], on Azure it will be 
# abfs://[something] and on CDSW cluster, it will be hdfs://[something]

# Install the requirements
!pip3 install -r requirements.txt --progress-bar off
  
# Create the directories and upload data

from cmlbootstrap import CMLBootstrap
from IPython.display import Javascript, HTML
import os
import time
import json
import requests
import xml.etree.ElementTree as ET
import datetime

try: 
  os.environ["SPARK_HOME"]
  print("Spark is enabled")
except:
  print('Spark is not enabled, please enable spark before running this script')
  raise KeyError('Spark is not enabled, please enable spark before running this script')

run_time_suffix = datetime.datetime.now()
run_time_suffix = run_time_suffix.strftime("%d%m%Y%H%M%S")

# Instantiate API Wrapper
cml = CMLBootstrap()

# Set the STORAGE environment variable
try : 
  storage=os.environ["STORAGE"]
except:
  storage = cml.get_cloud_storage()
  storage_environment_params = {"STORAGE":storage}
  storage_environment = cml.create_environment_variable(storage_environment_params)
  os.environ["STORAGE"] = storage

# Upload the data to the cloud storage
!hadoop fs -mkdir -p $STORAGE/datalake
!hadoop fs -mkdir -p $STORAGE/datalake/data
!hadoop fs -mkdir -p $STORAGE/datalake/data/churn
!hadoop fs -copyFromLocal /home/cdsw/raw/WA_Fn-UseC_-Telco-Customer-Churn-.csv $STORAGE/datalake/data/churn/WA_Fn-UseC_-Telco-Customer-Churn-.csv

