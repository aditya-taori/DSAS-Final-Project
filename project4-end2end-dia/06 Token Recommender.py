# Databricks notebook source
# MAGIC %md
# MAGIC ## Token Recommendation
# MAGIC <table border=0>
# MAGIC   <tr><td><img src='https://data-science-at-scale.s3.amazonaws.com/images/rec-application.png'></td>
# MAGIC     <td>Your application should allow a specific wallet address to be entered via a widget in your application notebook.  Each time a new wallet address is entered, a new recommendation of the top tokens for consideration should be made. <br> **Bonus** (3 points): include links to the Token contract on the blockchain or etherscan.io for further investigation.</td></tr>
# MAGIC   </table>

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# Grab the global variables
wallet_address,start_date = Utils.create_widgets()
print(wallet_address,start_date)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Your code starts here...

# COMMAND ----------

import mlflow
from pyspark.sql import functions as F
import pandas as pd
class ALS_prediction():

    def __init__(self, ALS_model):
        self.als = ALS_model
        self.triplet_subset = spark.table("g10_db.silver_tokenmatrix")
        self.token_meta = spark.table("g10_db.silvertokenmetadata").toPandas()
        
    def preprocess_input(self, userID):
        '''return pre-processed model_input'''
        used_tokens_list = []
        # generate dataframe of unlistened songs
        unused_tokens = self.triplet_subset.filter(~ self.triplet_subset['token_id'].isin(used_tokens_list)) \
                                                    .select(col('token_id').alias("tokenID")).withColumn('userID', F.lit(userID)).distinct()

        return unused_tokens.toPandas()
      
    def postprocess_result(self,processed_input, results):
        print(len(results))
        
        processed_input["prediction"] =results 
        #processed_input = processed_input.filter(processed_input['prediction'] != float('nan'))
        processed_input.columns = ["token_id","userID","prediction"]
        #display(self.token_meta)
        # print output
        print('Predicted Tokens:')
        #processed_input.select(col('tokenID').alias("token_id"),"userID","prediction")
        return pd.merge(processed_input,self.token_meta, on = 'token_id',how = "inner") 
    
    def prediction(self, model_input):
        processed_model_input = self.preprocess_input(model_input)
        display(processed_model_input)
        results = self.als.predict(processed_model_input)
        results.append(0)
        return self.postprocess_result(processed_model_input,results)

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()
model_versions = []
modelName = 'G10_model'

# COMMAND ----------

versions_list = client.search_model_versions(f"name='{modelName}'")
for i in versions_list:
    i = dict(i)
    c_stage = i["current_stage"]
    print(c_stage)
    if c_stage=="Staging":
        model_version = i["version"]
        print(model_version)
        break

# COMMAND ----------

import mlflow.pyfunc

model_version_uri = "models:/{model_name}/{model_version}".format(model_name=modelName,model_version=model_version)

print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_version_uri))
model_version_1 = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

prod_object = ALS_prediction(model_version_1)
recommendaton_out = prod_object.prediction(model_input = 20)

# COMMAND ----------

sentence = "<h3>Recommend Tokens for User address</h3> <img src='"+img+"'>"
#sentence = "the result passed the condition with a value of"+   str(my_result)

body_text = "<h3>Recommend Tokens for User address</h3> <p>"+wallet_address+"</p>" 

table_text = "<table>\
  <tr>\
    <td><img src='"+img+"'></td>\
    <td>"+ token_name+"</td>\
    <td><a href='"+token_link+"'>Link</td>\
  </tr>\
</table>"

comp_table_text = "<table> "

for i in range(5):
    img = recommendaton_out["image"].iloc[i]
    print(img)
    token_name = recommendaton_out["name"].iloc[i]
    token_link = recommendaton_out["links"].iloc[i]
    token_address = recommendaton_out["token_address"].iloc[i]
    ether_link = "https://etherscan.io/token/"+token_address
    table_text = "<tr> <td><img src='"+img+"'></td> <td>"+ token_name+"</td> <td><a href='"+ether_link+"'>Link</td> </tr>"
    comp_table_text = comp_table_text + table_text

comp_table_text = comp_table_text + "</table>"
#print(comp_table_text)
html_text = body_text + " "+ comp_table_text

print(html_text)

displayHTML(html_text)

# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
