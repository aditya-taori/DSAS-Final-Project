# Databricks notebook source
# MAGIC %md
# MAGIC ## Rubric for this module
# MAGIC - Implement a routine to "promote" your model at **Staging** in the registry to **Production** based on a boolean flag that you set in the code.
# MAGIC - Using wallet addresses from your **Staging** and **Production** model test data, compare the recommendations of the two models.

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# Grab the global variables
wallet_address,start_date = Utils.create_widgets()
print(wallet_address,start_date)

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## Your Code Starts Here...

# COMMAND ----------

deploy_prod_yn = 0

# COMMAND ----------

MY_EXPERIMENT = "/Users/" + dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get() + "/G10_experiment" 
#mlflow.set_experiment(MY_EXPERIMENT)
mlflow_client = mlflow.tracking.MlflowClient()
try:
    print("Inside Try Blocks")
    experimentID = mlflow_client.get_experiment_by_name(name=MY_EXPERIMENT).experiment_id
except:
    print("Inside Catch Block")
    mlflow.set_experiment(MY_EXPERIMENT)
    experimentID = mlflow_client.get_experiment_by_name(name=MY_EXPERIMENT).experiment_id

# COMMAND ----------

runs = mlflow_client.search_runs(experimentID, order_by=["attributes.start_time desc"], max_results=1)
runs[0].data.metrics

# COMMAND ----------

runs[0].info.run_id

# COMMAND ----------

test_df = spark.table("g10_db.silver_tokenmatrix")
display(test_df)

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()
model_versions = []
modelName = 'G10_model'

# COMMAND ----------

stage_model = mlflow.spark.load_model('models:/'+modelName+'/Staging')
stage_model

# COMMAND ----------

prod_model = mlflow.spark.load_model('models:/'+modelName+'/Production')
prod_model

# COMMAND ----------

versions_list = client.search_model_versions(f"name='{modelName}'")
model_version = ""
for i in versions_list:
    i = dict(i)
    c_stage = i["current_stage"]
    print(c_stage)
    if c_stage=="Production":
        model_version = i["version"]
        print(model_version)
        break
prod_version = model_version

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
staging_version = model_version

# COMMAND ----------

#test_df = test_df.toPandas()
test_input = test_df.select(col("token_id").alias("tokenID"),col("user_id").alias("userID"),"#_transfers")

#test_input = test_df[["token_id","user_id","#_transfers"]]
#test_input.columns = ["tokenID","userID","#_transfers"]
display(test_input)
reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="#_transfers", metricName="rmse")

# COMMAND ----------

prod_predictions = prod_model.transform(test_input)
staging_predictions = stage_model.transform(test_input)

prod_predictions = prod_predictions.filter(prod_predictions.prediction != float('nan'))
prod_predictions = prod_predictions.withColumn("prediction", F.abs(F.round(prod_predictions["prediction"],0)))

staging_predictions = staging_predictions.filter(staging_predictions.prediction != float('nan'))
staging_predictions = staging_predictions.withColumn("prediction", F.abs(F.round(staging_predictions["prediction"],0)))

prod_rmse = reg_eval.evaluate(prod_predictions)
staging_rmse = reg_eval.evaluate(staging_predictions)

# COMMAND ----------

print(prod_rmse)
print(staging_rmse)

# COMMAND ----------

if prod_rmse>staging_rmse:
    deploy_prod_yn = 1

# COMMAND ----------

print(deploy_prod_yn)
deploy_prod_yn = 1

# COMMAND ----------


client.transition_model_version_stage(
    name=modelName,
    version=staging_version,  # this model (current build)
    stage="Production"
)

client.transition_model_version_stage(
    name=modelName,
    version=str(int(staging_version)-1),  # this model (current build)
    stage="Staging"
)

# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))

# COMMAND ----------


