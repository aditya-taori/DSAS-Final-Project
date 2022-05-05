# Databricks notebook source
# MAGIC %md
# MAGIC ## Rubric for this module
# MAGIC - Using the silver delta table(s) that were setup by your ETL module train and validate your token recommendation engine. Split, Fit, Score, Save
# MAGIC - Log all experiments using mlflow
# MAGIC - capture model parameters, signature, training/test metrics and artifacts
# MAGIC - Tune hyperparameters using an appropriate scaling mechanism for spark.  [Hyperopt/Spark Trials ](https://docs.databricks.com/_static/notebooks/hyperopt-spark-ml.html)
# MAGIC - Register your best model from the training run at **Staging**.

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# Grab the global variables
wallet_address,start_date = Utils.create_widgets()
print(wallet_address,start_date)

# COMMAND ----------

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F
from mlflow.tracking import MlflowClient
from mlflow.types.schema import Schema, ColSpec
from mlflow.models.signature import ModelSignature

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Your Code starts here...

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
experimentID

# COMMAND ----------

import tempfile
modelName = 'G10_model'

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


# split the data set into train, validation and test anc cache them
# We'll hold out 60% for training, 20% of our data for validation, and leave 20% for testing
triplet_subset = spark.table("g10_db.silver_tokenmatrix").select(col('token_id').alias("tokenID"),col('user_id').alias("userID"),"#_transfers")
token_meta = spark.table("g10_db.silvertokenmetadata")

seed = 42
(split_60_df, split_a_20_df, split_b_20_df) = triplet_subset.randomSplit([0.6, 0.2, 0.2], seed = seed)

# Let's cache these datasets for performance
training_df = split_60_df.cache()
validation_df = split_a_20_df.cache()
test_df = split_b_20_df.cache()

#print('Training: {0}, validation: {1}, test: {2}\n'.format(
#  training_df.count(), validation_df.count(), test_df.count())
#)
training_df.show(3)
validation_df.show(3)
test_df.show(3)

als = ALS()

# Now set the parameters for the method
als.setMaxIter(5)\
   .setSeed(seed)\
   .setItemCol("tokenID")\
   .setRatingCol("#_transfers")\
   .setUserCol("userID")\
   .setColdStartStrategy("drop")

# Now let's compute an evaluation metric for our test dataset
# We Create an RMSE evaluator using the label and predicted columns
reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="#_transfers", metricName="rmse")


input_schema = Schema([
  ColSpec("integer", "tokenID"),
  ColSpec("integer", "userID"),
])
output_schema = Schema([ColSpec("double")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

with mlflow.start_run(experiment_id=experimentID,run_name="Basic ALS Experiment_v3") as run:
    mlflow.set_tags({"group": 'G10', "class": "DSCC202-402"})
    mlflow.log_params({"user_rating_training_data_version": training_df,"user_rating_testing_data_version": test_df,"rank":12,"regParam":0.25})

    als.setParams(rank = 12, regParam = 0.30)
    # Create the model with these parameters.
    model = als.fit(training_df)
    # Run the model to create a prediction. Predict against the validation_df.
    predict_df = model.transform(validation_df)

    # Remove NaN values from prediction (due to SPARK-14489)
    predicted_plays_df = predict_df.filter(predict_df.prediction != float('nan'))
    predicted_plays_df = predicted_plays_df.withColumn("prediction", F.abs(F.round(predicted_plays_df["prediction"],0)))
    
    training_RMSE = reg_eval.evaluate(predict_df)


    
    #mlflow.spark.log_model(spark_model=cvModel.bestModel, signature = signature,
    #                         artifact_path='als-model', registered_model_name=self.modelName)

    mlflow.spark.log_model(spark_model=model, artifact_path='als-best-model',signature = signature,registered_model_name=modelName)
    mlflow.log_metric("training_rmse", training_RMSE)
    runID = run.info.run_uuid
    experimentID = run.info.experiment_id

    temp = tempfile.NamedTemporaryFile(prefix="testing_data", suffix=".csv")
    temp_name = temp.name
    try:
        test_df.toPandas().to_csv(temp_name, index=False)
        mlflow.log_artifact(temp_name, "testing_data.csv")
    finally:
        temp.close() # Delete the temp file
    print(f"Inside MLflow Run with run_id `{runID}` and experiment_id `{experimentID}`")

client = MlflowClient()
model_versions = []
    # Transition this model to staging and archive the current staging model if there is one
for mv in client.search_model_versions(f"name='{modelName}'"):
    model_versions.append(dict(mv)['version'])
    #print(model_versions)
    if dict(mv)['current_stage'] == 'Staging':
        print("Archiving: {}".format(dict(mv)))
        # Archive the currently staged model
        client.transition_model_version_stage(
            name=modelName,
            version=dict(mv)['version'],
            stage="Archived"
        )
client.transition_model_version_stage(
    name=modelName,
    version=model_versions[0],  # this model (current build)
    stage="Staging"
)

# COMMAND ----------



# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
