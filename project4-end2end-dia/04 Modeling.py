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

x = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().getOrElse(None)


# COMMAND ----------

#
#exp_id
import os
os.environ['MLFLOW_EXPERIMENT_NAME'] = x
exp_id = mlflow.create_experiment(x)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Your Code starts here...

# COMMAND ----------

erc20_token_subset = spark.sql("select * from g10_db.erc20_token_transfers order by block_number limit 10000")
erc20_token_subset.head(5)
erc20_token_subset.cache()

# COMMAND ----------

erc20_token_subset.show(5)

# COMMAND ----------

erc20_token_subset.count()


# COMMAND ----------

#erc20_token_subset.select("from_address").where(col("from_address").isNull()).count()
erc20_token_subset.select("token_address").where(col("token_address").isNull()).count()

# COMMAND ----------

tokens_counts = erc20_token_subset.groupby('token_address').count()
display(tokens_counts)
tokens_counts = tokens_counts.select("*").withColumn("token_ids", monotonically_increasing_id())
display(tokens_counts)

# COMMAND ----------

from_address_counts = erc20_token_subset.groupby('from_address').count()
display(from_address_counts)
from_address_counts = from_address_counts.select("*").withColumn("user_id", monotonically_increasing_id())
display(from_address_counts)

# COMMAND ----------

erc20_token_subset = erc20_token_subset.join(from_address_counts,erc20_token_subset.from_address==from_address_counts.from_address)
display(erc20_token_subset)
erc20_token_subset = erc20_token_subset.join(tokens_counts,erc20_token_subset.token_address==tokens_counts.token_address)
display(erc20_token_subset)

# COMMAND ----------

triplet_df = (erc20_token_subset.groupBy("user_id","token_ids").count())

# COMMAND ----------

display(triplet_df)

# COMMAND ----------

seed = 42
(split_60_df, split_a_20_df, split_b_20_df) = triplet_df.randomSplit([0.6, 0.2, 0.2], seed = seed)

# Let's cache these datasets for performance
training_df = split_60_df.cache()
validation_df = split_a_20_df.cache()
test_df = split_b_20_df.cache()

print('Training: {0}, validation: {1}, test: {2}\n'.format(
  training_df.count(), validation_df.count(), test_df.count())
)
training_df.show(3)
validation_df.show(3)
test_df.show(3)

# COMMAND ----------

#Number of plays needs to be double type, not integers
validation_df = validation_df.withColumn("count", validation_df["count"].cast(DoubleType()))
validation_df.show(10)

#Number of plays needs to be double type, not integers
training_df = training_df.withColumn("count", training_df["count"].cast(DoubleType()))
training_df.show(10)

#Number of plays needs to be double type, not integers
test_df = test_df.withColumn("count", test_df["count"].cast(DoubleType()))
test_df.show(10)

# COMMAND ----------

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F
from mlflow.tracking import MlflowClient


# Let's initialize our ALS learner
als = ALS()

# Now set the parameters for the method
als.setMaxIter(5)\
   .setSeed(seed)\
   .setItemCol("token_ids")\
   .setRatingCol("count")\
   .setUserCol("user_id")\
   .setColdStartStrategy("drop")

# Now let's compute an evaluation metric for our test dataset
# We Create an RMSE evaluator using the label and predicted columns
reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="count", metricName="rmse")

grid = ParamGridBuilder() \
  .addGrid(als.maxIter, [10]) \
  .addGrid(als.regParam, [0.15, 0.2, 0.25]) \
  .addGrid(als.rank, [4, 8, 12, 16]) \
  .build()

# Create a cross validator, using the pipeline, evaluator, and parameter grid you created in previous steps.
cv = CrossValidator(estimator=als, evaluator=reg_eval, estimatorParamMaps=grid, numFolds=3)



tolerance = 0.03
ranks = [4, 8, 12, 16]
regParams = [0.15, 0.2, 0.25]
errors = [[0]*len(ranks)]*len(regParams)
models = [[0]*len(ranks)]*len(regParams)
err = 0
min_error = float('inf')
best_rank = -1
i = 0
for regParam in regParams:
    j = 0
    for rank in ranks:
    # Set the rank here:
        als.setParams(rank = rank, regParam = regParam)
        # Create the model with these parameters.
        model = als.fit(training_df)
        # Run the model to create a prediction. Predict against the validation_df.
        predict_df = model.transform(validation_df)

        # Remove NaN values from prediction (due to SPARK-14489)
        predicted_plays_df = predict_df.filter(predict_df.prediction != float('nan'))
        predicted_plays_df = predicted_plays_df.withColumn("prediction", F.abs(F.round(predicted_plays_df["prediction"],0)))
        # Run the previously created RMSE evaluator, reg_eval, on the predicted_ratings_df DataFrame
        error = reg_eval.evaluate(predicted_plays_df)
        errors[i][j] = error
        models[i][j] = model
        print( 'For rank %s, regularization parameter %s the RMSE is %s' % (rank, regParam, error))
        if error < min_error:
            min_error = error
            best_params = [i,j]
        j += 1
    i += 1

als.setRegParam(regParams[best_params[0]])
als.setRank(ranks[best_params[1]])
print( 'The best model was trained with regularization parameter %s' % regParams[best_params[0]])
print( 'The best model was trained with rank %s' % ranks[best_params[1]])
my_model = models[best_params[0]][best_params[1]]

# COMMAND ----------

predicted_plays_df.show(10)

# COMMAND ----------

test_df = test_df.withColumn("count", test_df["count"].cast(DoubleType()))
predict_df = my_model.transform(test_df)

# Remove NaN values from prediction (due to SPARK-14489)
predicted_test_df = predict_df.filter(predict_df.prediction != float('nan'))

# Round floats to whole numbers
predicted_test_df = predicted_test_df.withColumn("prediction", F.abs(F.round(predicted_test_df["prediction"],0)))
# Run the previously created RMSE evaluator, reg_eval, on the predicted_test_df DataFrame
test_RMSE = reg_eval.evaluate(predicted_test_df)

print('The model had a RMSE on the test set of {0}'.format(test_RMSE))

# COMMAND ----------

UserID = 125
used_tokens = erc20_token_subset.filter(erc20_token_subset.user_id == UserID) \
                                          .select('token_ids') \
                                          
# generate list of listened songs
used_tokens_list = []
for tokens in used_tokens.collect():
    used_tokens_list.append(tokens['token_ids'])

print('Tokens User has used :')
used_tokens.select('token_ids').show()

# generate dataframe of unlistened songs
unused_tokens = erc20_token_subset.filter(~ erc20_token_subset['token_ids'].isin(used_tokens_list)) \
                                            .select('token_ids').withColumn('user_id', F.lit(UserID)).distinct()

# feed unlistened songs into model
predicted_tokens = my_model.transform(unused_tokens)

# remove NaNs
predicted_tokens = predicted_tokens.filter(predicted_tokens['prediction'] != float('nan'))

# print output
print('Predicted Songs:')
predicted_tokens.join(erc20_token_subset, 'user_id') \
                 .join(erc20_token_subset, 'token_ids') \
                 .select('spark_catalog.g10_db.erc20_token_transfers.from_address', 'spark_catalog.g10_db.erc20_token_transfers.token_address', 'prediction') \
                 .distinct() \
                 .orderBy('prediction', ascending = False) \
                 .show(10)

# COMMAND ----------

erc20_token_subset.unpersist()

# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
