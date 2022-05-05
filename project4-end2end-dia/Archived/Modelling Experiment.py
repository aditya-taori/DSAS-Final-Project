# Databricks notebook source
# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F
from mlflow.tracking import MlflowClient
from mlflow.types.schema import Schema, ColSpec
from mlflow.models.signature import ModelSignature

# COMMAND ----------

triplet_df = spark.table("g10_db.silver_tokenmatrix")
triplet_df.show(3)

# COMMAND ----------

triplet_df.printSchema()

# COMMAND ----------

triplet_subset = triplet_df

# COMMAND ----------

seed = 42
(split_60_df, split_a_20_df, split_b_20_df) = triplet_df.randomSplit([0.6, 0.2, 0.2], seed = seed)

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

# COMMAND ----------

#Number of plays needs to be double type, not integers
validation_df = validation_df.withColumn("#_transfers", validation_df["#_transfers"].cast(DoubleType()))
#validation_df.show(3)

#Number of plays needs to be double type, not integers
training_df = training_df.withColumn("#_transfers", training_df["#_transfers"].cast(DoubleType()))
#training_df.show(3)

#Number of plays needs to be double type, not integers
test_df = test_df.withColumn("#_transfers", test_df["#_transfers"].cast(DoubleType()))
#test_df.show(3)

# COMMAND ----------



with mlflow.start_run(run_name="Basic ALS Experiment") as run:
    # Let's initialize our ALS learner
    als = ALS()

    # Now set the parameters for the method
    als.setMaxIter(5)\
       .setSeed(seed)\
       .setItemCol("token_id")\
       .setRatingCol("#_transfers")\
       .setUserCol("user_id")\
       .setColdStartStrategy("drop")

    # Now let's compute an evaluation metric for our test dataset
    # We Create an RMSE evaluator using the label and predicted columns
    reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="#_transfers", metricName="rmse")

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
    # Log model
    mlflow.sklearn.log_model(my_model, "ALS_Best_model")
    mlflow.log_metric("rmse", min_error)
    runID = run.info.run_uuid
    experimentID = run.info.experiment_id
  
    print(f"Inside MLflow Run with run_id `{runID}` and experiment_id `{experimentID}`")

# COMMAND ----------


# setup the schema for the model
input_schema = Schema([
  ColSpec("integer", "userID"),
  ColSpec("integer", "tokenID"),
])
output_schema = Schema([ColSpec("double")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    

# COMMAND ----------

MY_EXPERIMENT = "/Users/" + dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get() + "/G10_experiment/" 
name = mlflow.get_experiment_by_name(MY_EXPERIMENT)
print(name)
with mlflow.start_run(run_name="Basic ALS Experiment_v2") as run:
    # Let's initialize our ALS learner
    mlflow.log_params({"user_rating_training_data_version": triplet_subset})
    als = ALS()

    # Now set the parameters for the method
    als.setMaxIter(5)\
       .setSeed(seed)\
       .setItemCol("token_id")\
       .setRatingCol("#_transfers")\
       .setUserCol("user_id")\
       .setColdStartStrategy("drop")

    # Now let's compute an evaluation metric for our test dataset
    # We Create an RMSE evaluator using the label and predicted columns
    reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="#_transfers", metricName="rmse")
    
    als.setParams(rank = 12, regParam = 0.25)
    # Create the model with these parameters.
    model = als.fit(training_df)
    # Run the model to create a prediction. Predict against the validation_df.
    predict_df = model.transform(validation_df)

    # Remove NaN values from prediction (due to SPARK-14489)
    predicted_plays_df = predict_df.filter(predict_df.prediction != float('nan'))
    predicted_plays_df = predicted_plays_df.withColumn("prediction", F.abs(F.round(predicted_plays_df["prediction"],0)))
    
    #mlflow.spark.log_model(spark_model=cvModel.bestModel, signature = signature,
    #                         artifact_path='als-model', registered_model_name=self.modelName)
    
    mlflow.spark.log_model(spark_model=model, artifact_path='als-best-model',signature = signature,registered_model_name="G10_model")
    mlflow.log_metric("rmse", min_error)
    runID = run.info.run_uuid
    experimentID = run.info.experiment_id
  
    print(f"Inside MLflow Run with run_id `{runID}` and experiment_id `{experimentID}`")

# COMMAND ----------

client = MlflowClient()
model_versions = []
modelName = 'G10_model'
    # Transition this model to staging and archive the current staging model if there is one
for mv in client.search_model_versions(f"name='{modelName}'"):
    model_versions.append(dict(mv)['version'])
    print(model_versions)
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

# THIS SHOULD BE THE VERSION JUST TRANINED
model = mlflow.spark.load_model('models:/'+modelName+'/Staging')
# View the predictions
test_predictions = model.transform(test_df)
RMSE = reg_eval.evaluate(test_predictions)
print("Staging Model Root-mean-square error on the test dataset = " + str(RMSE))  

# COMMAND ----------

UserID = 15
used_tokens = triplet_subset.filter(triplet_subset.user_id == UserID) \
                                          .select('token_id') \
                                   
# generate list of listened songs
used_tokens_list = []
for tokens in used_tokens.collect():
    used_tokens_list.append(tokens['token_id'])

print('Tokens User has used :')
used_tokens.select('token_id').show()

used_tokens_list = []
# generate dataframe of unlistened songs
unused_tokens = triplet_subset.filter(~ triplet_subset['token_id'].isin(used_tokens_list)) \
                                            .select('token_id').withColumn('user_id', F.lit(UserID)).distinct()

model = mlflow.spark.load_model('models:/'+modelName+'/Staging')

# feed unlistened songs into model
predicted_tokens = model.transform(unused_tokens)

# remove NaNs
predicted_tokens = predicted_tokens.filter(predicted_tokens['prediction'] != float('nan'))

# print output
print('Predicted Songs:')
predicted_tokens.show(10)




# COMMAND ----------

token_meta = spark.table("g10_db.silvertokenmetadata")
token_meta.show(5)

y = predicted_tokens.join(token_meta, 'token_id') \
                 .select("*") \
                 .orderBy('prediction', ascending = False) 

# COMMAND ----------

display(y)

# COMMAND ----------

display(unused_tokens)

# COMMAND ----------

triplet_subset.printSchema()

# COMMAND ----------

class token_recommendation():
    def __init__(self)->None:
        self.modelName = 'G10_model'

        # create an MLflow experiment for this model
        MY_EXPERIMENT = "/Users/" + dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get() + "/G10_experiment/" 
        name = mlflow.get_experiment_by_name(MY_EXPERIMENT)

        # split the data set into train, validation and test anc cache them
        # We'll hold out 60% for training, 20% of our data for validation, and leave 20% for testing
        self.triplet_subset = spark.table("g10_db.silver_tokenmatrix")
        self.token_meta = spark.table("g10_db.silvertokenmetadata")

        seed = 42
        (split_60_df, split_a_20_df, split_b_20_df) = triplet_subset.randomSplit([0.6, 0.2, 0.2], seed = seed)

        # Let's cache these datasets for performance
        self.training_df = split_60_df.cache()
        self.validation_df = split_a_20_df.cache()
        self.test_df = split_b_20_df.cache()

        #print('Training: {0}, validation: {1}, test: {2}\n'.format(
        #  training_df.count(), validation_df.count(), test_df.count())
        #)
        self.training_df.show(3)
        self.validation_df.show(3)
        self.test_df.show(3)

        als = ALS()

        # Now set the parameters for the method
        als.setMaxIter(5)\
           .setSeed(seed)\
           .setItemCol("token_id")\
           .setRatingCol("#_transfers")\
           .setUserCol("user_id")\
           .setColdStartStrategy("drop")

        # Now let's compute an evaluation metric for our test dataset
        # We Create an RMSE evaluator using the label and predicted columns
        self.reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="#_transfers", metricName="rmse")
    
    def train(self):
        input_schema = Schema([
          ColSpec("integer", "userID"),
          ColSpec("integer", "tokenID"),
        ])
        output_schema = Schema([ColSpec("double")])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
        with mlflow.start_run(run_name="Basic ALS Experiment_v3") as run:
            mlflow.set_tags({"group": 'G10', "class": "DSCC202-402"})
            mlflow.log_params({"user_rating_training_data_version": self.training_df,"user_rating_testing_data_version": self.test_df})
            
            als.setParams(rank = 12, regParam = 0.25)
            # Create the model with these parameters.
            model = als.fit(self.training_df)
            # Run the model to create a prediction. Predict against the validation_df.
            predict_df = model.transform(self.validation_df)

            # Remove NaN values from prediction (due to SPARK-14489)
            predicted_plays_df = predict_df.filter(predict_df.prediction != float('nan'))
            predicted_plays_df = predicted_plays_df.withColumn("prediction", F.abs(F.round(predicted_plays_df["prediction"],0)))

            #mlflow.spark.log_model(spark_model=cvModel.bestModel, signature = signature,
            #                         artifact_path='als-model', registered_model_name=self.modelName)

            mlflow.spark.log_model(spark_model=model, artifact_path='als-best-model',signature = signature,registered_model_name=self.modelName)
            mlflow.log_metric("rmse", min_error)
            runID = run.info.run_uuid
            experimentID = run.info.experiment_id

            print(f"Inside MLflow Run with run_id `{runID}` and experiment_id `{experimentID}`")
        
        client = MlflowClient()
        model_versions = []
            # Transition this model to staging and archive the current staging model if there is one
        for mv in client.search_model_versions(f"name='{self.modelName}'"):
            model_versions.append(dict(mv)['version'])
            #print(model_versions)
            if dict(mv)['current_stage'] == 'Staging':
                print("Archiving: {}".format(dict(mv)))
                # Archive the currently staged model
                client.transition_model_version_stage(
                    name=self.modelName,
                    version=dict(mv)['version'],
                    stage="Archived"
                )
        client.transition_model_version_stage(
            name=self.modelName,
            version=model_versions[0],  # this model (current build)
            stage="Staging"
        )
        
    def test(self):
        # THIS SHOULD BE THE VERSION JUST TRANINED
        model = mlflow.spark.load_model('models:/'+self.modelName+'/Staging')
        # View the predictions
        test_predictions = model.transform(self.test_df)
        RMSE = self.reg_eval.evaluate(test_predictions)
        print("Staging Model Root-mean-square error on the test dataset = " + str(RMSE))
    
    def recommend(self, userId: int)->(DataFrame):
        used_tokens_list = []
        # generate dataframe of unlistened songs
        unused_tokens = self.triplet_subset.filter(~ self.triplet_subset['token_id'].isin(used_tokens_list)) \
                                                    .select('token_id').withColumn('user_id', F.lit(userId)).distinct()

        model = mlflow.spark.load_model('models:/'+self.modelName+'/Staging')

        # feed unlistened songs into model
        predicted_tokens = model.transform(unused_tokens)

        # remove NaNs
        predicted_tokens = predicted_tokens.filter(predicted_tokens['prediction'] != float('nan'))

        # print output
        print('Predicted Songs:')
        
        return predicted_tokens.join(self.token_meta, 'token_id') \
                 .select("*") \
                 .orderBy('prediction', ascending = False) 

# COMMAND ----------

token_rec = token_recommendation()
token_rec.train()
token_rec.test()
recommendation = token_rec.recommend(20)
display(recommendation)

# COMMAND ----------

recommendation.write.csv("Recommendation.csv")

# COMMAND ----------

modelName = 'G10_model'

# create an MLflow experiment for this model
MY_EXPERIMENT = "/Users/" + dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get() + "/G10_experiment/" 
name = mlflow.get_experiment_by_name(MY_EXPERIMENT)

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

with mlflow.start_run(run_name="Basic ALS Experiment_v3") as run:
    mlflow.set_tags({"group": 'G10', "class": "DSCC202-402"})
    mlflow.log_params({"user_rating_training_data_version": training_df,"user_rating_testing_data_version": test_df})

    als.setParams(rank = 12, regParam = 0.25)
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

class ALS_prediction(mlflow.pyfunc.PythonModel):

    def __init__(self, ALS_model):
        self.als = ALS_model
        self.triplet_subset = spark.table("g10_db.silver_tokenmatrix")
        self.token_meta = spark.table("g10_db.silvertokenmetadata")
        
    def preprocess_input(self, userID):
        '''return pre-processed model_input'''
        used_tokens_list = []
        # generate dataframe of unlistened songs
        unused_tokens = self.triplet_subset.filter(~ self.triplet_subset['token_id'].isin(used_tokens_list)) \
                                                    .select('token_id').withColumn('user_id', F.lit(userID)).distinct()

        return unused_tokens
      
    def postprocess_result(self, results):
        
        results = results.filter(results['prediction'] != float('nan'))

        # print output
        print('Predicted Tokens:')

        return results.join(self.token_meta, 'token_id') \
                 .select("*") \
                 .orderBy('prediction', ascending = False) 
    
    def predict(self, context, model_input):
        processed_model_input = self.preprocess_input(model_input.copy())
        results = self.als.transform(processed_model_input)
        return self.postprocess_result(results)

# COMMAND ----------

from mlflow.exceptions import MlflowException

model_path = f"{workingDir}/add_n_model2"
add5_model = ALS_predicion(n=5)

dbutils.fs.rm(model_path, True) # Allows you to rerun the code multiple times

mlflow.pyfunc.save_model(path=model_path.replace("dbfs:", "/dbfs"), python_model=add5_model)

# COMMAND ----------


