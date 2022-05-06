# Databricks notebook source
# MAGIC %md
# MAGIC ## Ethereum Blockchain Data Analysis - <a href=https://github.com/blockchain-etl/ethereum-etl-airflow/tree/master/dags/resources/stages/raw/schemas>Table Schemas</a>
# MAGIC - **Transactions** - Each block in the blockchain is composed of zero or more transactions. Each transaction has a source address, a target address, an amount of Ether transferred, and an array of input bytes. This table contains a set of all transactions from all blocks, and contains a block identifier to get associated block-specific information associated with each transaction.
# MAGIC - **Blocks** - The Ethereum blockchain is composed of a series of blocks. This table contains a set of all blocks in the blockchain and their attributes.
# MAGIC - **Receipts** - the cost of gas for specific transactions.
# MAGIC - **Traces** - The trace module is for getting a deeper insight into transaction processing. Traces exported using <a href=https://openethereum.github.io/JSONRPC-trace-module.html>Parity trace module</a>
# MAGIC - **Tokens** - Token data including contract address and symbol information.
# MAGIC - **Token Transfers** - The most popular type of transaction on the Ethereum blockchain invokes a contract of type ERC20 to perform a transfer operation, moving some number of tokens from one 20-byte address to another 20-byte address. This table contains the subset of those transactions and has further processed and denormalized the data to make it easier to consume for analysis of token transfer events.
# MAGIC - **Contracts** - Some transactions create smart contracts from their input bytes, and this smart contract is stored at a particular 20-byte address. This table contains a subset of Ethereum addresses that contain contract byte-code, as well as some basic analysis of that byte-code. 
# MAGIC - **Logs** - Similar to the token_transfers table, the logs table contains data for smart contract events. However, it contains all log data, not only ERC20 token transfers. This table is generally useful for reporting on any logged event type on the Ethereum blockchain.
# MAGIC 
# MAGIC In Addition, there is a price feed that changes daily (noon) that is in the **token_prices_usd** table
# MAGIC 
# MAGIC ### Rubric for this module
# MAGIC - Transform the needed information in ethereumetl database into the silver delta table needed by your modeling module
# MAGIC - Clearly document using the notation from [lecture](https://learn-us-east-1-prod-fleet02-xythos.content.blackboardcdn.com/5fdd9eaf5f408/8720758?X-Blackboard-Expiration=1650142800000&X-Blackboard-Signature=h%2FZwerNOQMWwPxvtdvr%2FmnTtTlgRvYSRhrDqlEhPS1w%3D&X-Blackboard-Client-Id=152571&response-cache-control=private%2C%20max-age%3D21600&response-content-disposition=inline%3B%20filename%2A%3DUTF-8%27%27Delta%2520Lake%2520Hands%2520On%2520-%2520Introduction%2520Lecture%25204.pdf&response-content-type=application%2Fpdf&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEAAaCXVzLWVhc3QtMSJHMEUCIQDEC48E90xPbpKjvru3nmnTlrRjfSYLpm0weWYSe6yIwwIgJb5RG3yM29XgiM%2BP1fKh%2Bi88nvYD9kJNoBNtbPHvNfAqgwQIqP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARACGgw2MzU1Njc5MjQxODMiDM%2BMXZJ%2BnzG25TzIYCrXAznC%2BAwJP2ee6jaZYITTq07VKW61Y%2Fn10a6V%2FntRiWEXW7LLNftH37h8L5XBsIueV4F4AhT%2Fv6FVmxJwf0KUAJ6Z1LTVpQ0HSIbvwsLHm6Ld8kB6nCm4Ea9hveD9FuaZrgqWEyJgSX7O0bKIof%2FPihEy5xp3D329FR3tpue8vfPfHId2WFTiESp0z0XCu6alefOcw5rxYtI%2Bz9s%2FaJ9OI%2BCVVQ6oBS8tqW7bA7hVbe2ILu0HLJXA2rUSJ0lCF%2B052ScNT7zSV%2FB3s%2FViRS2l1OuThecnoaAJzATzBsm7SpEKmbuJkTLNRN0JD4Y8YrzrB8Ezn%2F5elllu5OsAlg4JasuAh5pPq42BY7VSKL9VK6PxHZ5%2BPQjcoW0ccBdR%2Bvhva13cdFDzW193jAaE1fzn61KW7dKdjva%2BtFYUy6vGlvY4XwTlrUbtSoGE3Gr9cdyCnWM5RMoU0NSqwkucNS%2F6RHZzGlItKf0iPIZXT3zWdUZxhcGuX%2FIIA3DR72srAJznDKj%2FINdUZ2s8p2N2u8UMGW7PiwamRKHtE1q7KDKj0RZfHsIwRCr4ZCIGASw3iQ%2FDuGrHapdJizHvrFMvjbT4ilCquhz4FnS5oSVqpr0TZvDvlGgUGdUI4DCdvOuSBjqlAVCEvFuQakCILbJ6w8WStnBx1BDSsbowIYaGgH0RGc%2B1ukFS4op7aqVyLdK5m6ywLfoFGwtYa5G1P6f3wvVEJO3vyUV16m0QjrFSdaD3Pd49H2yB4SFVu9fgHpdarvXm06kgvX10IfwxTfmYn%2FhTMus0bpXRAswklk2fxJeWNlQF%2FqxEmgQ6j4X6Q8blSAnUD1E8h%2FBMeSz%2F5ycm7aZnkN6h0xkkqQ%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20220416T150000Z&X-Amz-SignedHeaders=host&X-Amz-Expires=21600&X-Amz-Credential=ASIAZH6WM4PLXLBTPKO4%2F20220416%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=321103582bd509ccadb1ed33d679da5ca312f19bcf887b7d63fbbb03babae64c) how your pipeline is structured.
# MAGIC - Your pipeline should be immutable
# MAGIC - Use the starting date widget to limit how much of the historic data in ethereumetl database that your pipeline processes.

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# Grab the global variables
wallet_address,start_date = Utils.create_widgets()
print(wallet_address,start_date)
spark.conf.set('wallet.address',wallet_address)
spark.conf.set('start.date',start_date)

# COMMAND ----------

# MAGIC %md
# MAGIC ## YOUR SOLUTION STARTS HERE...

# COMMAND ----------

#01-ETL
#Set shuffle partitions:
sqlContext.setConf('spark.sql.shuffle.partitions', 'auto')

# COMMAND ----------

#Blocks contain transactions and some important data such as previous hash that ensures immutability and security in the blockchain network. Each block stores a previous hash sequentially so it is almost infeasible to reverse and tamper data.
#nonce is a property of transaction originating address
#gas refers to the cost necessary to perform a transaction on the network
blocks = spark.table("ethereumetl.blocks")
#contracts = spark.table("ethereumetl.contracts")
#logs = spark.table("ethereumetl.logs")
#receipts = spark.table("ethereumetl.receipts")
#blocks_date = spark.table("g10_db.blocks_date")
#contracts = spark.table("ethereumetl.contracts")
#logs = spark.table("ethereumetl.logs")
#receipts = spark.table("ethereumetl.receipts")
silver_contracts = spark.table("ethereumetl.silver_contracts")
token_prices_usd = spark.table("g10_db.token_prices_usd")
token_transfers = spark.table("g10_db.token_transfers")
tokens = spark.table("g10_db.tokens")
transactions = spark.table("g10_db.transactions")
erc20_token_transfers = spark.table("g10_db.erc20_token_transfers")


# COMMAND ----------

from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer
token_prices_usd_unique = token_prices_usd.dropDuplicates(["contract_address"]) #Drop repeat contract address to get unique tokens
token_prices_usd_unique = token_prices_usd_unique.filter(col("price_usd").isNotNull() & col("contract_address").isNotNull()) #Drop null price and address values
token_prices_usd_unique = token_prices_usd_unique.select(["name","contract_address","price_usd","links","image"]) #Select necessary metadata
token_prices_usd_unique = token_prices_usd_unique.withColumn("token_address",col("contract_address")).drop("contract_address") #Rename to token_address
#Index the contract_address column
stringIndexer = StringIndexer(inputCol="token_address", outputCol="token_address_id") #Use string indexer to assign integer values to the token IDs
model = stringIndexer.fit(token_prices_usd_unique)
token_prices_usd_unique = model.transform(token_prices_usd_unique)
token_prices_usd_unique = token_prices_usd_unique.withColumn("token_id", col("token_address_id").cast("int")).drop("token_address_id")
#Now we have a unique metadata table

# COMMAND ----------

#Select records only after the start date
blocks = (blocks
          .withColumn("transformed_timestamp",to_timestamp(col("timestamp")))
          .withColumn("block_number",col("number"))
          .select("block_number","transformed_timestamp")
         )
blocks = blocks.filter(blocks["transformed_timestamp"] >= (lit(start_date)))

erc20_token_transfers = erc20_token_transfers.join(blocks, "block_number","inner")

# COMMAND ----------

#Combine the pricing information with the token transfer information, left join by token transfers
#An inner join should exclude tokens in the transfer table that do not have pricing information, since we exluded that in the pricing table.
erc20_token_transfers = erc20_token_transfers.select(["token_address","from_address","value"])
silverTokenTransfers = erc20_token_transfers.join(token_prices_usd_unique, 'token_address', 'inner')
#Now multiply value*price_usd to normalize transfer values to USD
#silverTokenTransfers = silverTokenTransfers.withColumn("token_USD_value_transferred",col("value")*col("price_usd"))
silverTokenTransfers.printSchema()
#display(silverTokenTransfers)


# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id 
user_mapping = silverTokenTransfers.select("from_address").dropDuplicates().withColumn("user_id",monotonically_increasing_id().cast("int"))
user_mapping.cache()
user_mapping.count()

# COMMAND ----------

#Now groupby user_id and token_id to create user_id, token_id, # of transfers matrix to feed into Collab Filtering 
#from pyspark.sql.functions import monotonically_increasing_id 
silverTokenMatrix = silverTokenTransfers.select("from_address","token_id").groupby("from_address","token_id").count().withColumnRenamed("count","#_transfers_long").withColumn("#_transfers",col("#_transfers_long").cast("int")).drop("#_transfers_long")

silverTokenMatrix = silverTokenMatrix.join(user_mapping, "from_address")

#silverTokenMatrix = silverTokenMatrix.withColumn("user_id",monotonically_increasing_id().cast("int")) #Since the user_addresses are now unique (grouped and aggregated), just assign integers to them
silverTokenMatrix = silverTokenMatrix.select("user_id","token_id","#_transfers") #Select just the triplet values. Our implicit rating system is #_transfers for each token
silverTokenMatrix.schema['user_id'].nullable = True
silverTokenMatrix.schema['token_id'].nullable = True
silverTokenMatrix.schema['#_transfers'].nullable = True
silverTokenMatrix.cache
silverTokenMatrix.count()

# COMMAND ----------

from pyspark.sql.types import * 

#Manurally define the schema we want for the silver matrix:
expected_schema = StructType([StructField("user_id",IntegerType(), True),
                              StructField("token_id",IntegerType(), True),
                              StructField("#_transfers",IntegerType(),True)])
#Assert that it is correct
assert silverTokenMatrix.schema == expected_schema, "Schema Incorrect"
print("Assertion passed.")




# COMMAND ----------

#Repeat with the silver metadata table
from pyspark.sql.types import _parse_datatype_string

metadata_expected_schema = "name STRING, price_usd DOUBLE, links STRING, image STRING, token_address STRING, token_id INTEGER"
assert token_prices_usd_unique.schema == _parse_datatype_string(metadata_expected_schema), "Schema Incorrect"
print("Assertion passed.")

# COMMAND ----------

#Write out the Triplet Matrix
silverTokenMatrix.write.format("delta").option("mergeSchema", "true").mode("overwrite").partitionBy("token_id").saveAsTable("g10_db.silver_TokenMatrix")

# COMMAND ----------

#Write out the metadata
token_prices_usd_unique.write.format("delta").mode("overwrite").partitionBy("token_id").saveAsTable("g10_db.silverTokenMetaData")

# COMMAND ----------

#Write out user_id mappings from from_address
user_mapping.write.format("delta").mode("overwrite").saveAsTable("g10_db.silver_userid_mapping")
#OPTIMIZE silver_TokenMatrix ZORDER BY (user_id)
#VACUUM silver_tokenmatrix retain 169 hours dry run

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
