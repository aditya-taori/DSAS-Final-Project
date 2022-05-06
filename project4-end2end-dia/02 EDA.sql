-- Databricks notebook source
-- MAGIC %md
-- MAGIC ## Ethereum Blockchain Data Analysis - <a href=https://github.com/blockchain-etl/ethereum-etl-airflow/tree/master/dags/resources/stages/raw/schemas>Table Schemas</a>
-- MAGIC - **Transactions** - Each block in the blockchain is composed of zero or more transactions. Each transaction has a source address, a target address, an amount of Ether transferred, and an array of input bytes. This table contains a set of all transactions from all blocks, and contains a block identifier to get associated block-specific information associated with each transaction.
-- MAGIC - **Blocks** - The Ethereum blockchain is composed of a series of blocks. This table contains a set of all blocks in the blockchain and their attributes.
-- MAGIC - **Receipts** - the cost of gas for specific transactions.
-- MAGIC - **Traces** - The trace module is for getting a deeper insight into transaction processing. Traces exported using <a href=https://openethereum.github.io/JSONRPC-trace-module.html>Parity trace module</a>
-- MAGIC - **Tokens** - Token data including contract address and symbol information.
-- MAGIC - **Token Transfers** - The most popular type of transaction on the Ethereum blockchain invokes a contract of type ERC20 to perform a transfer operation, moving some number of tokens from one 20-byte address to another 20-byte address. This table contains the subset of those transactions and has further processed and denormalized the data to make it easier to consume for analysis of token transfer events.
-- MAGIC - **Contracts** - Some transactions create smart contracts from their input bytes, and this smart contract is stored at a particular 20-byte address. This table contains a subset of Ethereum addresses that contain contract byte-code, as well as some basic analysis of that byte-code. 
-- MAGIC - **Logs** - Similar to the token_transfers table, the logs table contains data for smart contract events. However, it contains all log data, not only ERC20 token transfers. This table is generally useful for reporting on any logged event type on the Ethereum blockchain.
-- MAGIC 
-- MAGIC ### Rubric for this module
-- MAGIC Answer the quetions listed below.

-- COMMAND ----------

-- MAGIC %run ./includes/utilities

-- COMMAND ----------

-- MAGIC %run ./includes/configuration

-- COMMAND ----------

-- MAGIC %python 
-- MAGIC token_transfer = spark.table("g10_db.silver_tokenmatrix")
-- MAGIC token_transfer.show(3)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Grab the global variables
-- MAGIC wallet_address,start_date = Utils.create_widgets()
-- MAGIC print(wallet_address,start_date)
-- MAGIC spark.conf.set('wallet.address',wallet_address)
-- MAGIC spark.conf.set('start.date',start_date)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC 
-- MAGIC blocks_df = spark.sql("select * from ethereumetl.blocks")
-- MAGIC contracts_df = spark.sql("select * from ethereumetl.contracts")
-- MAGIC logs_df = spark.sql("select * from ethereumetl.logs")
-- MAGIC receipts_df = spark.sql("select * from ethereumetl.receipts")
-- MAGIC token_prices_usd_df = spark.sql("select * from ethereumetl.token_prices_usd")
-- MAGIC token_transfers_df = spark.sql("select * from ethereumetl.token_transfers")
-- MAGIC token_df = spark.sql("select * from ethereumetl.tokens")
-- MAGIC transactions_df = spark.sql("select * from ethereumetl.transactions")
-- MAGIC 
-- MAGIC 
-- MAGIC display(blocks_df.select("*"))

-- COMMAND ----------

--select min(transaction_index),max(transaction_index) from ethereumetl.transactions;
select count(*) from ethereumetl.transactions;

-- COMMAND ----------

--select count(*) from ethereumetl.tokens;
--select count(*) from ethereumetl.token_transfers;
--select count(*) from ethereumetl.token_prices_usd;
--select count(*) from ethereumetl.receipts;
--select count(*) from ethereumetl.logs;
--select count(*) from ethereumetl.contracts;
select count(*) from ethereumetl.blocks;

-- COMMAND ----------



-- COMMAND ----------

-- MAGIC %python 
-- MAGIC 
-- MAGIC transactions_df.select(min("transaction_index"),max("transaction_index")).show()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: What is the maximum block number and date of block in the database

-- COMMAND ----------

-- TBD
select number,to_timestamp(timestamp) from ethereumetl.blocks where number= (select max(number) from ethereumetl.blocks);

-- COMMAND ----------

CREATE OR REPLACE VIEW g10_db.blocks_date 
COMMENT 'View of Blocks with date'
AS SELECT *,to_timestamp(timestamp) AS creation_date FROM ethereumetl.blocks;


-- COMMAND ----------

CREATE OR REPLACE VIEW g10_db.blocks 
COMMENT 'Copy of Blocks in g10db'
AS SELECT * FROM ethereumetl.blocks;

CREATE OR REPLACE VIEW g10_db.tokens 
COMMENT 'Copy of Tokens in g10db'
AS SELECT * FROM ethereumetl.tokens;

CREATE OR REPLACE VIEW g10_db.token_transfers 
COMMENT 'Copy of Token Transfers in g10db'
AS SELECT * FROM ethereumetl.token_transfers;

CREATE OR REPLACE VIEW g10_db.token_prices_usd 
COMMENT 'Copy of Token Prices USD in g10db'
AS SELECT * FROM ethereumetl.token_prices_usd;

CREATE OR REPLACE VIEW g10_db.receipts 
COMMENT 'Copy of Receipts in g10db'
AS SELECT * FROM ethereumetl.receipts;

CREATE OR REPLACE VIEW g10_db.logs 
COMMENT 'Copy of Logs in g10db'
AS SELECT * FROM ethereumetl.logs;

CREATE OR REPLACE VIEW g10_db.contracts 
COMMENT 'Copy of Contracts in g10db'
AS SELECT * FROM ethereumetl.contracts;

CREATE OR REPLACE VIEW g10_db.transactions 
COMMENT 'Copy of Contracts in g10db'
AS SELECT * FROM ethereumetl.transactions;

-- COMMAND ----------

CREATE OR REPLACE VIEW g10_db.erc20_token_transfers_limit_10000 
COMMENT 'Copy of ERC 20 token transfers'
AS select ttr.token_address,ttr.from_address from g10_db.token_transfers ttr inner join g10_db.tokens tok on (ttr.token_address=tok.address) limit 10;

-- COMMAND ----------

select ttr.token_address,ttr.from_address from g10_db.token_transfers ttr inner join g10_db.tokens tok on (ttr.token_address=tok.address) limit 10;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: At what block did the first ERC20 transfer happen?

-- COMMAND ----------

-- TBD
select tr.* from 
   (select token_address,from_address,block_number from g10_db.token_transfers) tr inner join 
      (select * from g10_db.tokens) tok on (tr.token_address=tok.address) order by block_number;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: How many ERC20 compatible contracts are there on the blockchain?

-- COMMAND ----------

-- TBD
select tk_prices.* from g10_db.token_prices_usd tk_prices inner join tokens tok on (tk_prices.symbol = tok.symbol);

-- COMMAND ----------

select count(*) from (select distinct tok.symbol from tokens tok) s; 

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ## Q: What percentage of transactions are calls to contracts

-- COMMAND ----------

-- TBD
select count(*) as tran,(select count(*) from g10_db.transactions tr inner join g10_db.contracts con on (tr.to_address = con.address)) as ctc from g10_db.transactions ;  

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: What are the top 100 tokens based on transfer count?

-- COMMAND ----------

-- TBD
(select ttr.token_address,count(*) from g10_db.token_transfers ttr group by ttr.token_address order by count(*) desc limit 100;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: What fraction of ERC-20 transfers are sent to new addresses
-- MAGIC (i.e. addresses that have a transfer count of 1 meaning there are no other transfers to this address for this token this is the first)

-- COMMAND ----------

-- TBD
select erc20_tr.to_address,count(*) from g10_db.erc20_token_transfers erc20_tr group by erc20_tr.to_address having count(*)=1;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: In what order are transactions included in a block in relation to their gas price?
-- MAGIC - hint: find a block with multiple transactions 

-- COMMAND ----------

-- TBD
select * from g10_db.transactions where block_number in (select block_number from g10_db.transactions group by block_number having count(*)>1) order by gas_price;

-- COMMAND ----------


0xefd8af0b52f0fa6522ea58222af52ccbec9ab55460c9e6c3c553569e2e2dc8be

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: What was the highest transaction throughput in transactions per second?
-- MAGIC hint: assume 15 second block time

-- COMMAND ----------

-- TBD
select MAX(transaction_count)/15 from blocks

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: What is the total Ether volume?
-- MAGIC Note: 1x10^18 wei to 1 eth and value in the transaction table is in wei

-- COMMAND ----------

-- TBD
select sum(value) from g10_db.transactions;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: What is the total gas used in all transactions?

-- COMMAND ----------

-- TBD
select sum(gas) from g10_db.transactions;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: Maximum ERC-20 transfers in a single transaction

-- COMMAND ----------

-- TBD
select max(value) from g10_db.erc20_token_transfers;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: Token balance for any address on any date?

-- COMMAND ----------

-- TBD
select token_address, date(cast(blocks.timestamp as TIMESTAMP)) as forDate,
sum(case when from_address = token_address then value*-1 else value end)
from token_transfers join blocks on token_transfers.block_number = blocks.number
where from_address = token_address or to_address = token_address 
group by forDate, token_address
order by forDate

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Viz the transaction count over time (network use)

-- COMMAND ----------

-- TBD
select sum(transaction_count) as transaction_count, cast(timestamp as TIMESTAMP) from blocks group by cast(timestamp as TIMESTAMP)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Viz ERC-20 transfer count over time
-- MAGIC interesting note: https://blog.ins.world/insp-ins-promo-token-mixup-clarified-d67ef20876a3

-- COMMAND ----------

-- TBD
select sum(transfer_count) as tranfer_count, cast(timestamp as TIMESTAMP) 
from blocks join (select block_number, count(transaction_hash) as transfer_count from token_transfers group by block_number) b on blocks.number=b.block_number
group by cast(timestamp as TIMESTAMP)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Return Success
-- MAGIC dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
