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
--select count(*) from ethereumetl.transactions;

-- COMMAND ----------

--select count(*) from ethereumetl.tokens;
--select count(*) from ethereumetl.token_transfers;
--select count(*) from ethereumetl.token_prices_usd;
--select count(*) from ethereumetl.receipts;
--select count(*) from ethereumetl.logs;
--select count(*) from ethereumetl.contracts;
--select count(*) from ethereumetl.blocks;

-- COMMAND ----------

drop table if exists g10_db.token_transfers_erc20

-- COMMAND ----------

-- MAGIC %python
-- MAGIC spark.sql("create table gxx_db.token_transfers_erc20 LIKE ethereumetl.token_transfers")

-- COMMAND ----------

SET hive.exec.dynamic.partition = true;
SET hive.exec.dynamic.partition.mode = nonstrict;

-- COMMAND ----------

-- MAGIC %python
-- MAGIC spark.sql("""
-- MAGIC insert into g10_db.token_transfers_erc20
-- MAGIC partition(start_block, end_block) 
-- MAGIC select * 
-- MAGIC from ethereumetl.token_transfers
-- MAGIC where token_address in (select address from ethereumetl.silver_contracts where is_erc20 = True)
-- MAGIC """)

-- COMMAND ----------

-- MAGIC %python 
-- MAGIC 
-- MAGIC transactions_df.select(min("transaction_index"),max("transaction_index")).show()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: What is the maximum block number and date of block in the database

-- COMMAND ----------

-- TBD
SELECT number AS Maximum_Block_Number, CAST(to_timestamp(timestamp) AS DATE) AS Date_of_block
FROM ethereumetl.blocks
WHERE number= (SELECT MAX(number) FROM ethereumetl.blocks);

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
AS select ttr.* from g10_db.token_transfers ttr inner join g10_db.tokens tok on (ttr.token_address=tok.address) limit 10000;

-- COMMAND ----------

select * from g10_db.erc20_token_transfers_limit_10000;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: At what block did the first ERC20 transfer happen?

-- COMMAND ----------

-- TBD
-- select tr.* from 
-- (select token_address,block_number from g10_db.token_transfers) tr inner join 
-- (select * from g10_db.tokens) tok on (tr.token_address=tok.address) order by block_number;
select min(block_number) as block_num from g10_db.token_transfers_erc20



-- COMMAND ----------

-- MAGIC %python
-- MAGIC spark.sql("drop table if exists g10_db.silver_contracts_erc20_1")
-- MAGIC 
-- MAGIC spark.sql("""
-- MAGIC create table g10_db.silver_contracts_erc20_1 as
-- MAGIC select *
-- MAGIC from ethereumetl.silver_contracts where is_erc20 = True
-- MAGIC """)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: How many ERC20 compatible contracts are there on the blockchain?

-- COMMAND ----------

-- TBD
--select tk_prices.* from g10_db.token_prices_usd tk_prices inner join tokens tok on (tk_prices.symbol = tok.symbol);
select count(distinct address) from g10_db.silver_contracts_erc20_1


-- COMMAND ----------

--select count(*) from (select distinct tok.symbol from tokens tok) s; 

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ## Q: What percentage of transactions are calls to contracts

-- COMMAND ----------

-- TBD
--select count(*) as tran,(select count(*) from g10_db.transactions tr inner join g10_db.contracts con on (tr.to_address = con.address)) as ctc from g10_db.transactions ;  

select 
round((sum(cast((sc.address is not null) as integer))/count(1))*100, 2) as percentage_value
from ethereumetl.transactions t
left join ethereumetl.silver_contracts sc on t.to_address = sc.address


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: What are the top 100 tokens based on transfer count?

-- COMMAND ----------

-- TBD
--(select ttr.token_address,count(*) from g10_db.token_transfers ttr group by ttr.token_address order by count(*) desc limit 100;
select
token_address, count(distinct transaction_hash) as transfer_count
from ethereumetl.token_transfers
group by token_address
order by transfer_count desc
limit 100

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: What fraction of ERC-20 transfers are sent to new addresses
-- MAGIC (i.e. addresses that have a transfer count of 1 meaning there are no other transfers to this address for this token this is the first)

-- COMMAND ----------

-- TBD
--select erc20_tr.to_address,count(*) from g10_db.erc20_token_transfers erc20_tr group by erc20_tr.to_address having count(*)=1;
select round((count(distinct concat(token_address, to_address))/count(1))*100, 2) as percentage_value
from g10_db.token_transfers_erc20

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: In what order are transactions included in a block in relation to their gas price?
-- MAGIC - hint: find a block with multiple transactions 

-- COMMAND ----------

-- TBD
select block_number, transaction_index, gas_price from g10_db.transactions where block_number in (select block_number from g10_db.transactions group by block_number having count(*)>1) order by gas_price;
--select block_number, transaction_index, gas_price, hash
--from ethereumetl.transactions
--where start_block >= 14030000 and block_number in (14030400, 14030401, 14030300)


-- COMMAND ----------


0xefd8af0b52f0fa6522ea58222af52ccbec9ab55460c9e6c3c553569e2e2dc8be

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: What was the highest transaction throughput in transactions per second?
-- MAGIC hint: assume 15 second block time

-- COMMAND ----------

-- TBD
select max(transaction_count)/15 as max_throughput from ethereumetl.blocks


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: What is the total Ether volume?
-- MAGIC Note: 1x10^18 wei to 1 eth and value in the transaction table is in wei

-- COMMAND ----------

-- TBD
--select sum(value) from g10_db.transactions;
select sum(value)/power(10,18) as total_ether_volume from ethereumetl.transactions


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: What is the total gas used in all transactions?

-- COMMAND ----------

-- TBD
--select sum(gas) from g10_db.transactions;
select sum(gas_used) as total_gas_used from ethereumetl.receipts


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: Maximum ERC-20 transfers in a single transaction

-- COMMAND ----------

-- TBD
--select max(value) from g10_db.erc20_token_transfers;
select max(transfer_count)
from
(
    select transaction_hash, count(*) as transfer_count
    from g10_db.token_transfers_erc20
    group by transaction_hash
)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Q: Token balance for any address on any date?

-- COMMAND ----------

-- MAGIC %python
-- MAGIC 
-- MAGIC df_q12_final = spark.sql(""" select token_address,sum(case when from_address = '${wallet.address}' then -1*value else value end) as balance 
-- MAGIC from g10_db.token_transfers_erc20 t 
-- MAGIC inner join ethereumetl.blocks b on b.start_block = t.start_block and b.end_block = t.end_block and b.number = t.block_number
-- MAGIC and to_date(cast(b.`timestamp` as TIMESTAMP)) <= '${start.date}' and (from_address = '${wallet.address}' or to_address = '${wallet.address}')
-- MAGIC group by token_address order by balance desc """)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Viz the transaction count over time (network use)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df_q13 = spark.sql("""
-- MAGIC select to_date(CAST(`timestamp` as timestamp)) as `date`,
-- MAGIC sum(transaction_count) as transaction_count_in_a_day
-- MAGIC from ethereumetl.blocks
-- MAGIC where year(to_date(CAST(`timestamp` as timestamp))) >= 2015
-- MAGIC group by `date`
-- MAGIC order by `date`
-- MAGIC  """)
-- MAGIC  
-- MAGIC display(df_q13)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Viz ERC-20 transfer count over time
-- MAGIC interesting note: https://blog.ins.world/insp-ins-promo-token-mixup-clarified-d67ef20876a3

-- COMMAND ----------

-- MAGIC 
-- MAGIC %python
-- MAGIC df_q14 = spark.sql("""
-- MAGIC select `date`, sum(transfer_count) as transfer_count_in_a_day
-- MAGIC from(
-- MAGIC     select start_block, end_block, number, to_date(CAST(`timestamp` AS timestamp)) as `date`
-- MAGIC     from ethereumetl.blocks 
-- MAGIC  ) b 
-- MAGIC  left join
-- MAGIC  (
-- MAGIC     select start_block, end_block, block_number, count(distinct transaction_hash) as transfer_count
-- MAGIC     from g10_db.token_transfers_erc20
-- MAGIC     group by start_block, end_block, block_number
-- MAGIC  ) tt on b.start_block = tt.start_block and b.end_block = tt.end_block and b.number = tt.block_number
-- MAGIC  where year(`date`) >= 2015
-- MAGIC  group by `date`
-- MAGIC  order by `date` 
-- MAGIC  """)
-- MAGIC 
-- MAGIC display(df_q14)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Return Success
-- MAGIC dbutils.notebook.exit(json.dumps({"exit_code": "OK"}))
