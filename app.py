from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()
import neo4j
# URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"

URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()
    print("Connection established.")


summary = driver.execute_query(
    "CREATE (:Person {name: $name})",
    name="Alice",
    database_="neo4j",
).summary

print("Created {nodes_created} nodes in {time} ms.".format(
    nodes_created=summary.counters.nodes_created,
    time=summary.result_available_after
))

records, summary, keys = driver.execute_query(
    "MATCH (p:Person) RETURN p.name AS name",
    database_="neo4j",
)

# Loop through results and do something with them
for record in records:
    print(record.data())  # obtain record as dict

# Summary information
print("The query `{query}` returned {records_count} records in {time} ms.".format(
    query=summary.query, records_count=len(records),
    time=summary.result_available_after
))


records, summary, keys = driver.execute_query("""
    MATCH (p:Person {name: $name})
    SET p.age = $age
    """, name="Alice", age=42,
    database_="neo4j",
)
print(f"Query counters: {summary.counters}.")


records, summary, keys = driver.execute_query("""
    MATCH (alice:Person {name: $name})
    MATCH (bob:Person {name: $friend})
    CREATE (alice)-[:KNOWS]->(bob)
    """, name="Alice", friend="Bob",
    database_="neo4j",
)
print(f"Query counters: {summary.counters}.")

ecords, summary, keys = driver.execute_query("""
    MATCH (p:Person {name: $name})
    DETACH DELETE p
    """, name="Alice",
    database_="neo4j",
)
print(f"Query counters: {summary.counters}.")

driver.execute_query(
    "MERGE (:Person {name: $name})",
    name="Alice", age=42,
    database_="neo4j",
)

parameters = {
    "name": "Alice",
    "age": 42
}
driver.execute_query(
    "MERGE (:Person {name: $name})",
    parameters_=parameters,
    database_="neo4j",
)

driver.execute_query(
    "MATCH (p:Person) RETURN p.name",
    database_="neo4j",
)

driver.execute_query(
    "MATCH (p:Person) RETURN p.name",
    routing_="r",  # short for neo4j.RoutingControl.READ
    database_="neo4j",
)

driver.execute_query(
    "MATCH (p:Person) RETURN p.name",
    auth_=("somebody_else", "their_password"),
    database_="neo4j",
)

driver.execute_query(
    "MATCH (p:Person) RETURN p.name",
    impersonated_user_="somebody_else",
    database_="neo4j",
)

ecords, summary, keys = driver.execute_query(
    "MATCH (a:Person) RETURN a.name AS name, a.age AS age",
    database_="neo4j",
)
for person in records:
    print(person)
    # person["name"] or person["age"] are also valid

# Some summary information
print("Query `{query}` returned {records_count} records in {time} ms.".format(
    query=summary.query, records_count=len(records),
    time=summary.result_available_after
))

print(f"Available keys are {keys}")  # ['name', 'age']

pandas_df = driver.execute_query(
    "UNWIND range(1, 10) AS n RETURN n, n+1 AS m",
    database_="neo4j",
    result_transformer_=neo4j.Result.to_df
)
print(type(pandas_df))  # <class 'pandas.core.frame.DataFrame'>

