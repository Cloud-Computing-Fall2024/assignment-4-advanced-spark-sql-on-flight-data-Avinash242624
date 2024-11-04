from pyspark.sql import SparkSession
from pyspark.sql.functions import abs, unix_timestamp, stddev, when, count, avg, hour
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf

# Initialize Spark session
spark = SparkSession.builder.appName("Flight Data Analysis").getOrCreate()

# Load the datasets
flights = spark.read.csv("flights.csv", header=True, inferSchema=True)
airports = spark.read.csv("airports.csv", header=True, inferSchema=True)
carriers = spark.read.csv("carriers.csv", header=True, inferSchema=True)

# Calculate dep_delay as the difference between ActualDeparture and ScheduledDeparture
flights = flights.withColumn(
    "dep_delay",
    (unix_timestamp("ActualDeparture") - unix_timestamp("ScheduledDeparture")) / 60  # Convert seconds to minutes
)

# Task 1: Flights with the Largest Discrepancy Between Scheduled and Actual Travel Time
flights.createOrReplaceTempView("flights")
task1_query = """
SELECT FlightNum, Origin, Destination,
       ABS(UNIX_TIMESTAMP(ActualArrival) - UNIX_TIMESTAMP(ScheduledArrival)) AS discrepancy
FROM flights
ORDER BY discrepancy DESC
LIMIT 10
"""
task1_result = spark.sql(task1_query)
task1_result.write.mode("overwrite").csv("output/task1_largest_discrepancy.csv", header=True)

# Task 2: Most Consistently On-Time Airlines Using Standard Deviation
task2_query = """
SELECT CarrierCode, stddev(dep_delay) AS dep_delay_stddev
FROM flights
GROUP BY CarrierCode
HAVING COUNT(*) > 100
ORDER BY dep_delay_stddev ASC
"""
task2_result = spark.sql(task2_query)
task2_result.write.mode("overwrite").csv("output/task2_consistent_airlines.csv", header=True)

# Task 3: Origin-Destination Pairs with the Highest Percentage of Canceled Flights
# Ensure `canceled` column exists and is properly represented (you may need to adapt based on actual column names)
flights = flights.withColumn("canceled", when(flights["ActualDeparture"].isNull(), 1).otherwise(0))
airports.createOrReplaceTempView("airports")
flights.createOrReplaceTempView("flights")

task3_query = """
SELECT f.Origin, f.Destination,
       a1.AirportName AS origin_name, a2.AirportName AS destination_name,
       (SUM(CASE WHEN f.canceled = 1 THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS cancellation_rate
FROM flights f
JOIN airports a1 ON f.Origin = a1.AirportCode
JOIN airports a2 ON f.Destination = a2.AirportCode
GROUP BY f.Origin, f.Destination, a1.AirportName, a2.AirportName
ORDER BY cancellation_rate DESC
LIMIT 10
"""
task3_result = spark.sql(task3_query)
task3_result.write.mode("overwrite").csv("output/task3_canceled_routes.csv", header=True)

# Task 4: Carrier Performance Based on Time of Day
def time_of_day(dep_hour):
    if 5 <= dep_hour < 12:
        return "morning"
    elif 12 <= dep_hour < 17:
        return "afternoon"
    elif 17 <= dep_hour < 21:
        return "evening"
    else:
        return "night"

time_of_day_udf = udf(time_of_day, StringType())

# Apply the UDF to classify time of day
flights = flights.withColumn("time_of_day", time_of_day_udf(hour("ScheduledDeparture")))
flights.createOrReplaceTempView("flights")

task4_query = """
SELECT CarrierCode, time_of_day, AVG(dep_delay) AS avg_dep_delay
FROM flights
GROUP BY CarrierCode, time_of_day
ORDER BY time_of_day, avg_dep_delay
"""
task4_result = spark.sql(task4_query)
task4_result.write.mode("overwrite").csv("output/task4_carrier_performance_time_of_day.csv", header=True)

# Stop Spark session
spark.stop()
