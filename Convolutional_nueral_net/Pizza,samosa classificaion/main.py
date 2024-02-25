import pyspark as ps
from pyspark.sql import SparkContext

a = list(range(1,10))
print(a)

sc = SparkContext("local", "First App")

D1 = sc.parallalize(a)
print(D1.colect())

print(D1)