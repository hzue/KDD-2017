import util
import pandas as pd
import numpy as np

df = util.read_conclusion_file("../res/conclusion/training_20min_avg_travel_time.csv")
df = df.head(20)
df = df.drop(['end', 'intersection_id'], axis=1)
s = df.set_index('from')
sagg = s.rolling('10D').avg_travel_time.agg(['sum', 'count']).rename(columns=str.title)
agged = df.join(sagg, on='from')
print(agged)
df = df.assign(before_1hr_mean=agged.eval('(Sum - avg_travel_time) / (Count - 1)'))
# print(df.to_string())
