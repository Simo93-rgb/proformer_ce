import pm4py

log = pm4py.read_xes('data\ALL_20DRG_2022_2023_CLASS_Duration_ricovero_dimissioni_LAST_17Jan2025_padded.xes')
#log = pm4py.read_xes('data/ALL_20DRG_2022_2023_CLASS_no_doppi.xes')
print (log)

df = pm4py.convert_to_dataframe(log)
df = df.sort_values('time:timestamp')

cols = {"case:concept:name": "case_id", "concept:name": "activity", "time:timestamp": "timestamp"}
df = df.rename(columns=cols)
df = df[cols.values()]

df.to_csv('data\ALL_20DRG_2022_2023_CLASS_Duration_ricovero_dimissioni_LAST_17Jan2025_padded.csv', index=False)
#df.to_csv('data/ALL_20DRG_2022_2023_CLASS_no_doppi.csv', index=False)

print (df.groupby("case_id").apply(lambda x : len(x.activity)).max())

print (df.activity.unique())