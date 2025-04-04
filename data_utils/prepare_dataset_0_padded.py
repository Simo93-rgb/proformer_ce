import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.obj import EventLog, EventStream, Trace, Event
from datetime import datetime, time, timedelta
import pandas as pd 

# Path to the input XES file
input_file = "data\ALL_20DRG_2022_2023_CLASS_Duration_ricovero_dimissioni_LAST_17Jan2025.xes"

# Load the event log
log = xes_importer.apply(input_file)

# Collecting statistics...

# maxlength= 0
# totalcount= 0
# for trace in log:
#     totalcount+=1
#     currentlength= 0
#     for event in trace:
#         currentlength+=1
#
#     print ("Collecting trace: ", trace.attributes["concept:name"], "  -> trace length: ", currentlength)
#     if (currentlength > maxlength): maxlength= currentlength


# Adding -padded missing actions...

totalcount= 0
for trace in log:
    totalcount+=1
    print ("Elaborating trace: ", trace.attributes["concept:name"])
    # currentlength= 0
    # for event in trace:
    #     currentlength+=1
    #
    # # Performing padding
    dt= datetime.now()
    # t= 0
    # for i in range (currentlength, maxlength):
    #         event = Event()
    #         event["concept:name"] = "PAD"
    #         event["time:timestamp"] = pd.to_datetime(dt  + timedelta(seconds= t * 60)) #important to convert to pd.timestamp to parse into ProM
    #         trace.append(event)
    #         t+=1
            
    # Adding final class event
    event = Event()
    event["concept:name"] = "class_" + str(trace.attributes["class"])
    event["time:timestamp"] = pd.to_datetime(dt) #important to convert to pd.timestamp to parse into ProM
    trace.append(event)
    


# Verifying statistics...

maxlength= 0
totalcount= 0
for trace in log:
    totalcount+=1
    currentlength= 0
    for event in trace:
        currentlength+=1
    
    print ("Verifying trace: ", trace.attributes["concept:name"], "  -> trace length: ", currentlength)
    if (currentlength > maxlength): maxlength= currentlength


    
print ("\n\n Finished: maximum trace length: ", maxlength)
print ("Total traces: ", totalcount)
print ("\n Exporting to output log...")

# Path to the output XES file
output_file = "data\ALL_20DRG_2022_2023_CLASS_Duration_ricovero_dimissioni_LAST_17Jan2025_padded.xes"

# Export the updated log
xes_exporter.apply(log, output_file)

print(f"Updated log exported to {output_file}")

