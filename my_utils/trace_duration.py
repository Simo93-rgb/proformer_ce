import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
import datetime

# Path to the input XES file
input_file = "../data/ALL_20DRG_2022_2023_CLASS_no_doppi_no_outlier.xes"

# Load the event log
log = xes_importer.apply(input_file)

# Iterate over traces to calculate durations and add attributes
for trace in log:
    print ("Elaborating trace: ", trace.attributes["concept:name"])
    # Get the start and end timestamps of the trace
    start_time = trace[0]["time:timestamp"]
    end_time = trace[-1]["time:timestamp"]

    # Calculate duration in days
    duration = (end_time - start_time).days
    # Add the trace_duration attribute
    trace.attributes["trace_duration"] = duration

    # Add the class_trace attribute
    trace.attributes["class_trace"] = 0 if duration <= 20 else 1

# Path to the output XES file
output_file = "ALL_20DRG_2022_2023_CLASS_Duration.xes"

# Export the updated log
xes_exporter.apply(log, output_file)

print(f"Updated log exported to {output_file}")
