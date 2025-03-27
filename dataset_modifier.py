import pandas as pd


def aggregate_case_details(input_csv, output_csv):
    # Read the CSV file with all entries as strings
    df = pd.read_csv(input_csv, dtype=str)

    # Convert the timestamp column to datetime for proper sorting (adjust the format if needed)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Sort by case_id and timestamp so that aggregated timestamps and actions remain in order
    df_sorted = df.sort_values(by=['case_id', 'timestamp'])

    # Aggregate timestamps and activity columns; for class, select the first non-null instance
    aggregated = df_sorted.groupby('case_id').agg({
        'timestamp': lambda ts: ','.join(ts.dropna().apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S%z'))),
        'activity': lambda acts: ','.join(acts),
    }).reset_index()

    # Save the aggregated data to CSV
    aggregated.to_csv(output_csv, index=False)


if __name__ == '__main__':
    input_csv = r'data/ALL_20DRG_2022_2023_CLASS_Duration_ricovero_dimissioni_LAST_17Jan2025_padded.csv'
    output_csv = r'data/aggregated_case_details.csv'
    aggregate_case_details(input_csv, output_csv)
    print(f"Aggregated file saved to {output_csv}")