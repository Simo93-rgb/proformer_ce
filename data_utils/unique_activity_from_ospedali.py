import csv

import numpy as np

from config import DATA_DIR

from collections import defaultdict

def calculate_total_length_grouped(csv_path, group_column='case_id', data_column_index=1):
    """Reads a CSV file, groups data by a specified column,
    and calculates the total length of entries in another column within each group.

    Args:
        csv_path (str): The path to the CSV file.
        group_column (str, optional): The name of the column to group by.
                                       Defaults to 'case_id'.
        data_column_index (int, optional): The index of the column containing
                                            the strings to measure (0-based).
                                            Defaults to 1 (second column).

    Returns:
        dict: A dictionary where keys are the unique values from the
              `group_column` and values are the total length of strings
              in the `data_column_index` for that group.
              Returns an empty dictionary if the file is empty or columns are not found.
    """
    grouped_total_lengths = defaultdict(int)
    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if group_column not in reader.fieldnames:
                print(f"Error: Grouping column '{group_column}' not found in CSV header.")
                return {}
            if data_column_index >= len(reader.fieldnames):
                print(f"Error: Data column at index {data_column_index} not found in CSV header.")
                return {}

            data_column_name = reader.fieldnames[data_column_index]

            for row in reader:
                group_id = row[group_column]
                data_entry = row[data_column_name]
                grouped_total_lengths[group_id] += len(data_entry)

    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}

    return dict(grouped_total_lengths)

def get_unique_activities(csv_path):
    """Reads a CSV file and returns a list of unique activity values.

    Args:
        csv_path (str): The path to the CSV file.

    Returns:
        list: Unique activity names from the CSV.
    """
    activities = set()
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            activities.add(row['activity'])
    return list(activities)

def find_longest_entry_length(csv_path, column_index=1):
    """Reads a CSV file, takes data from the specified column,
    and returns the length of the longest entry.

    Args:
        csv_path (str): The path to the CSV file.
        column_index (int, optional): The index of the column to read (0-based).
                                       Defaults to 1 (second column).

    Returns:
        int: The length of the longest string in the specified column.
             Returns 0 if the file is empty or the column is not found.
    """
    max_length = 0
    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) > column_index:
                    entry = row[column_index]
                    max_length = max(max_length, len(entry))
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0
    return max_length

if __name__ == '__main__':
    csv_file = f'{DATA_DIR}/ALL_20DRG_2022_2023_CLASS_Duration_ricovero_dimissioni_LAST_17Jan2025_padded_edited.csv'
    unique_activities = get_unique_activities(csv_file)
    print(unique_activities)

    longest_length = find_longest_entry_length(csv_file)
    print(f"La lunghezza dell'entry più lunga nella seconda colonna è: {longest_length}")

    total_lengths_by_case = calculate_total_length_grouped(csv_file)

    if total_lengths_by_case:
        lengths = list(total_lengths_by_case.values())
        ninety_fifth_percentile = np.percentile(lengths, 95)
        print(f"Il 95° percentile delle lunghezze totali è: {ninety_fifth_percentile}")

        # cases_below_percentile = {
        #     case_id: total_length
        #     for case_id, total_length in total_lengths_by_case.items()
        #     if total_length < ninety_fifth_percentile
        # }
        # print("\nCase ID con lunghezza totale inferiore al 95° percentile:")
        # for case_id, total_length in cases_below_percentile.items():
        #     print(f"Case ID: {case_id}, Lunghezza totale: {total_length}")

        # print("Lunghezza totale delle entries per ogni case_id:")
        # for case_id, total_length in total_lengths_by_case.items():
        #     print(f"Case ID: {case_id}, Lunghezza totale: {total_length}")

    print(max(total_lengths_by_case.items(), key=lambda item: item[1]))