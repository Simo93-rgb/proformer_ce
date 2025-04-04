import csv

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

if __name__ == '__main__':
    csv_file = 'data/ALL_20DRG_2022_2023_CLASS_Duration_ricovero_dimissioni_LAST_17Jan2025_padded.csv'
    unique_activities = get_unique_activities(csv_file)
    print(unique_activities)