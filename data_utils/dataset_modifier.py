import pandas as pd
import xml.etree.ElementTree as ET
import csv
import os
from datetime import datetime

def convert_timestamps(input_csv, output_csv):
    """
    Legge un file CSV, converte i timestamp nel formato desiderato e salva il risultato in un nuovo file CSV.
    
    Args:
        input_csv (str): Percorso del file CSV di input.
        output_csv (str): Percorso del file CSV di output.
    """
    # Leggi il file CSV
    df = pd.read_csv(input_csv, dtype=str)
    
    # Controlla se esiste una colonna 'timestamp'
    if 'timestamp' in df.columns:
        # Funzione per convertire i timestamp
        def convert_timestamp(ts):
            try:
                dt = datetime.fromisoformat(ts)
                return dt.isoformat(sep=' ', timespec='seconds')
            except ValueError:
                # Ritorna il valore originale se non è un timestamp valido
                return ts
        
        # Applica la conversione alla colonna 'timestamp'
        df['timestamp'] = df['timestamp'].apply(convert_timestamp)
    else:
        print("La colonna 'timestamp' non è presente nel file CSV.")
    
    # Salva il risultato in un nuovo file CSV
    df.to_csv(output_csv, index=False)
    print(f"File salvato con successo in: {output_csv}")


def aggregate_case_details(input_csv, output_csv):
    # Read the CSV file with all entries as strings
    df = pd.read_csv(input_csv, dtype=str)

    # Convert the timestamp column to datetime for proper sorting (adjust the format if needed)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Handle NaT values in the timestamp column
    if df['timestamp'].isna().any():
        print("Warning: Some timestamps could not be parsed and will be ignored.")

    # Sort by case_id and timestamp so that aggregated timestamps and actions remain in order
    df_sorted = df.sort_values(by=['case_id', 'timestamp'])

    # Define a helper function to format timestamps uniformly
    def format_timestamps(ts):
        return ','.join(ts.dropna().apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S%z')))

    # Aggregate timestamps and activity columns; for class, select the first non-null instance
    aggregated = df_sorted.groupby('case_id').agg({
        'timestamp': format_timestamps,
        'activity': lambda acts: ','.join(acts.dropna()),
    }).reset_index()

    # Save the aggregated data to CSV
    aggregated.to_csv(output_csv, index=False)


def add_class_column(input_csv, output_csv):
    """
    Aggiunge una colonna 'class' al CSV, contenente l'ultimo elemento della terza colonna.
    
    Args:
        input_csv (str): Percorso del file CSV di input.
        output_csv (str): Percorso del file CSV di output.
    """
    # Leggi il file CSV
    df = pd.read_csv(input_csv, dtype=str)
    
    # Controlla che ci siano almeno 3 colonne
    if len(df.columns) < 3:
        raise ValueError("Il file CSV deve avere almeno 3 colonne.")
    
    # Aggiungi la colonna 'class' con l'ultimo elemento della terza colonelimina l'ultimo elemento della terza colonnana
    df['class'] = df.iloc[:, 2].apply(lambda x: x.split(',')[-1] if isinstance(x, str) and ',' in x else x)
    df.iloc[:, 2] = df.iloc[:, 2].apply(lambda x: ','.join(x.split(',')[:-1]) if isinstance(x, str) and ',' in x else x)
    
    # Salva il risultato in un nuovo file CSV
    df.to_csv(output_csv, index=False)
    print(f"Colonna 'class' aggiunta e file salvato in: {output_csv}")


def aggregate_case_details_tuple(input_csv, output_csv):
    # Read the CSV file with all entries as strings
    df = pd.read_csv(input_csv, dtype=str)
    
    # Convert the timestamp column to datetime for proper sorting
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Sort by case_id and timestamp to maintain event order
    df_sorted = df.sort_values(by=['case_id', 'timestamp'])
    
    # Process each case individually
    aggregated_cases = []
    for case_id, group in df_sorted.groupby('case_id'):
        activities = []
        case_class = None
        class_timestamp = None
        
        # Extract activity with corresponding timestamps
        for _, row in group.iterrows():
            if not pd.isna(row['activity']) and not pd.isna(row['timestamp']):
                formatted_ts = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                activities.append(f"('{row['activity']}', '{formatted_ts}')")
        
        # The class is the last activity in the sorted group
        if not group['activity'].isnull().all():
            last_row = group.iloc[-1]
            if not pd.isna(last_row['activity']) and not pd.isna(last_row['timestamp']):
                case_class = last_row['activity']
                class_timestamp = last_row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                # Remove the last entry (class) from activities
                activities.pop()
        
        # Create the aggregated case entry
        case_entry = {
            'case_id': case_id,
            'activities': ', '.join(activities),
            'class': f"('{case_class}', '{class_timestamp}')" if case_class and class_timestamp else None
        }
        aggregated_cases.append(case_entry)
    
    # Convert to DataFrame and save to CSV
    aggregated_df = pd.DataFrame(aggregated_cases)
    aggregated_df.to_csv(output_csv, index=False)

def extract_patient_data(xes_file_path, output_csv_path):
    """
    Extract patient data from XES file to CSV where each row is a patient ID.
    
    Args:
        xes_file_path: Path to the XES file
        output_csv_path: Path where to save the CSV output
    """
    print(f"Starting extraction from {xes_file_path}...")
    
    # Parse the XES file
    tree = ET.parse(xes_file_path)
    root = tree.getroot()
    
    # Remove namespaces for easier processing
    for elem in root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
    
    # Count traces for debugging
    traces = root.findall('.//trace')
    trace_count = len(traces)
    print(f"Found {trace_count} traces in the XES file")
    
    all_patients = []
    processed_count = 0
    
    # Find all trace elements
    for trace in traces:
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count}/{trace_count} traces...")
        
        # Initialize patient data dictionary
        patient_data = {
            'patient_id': None,
            'ID_Paziente': None,
            'diagnosis': None,
            'Diagnosi_Principale_cod': None,
            'admission_date': None,
            'discharge_date': None,
            'hospital': None,
            'department': None,
            'department_name': None,
            'duration': None,
            'MDC_des': None,
            'DRG_des': None,
            'Procedura_Principale_des': None,
            'Procedura_Principale_cod': None,
            'Fascia_Eta': None,
            'Modalita_Dimissione_des': None,
            'Tipo_Ricovero_des': None,
            'class': None,
            'num_events': 0,
            'num_radiografie': 0,
            'num_tc': 0,
            'num_visite': 0,
            'num_riabilitazione': 0,
            'activities': [],
            'timestamps': [],
        }
        
        # Extract trace-level attributes
        for elem in trace:
            if elem.tag == 'string':
                key = elem.attrib.get('key', '')
                value = elem.attrib.get('value', '')
                
                if key == 'concept:name':
                    patient_data['patient_id'] = value
                elif key == 'Diagnosi_Principale_des':
                    patient_data['diagnosis'] = value
                elif key == 'Diagnosi_Principale_cod':
                    patient_data['Diagnosi_Principale_cod'] = value
                elif key == 'Data_Ricovero.1':
                    patient_data['admission_date'] = value
                elif key == 'Data_Dimissione.1':
                    patient_data['discharge_date'] = value
                elif key == 'HSP_des':
                    patient_data['hospital'] = value
                elif key == 'Duration':
                    patient_data['duration'] = value
                elif key == 'MDC_des':
                    patient_data['MDC_des'] = value
                elif key == 'DRG_des':
                    patient_data['DRG_des'] = value
                elif key == 'Procedura_Principale_des':
                    patient_data['Procedura_Principale_des'] = value
                elif key == 'AMB_DESCR_BRV':
                    patient_data['department_name'] = value
                elif key == 'Fascia_Eta':
                    patient_data['Fascia_Eta'] = value
                elif key == 'Modalita_Dimissione_des':
                    patient_data['Modalita_Dimissione_des'] = value
                elif key == 'Tipo_Ricovero_des':
                    patient_data['Tipo_Ricovero_des'] = value
            
            elif elem.tag == 'int':
                key = elem.attrib.get('key', '')
                value = elem.attrib.get('value', '')
                
                if key == 'class' or key == 'class_ricovero_dimissioni':
                    patient_data['class'] = value
                elif key == 'ID_Paziente':
                    patient_data['ID_Paziente'] = value
                elif key == 'Duration':
                    patient_data['duration'] = value
                elif key == 'Procedura_Principale_cod':
                    patient_data['Procedura_Principale_cod'] = value
        
        # Extract events data
        events = trace.findall('./event')
        patient_data['num_events'] = len(events)
        
        # Extract additional information from the first event
        if events:
            first_event = events[0]
            for string_elem in first_event.findall('./string'):
                key = string_elem.attrib.get('key', '')
                value = string_elem.attrib.get('value', '')
                
                if key == 'Data_Ricovero.1' and not patient_data['admission_date']:
                    patient_data['admission_date'] = value
                    
                if key == 'AMB_DESCR_BRV' and not patient_data['department_name']:
                    patient_data['department_name'] = value
        
        for event in events:
            activity = None
            timestamp = None
            department = None
            lifecycle = None
            
            for string_elem in event.findall('./string'):
                key = string_elem.attrib.get('key', '')
                value = string_elem.attrib.get('value', '')
                
                if key == 'concept:name':
                    if not value.startswith('class_'):
                        activity = value
                elif key == 'CD_REPARTO':
                    department = value
                elif key == 'lifecycle:transition':
                    lifecycle = value
            
            for date_elem in event.findall('./date'):
                key = date_elem.attrib.get('key', '')
                value = date_elem.attrib.get('value', '')
                
                if key == 'time:timestamp':
                    timestamp = value
            
            # Count only valid activities
            if activity and timestamp and not activity.startswith('class_'):
                # Add to activity and timestamp lists
                patient_data['activities'].append(activity)
                patient_data['timestamps'].append(timestamp)
                
                # Count by type (only completed events)
                if lifecycle == 'complete':
                    if 'RX' in activity:
                        patient_data['num_radiografie'] += 1
                    elif 'TC' in activity:
                        patient_data['num_tc'] += 1
                    elif 'VISITA' in activity:
                        patient_data['num_visite'] += 1
                    elif 'RIEDUCAZIONE' in activity or 'RIABILITA' in activity:
                        patient_data['num_riabilitazione'] += 1
                
                if department and not patient_data['department']:
                    patient_data['department'] = department
        
        # Add patient data if we have an ID
        if patient_data['patient_id'] or patient_data['ID_Paziente']:
            # Use ID_Paziente as backup if patient_id isn't set
            if not patient_data['patient_id'] and patient_data['ID_Paziente']:
                patient_data['patient_id'] = patient_data['ID_Paziente']
                
            # Convert lists to comma-separated strings
            patient_data['activities'] = ','.join(patient_data['activities']) if patient_data['activities'] else ""
            patient_data['timestamps'] = ','.join(patient_data['timestamps']) if patient_data['timestamps'] else ""
            all_patients.append(patient_data)
    
    # Write to CSV
    if all_patients:
        fields = all_patients[0].keys()
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_patients)
        
        print(f"Successfully extracted data for {len(all_patients)} patients to {output_csv_path}")
    else:
        print("No patients found in the XES file")
        # Debug information if no patients found
        if trace_count > 0:
            print("Debug info for first trace:")
            for elem in traces[0]:
                if elem.tag in ['string', 'int']:
                    print(f"  {elem.tag}: {elem.attrib}")

if __name__ == "__main__":
    # xes_file = "data/ALL_20DRG_2022_2023_CLASS_Duration_ricovero_dimissioni_LAST_17Jan2025_padded.xes"
    # csv_file = "data/ALL_20DRG_2022_2023_CLASS_Duration_ricovero_dimissioni_LAST_17Jan2025_padded.csv"
    # csv_file_edited = "data/ALL_20DRG_2022_2023_CLASS_Duration_ricovero_dimissioni_LAST_17Jan2025_padded_edited.csv"
    # output_csv = "data/aggregated_case_2.csv"
    # output_csv_tuple = "data/aggregated_case_tuple.csv"
    # # extract_patient_data(xes_file, output_csv)
    # convert_timestamps(csv_file, csv_file_edited)
    # aggregate_case_details(csv_file_edited, output_csv)
    # aggregate_case_details_tuple(csv_file_edited, output_csv_tuple)
    add_class_column("data/aggregated_case_detailed_to_classify.csv", "data/aggregated_case_detailed_to_classify_final.csv")