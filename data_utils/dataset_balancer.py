import pandas as pd
import numpy as np
import random
import os
from config import DATA_DIR
from typing import Dict, List, Tuple, Optional
from imblearn.over_sampling import SMOTE, ADASYN
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse
import datetime


def balance_dataset(input_csv: str, output_csv: str, balance_ratio: float = 0.8) -> None:
    """
    Legge un CSV con dati di processo, bilancia il dataset usando tecniche avanzate di augmentation
    e salva il risultato in un nuovo CSV, assicurando che le classi siano sempre l'ultima azione.

    Args:
        input_csv: Percorso del CSV di input
        output_csv: Percorso del CSV di output bilanciato
        balance_ratio: Rapporto target tra classe minoritaria e maggioritaria (1.0 = perfettamente bilanciato)
    """
    print(f"Lettura del file {input_csv}")
    df = pd.read_csv(input_csv,
                     parse_dates=["timestamp"] if "timestamp" in pd.read_csv(input_csv, nrows=1).columns else None)

    # Assicurati che le colonne corrette siano presenti
    required_cols = ['case_id', 'activity']
    if not all(col in df.columns for col in required_cols):
        # Rinomina automaticamente le colonne se hanno nomi simili
        col_mapping = {}
        for col in df.columns:
            if 'case' in col.lower() and 'id' in col.lower():
                col_mapping[col] = 'case_id'
            elif 'activ' in col.lower():
                col_mapping[col] = 'activity'
            elif 'time' in col.lower() or 'stamp' in col.lower():
                col_mapping[col] = 'timestamp'

        df = df.rename(columns=col_mapping)
        print(f"Colonne rinominate: {col_mapping}")

    # Raggruppa per case_id
    grouped = df.groupby('case_id')

    # Estrai le sequenze e identifica la classe di ogni sequenza
    sequences: Dict[int, List[Tuple[str, pd.Timestamp]]] = {}
    sequence_classes: Dict[int, int] = {}
    case_features: Dict[int, np.ndarray] = {}

    print("Raggruppamento delle sequenze e identificazione delle classi...")
    for case_id, group in grouped:
        # Ordina per timestamp se disponibile
        if 'timestamp' in group.columns:
            group = group.sort_values('timestamp')

        # Estrai attività e timestamp
        activities = group['activity'].tolist()
        timestamps = group['timestamp'].tolist() if 'timestamp' in group.columns else [None] * len(activities)

        # Trova e rimuovi la classe, poi posizionala alla fine
        non_class_activities = []
        non_class_timestamps = []
        class_activity = None
        class_timestamp = None

        # Identifica tutte le posizioni delle classi
        class_positions = [i for i, act in enumerate(activities) if act in ['class_0', 'class_1']]

        if class_positions:
            # Prendi l'ultima classe
            last_class_pos = class_positions[-1]
            class_activity = activities[last_class_pos]
            class_timestamp = timestamps[last_class_pos]

            # Rimuovi tutte le classi dalla sequenza
            non_class_activities = [act for i, act in enumerate(activities) if i not in class_positions]
            non_class_timestamps = [ts for i, ts in enumerate(timestamps) if i not in class_positions]

            # Aggiungi la classe alla fine
            non_class_activities.append(class_activity)
            non_class_timestamps.append(class_timestamp if class_timestamp is not None else timestamps[-1])

            # Salva la sequenza riorganizzata
            sequences[case_id] = list(zip(non_class_activities, non_class_timestamps))

            # Determina il valore numerico della classe
            sequence_classes[case_id] = 1 if class_activity == 'class_1' else 0

            # Crea feature per ogni caso (per SMOTE/ADASYN)
            # Creiamo un vettore di caratteristiche basato sulla durata e sul pattern di attività
            if timestamps[0] is not None:
                # Calcola durata totale
                duration = (timestamps[-1] - timestamps[0]).total_seconds()
                # Calcola intervalli medi tra attività
                intervals = []
                for i in range(1, len(timestamps)):
                    if timestamps[i - 1] is not None and timestamps[i] is not None:
                        intervals.append((timestamps[i] - timestamps[i - 1]).total_seconds())

                avg_interval = np.mean(intervals) if intervals else 0
                std_interval = np.std(intervals) if intervals else 0

                # Crea feature vector
                case_features[case_id] = np.array([
                    duration,
                    avg_interval,
                    std_interval,
                    len(activities)
                ])
            else:
                # Se non ci sono timestamp, usa solo la lunghezza della sequenza
                case_features[case_id] = np.array([len(activities)])
        else:
            # Nessuna classe trovata, salva la sequenza originale
            sequences[case_id] = list(zip(activities, timestamps))
            sequence_classes[case_id] = -1  # Classe sconosciuta

    # Conta le sequenze per classe
    class_0_cases = [case_id for case_id, cls in sequence_classes.items() if cls == 0]
    class_1_cases = [case_id for case_id, cls in sequence_classes.items() if cls == 1]
    unknown_cases = [case_id for case_id, cls in sequence_classes.items() if cls == -1]

    print(
        f"Distribuzione originale: Classe 0: {len(class_0_cases)}, Classe 1: {len(class_1_cases)}, Sconosciuta: {len(unknown_cases)}")

    # Identifica la classe minoritaria e maggioritaria
    minority_class = 1 if len(class_1_cases) < len(class_0_cases) else 0
    majority_class = 1 - minority_class
    minority_cases = class_1_cases if minority_class == 1 else class_0_cases
    majority_cases = class_0_cases if minority_class == 1 else class_1_cases

    # Preparazione per data augmentation avanzata
    print("Applicazione di tecniche di data augmentation avanzate...")
    # Converti in matrice di feature e array di classi per SMOTE/ADASYN
    feature_ids = sorted(case_features.keys())
    X = np.array([case_features[fid] for fid in feature_ids])
    y = np.array([sequence_classes[fid] for fid in feature_ids])

    # Rimuovi casi con classe sconosciuta
    valid_indices = np.where(y != -1)[0]
    X = X[valid_indices]
    y = y[valid_indices]
    valid_feature_ids = [feature_ids[i] for i in valid_indices]

    # Calcola quanti casi aggiungere
    current_ratio = len(minority_cases) / max(1, len(majority_cases))
    target_ratio = min(1.0, balance_ratio)  # Limita a 1.0 per evitare oversampling eccessivo

    if len(X) > 0 and len(np.unique(y)) > 1:
        try:
            # Prova con ADASYN (più avanzato di SMOTE)
            sampler = ADASYN(sampling_strategy=target_ratio, random_state=42,
                             n_neighbors=min(5, len(minority_cases) - 1) if len(minority_cases) > 1 else 1)
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            print(f"ADASYN ha generato {len(X_resampled) - len(X)} nuovi casi sintetici")
        except Exception as e:
            print(f"ADASYN fallito: {e}, provo con SMOTE...")
            try:
                # Fallback a SMOTE
                sampler = SMOTE(sampling_strategy=target_ratio, random_state=42,
                                k_neighbors=min(5, len(minority_cases) - 1) if len(minority_cases) > 1 else 1)
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                print(f"SMOTE ha generato {len(X_resampled) - len(X)} nuovi casi sintetici")
            except Exception as e2:
                print(f"Anche SMOTE fallito: {e2}, utilizzo metodo di augmentation manuale")
                # Fallback al metodo manuale
                X_resampled, y_resampled = X, y
    else:
        # Non abbastanza dati o classi per SMOTE/ADASYN, usa metodo manuale
        print("Non abbastanza dati per SMOTE/ADASYN, utilizzo metodo di augmentation manuale")
        X_resampled, y_resampled = X, y

    # Se SMOTE/ADASYN non hanno generato abbastanza campioni, aggiungi con metodi manuali
    if len(X_resampled) == len(X):
        # Calcola quanti casi aggiungere
        target_minority_count = int(len(majority_cases) * target_ratio)
        cases_to_add = max(0, target_minority_count - len(minority_cases))

        print(f"Rapporto attuale: {current_ratio:.2f}, Rapporto target: {target_ratio:.2f}")
        print(f"Classe {minority_class} è minoritaria. Aggiunta di {cases_to_add} casi con metodi avanzati.")

        # Crea nuovi case_id unici
        max_case_id = max(sequences.keys()) if sequences else 0
        next_case_id = max_case_id + 1

        # Applica tecniche di augmentation avanzate
        for _ in range(cases_to_add):
            # Scegli un caso casuale da aumentare
            original_case_id = random.choice(minority_cases)
            original_sequence = sequences[original_case_id]

            # Applica una delle seguenti tecniche di augmentation in modo casuale
            augmentation_type = random.choice(['jittering', 'permutation', 'substitution', 'time_warp'])

            if augmentation_type == 'jittering':
                # Jittering: aggiunge variazioni più significative ai timestamp
                new_sequence = []
                for activity, timestamp in original_sequence:
                    if timestamp is not None:
                        # Aggiungi jitter significativo ma plausibile (±60 minuti)
                        jitter_seconds = random.uniform(-3600, 3600)
                        new_timestamp = timestamp + pd.Timedelta(seconds=jitter_seconds)
                        new_sequence.append((activity, new_timestamp))
                    else:
                        new_sequence.append((activity, None))

            elif augmentation_type == 'permutation' and len(original_sequence) > 3:
                # Permutazione: scambia alcune attività intermedie mantenendo l'ordine logico
                # Non tocca la prima, l'ultima e la classe (che è sempre l'ultima)
                new_sequence = list(original_sequence)  # Copia
                # Permuta solo attività centrali (evita la prima, l'ultima e la classe)
                perm_range = range(1, len(new_sequence) - 2)
                if len(perm_range) >= 2:
                    idx1, idx2 = random.sample(list(perm_range), 2)
                    new_sequence[idx1], new_sequence[idx2] = new_sequence[idx2], new_sequence[idx1]

            elif augmentation_type == 'substitution':
                # Sostituzione: sostituisce alcune attività con altre simili del dataset
                # Mantiene la stessa classe alla fine
                new_sequence = []
                all_activities = set()
                for seq in sequences.values():
                    for act, _ in seq:
                        if act not in ['class_0', 'class_1']:
                            all_activities.add(act)

                for i, (activity, timestamp) in enumerate(original_sequence):
                    if i < len(original_sequence) - 1 and activity not in ['class_0',
                                                                           'class_1'] and random.random() < 0.3:
                        # 30% di probabilità di sostituire attività non-classe
                        similar_activities = list(all_activities - {'class_0', 'class_1'})
                        if similar_activities:
                            new_activity = random.choice(similar_activities)
                            new_sequence.append((new_activity, timestamp))
                        else:
                            new_sequence.append((activity, timestamp))
                    else:
                        new_sequence.append((activity, timestamp))

            elif augmentation_type == 'time_warp':
                # Time Warping: comprime o espande il tempo tra alcune attività
                if all(ts is not None for _, ts in original_sequence):
                    activities = [a for a, _ in original_sequence]
                    timestamps = [ts for _, ts in original_sequence]

                    # Calcola gli intervalli tra attività
                    intervals = [(timestamps[i + 1] - timestamps[i]).total_seconds()
                                 for i in range(len(timestamps) - 1)]

                    # Applica warping (compressione/espansione) casuale agli intervalli
                    warped_intervals = []
                    for interval in intervals:
                        warp_factor = random.uniform(0.7, 1.3)  # Comprime o espande fino al 30%
                        warped_intervals.append(interval * warp_factor)

                    # Ricostruisci i timestamp
                    new_timestamps = [timestamps[0]]
                    for interval in warped_intervals:
                        new_timestamps.append(new_timestamps[-1] + pd.Timedelta(seconds=interval))

                    new_sequence = list(zip(activities, new_timestamps))
                else:
                    new_sequence = original_sequence

            else:
                # Fallback: usa il metodo originale
                new_sequence = list(original_sequence)
                if new_sequence[0][1] is not None:
                    for i in range(len(new_sequence)):
                        activity, timestamp = new_sequence[i]
                        if timestamp is not None:
                            random_seconds = random.randint(-300, 300)
                            new_timestamp = timestamp + pd.Timedelta(seconds=random_seconds)
                            new_sequence[i] = (activity, new_timestamp)

            # Salva la nuova sequenza con il nuovo case_id
            sequences[next_case_id] = new_sequence
            sequence_classes[next_case_id] = minority_class
            next_case_id += 1
    else:
        # SMOTE/ADASYN ha generato nuovi campioni
        print(f"Utilizzo di {len(X_resampled) - len(X)} campioni generati da SMOTE/ADASYN")

        # Crea nuovi case_id per i campioni sintetici
        max_case_id = max(sequences.keys()) if sequences else 0
        next_case_id = max_case_id + 1

        # Per ogni nuovo campione sintetico
        for i in range(len(X), len(X_resampled)):
            # Trova il caso più vicino nella classe minoritaria
            nearest_minority_idx = np.argmin(np.sum((X[y == minority_class] - X_resampled[i]) ** 2, axis=1))
            nearest_minority_case_id = [case_id for j, case_id in enumerate(valid_feature_ids)
                                        if y[j] == minority_class][nearest_minority_idx]

            # Usa la sequenza del caso più vicino come base
            base_sequence = sequences[nearest_minority_case_id]

            # Calcola la differenza percentuale nelle feature per adattare la sequenza
            if len(X[0]) > 1:  # Se abbiamo più di una feature
                duration_ratio = X_resampled[i][0] / X[nearest_minority_idx][0] if X[nearest_minority_idx][0] > 0 else 1
                interval_ratio = X_resampled[i][1] / X[nearest_minority_idx][1] if X[nearest_minority_idx][1] > 0 else 1
            else:
                # Se abbiamo solo la lunghezza come feature
                duration_ratio = 1
                interval_ratio = 1

            # Crea una nuova sequenza adattata
            new_sequence = []
            for j, (activity, timestamp) in enumerate(base_sequence):
                if timestamp is not None and j > 0 and base_sequence[j - 1][1] is not None:
                    # Adatta l'intervallo temporale in base al ratio calcolato
                    prev_ts = base_sequence[j - 1][1]
                    interval = (timestamp - prev_ts).total_seconds()
                    new_interval = interval * interval_ratio
                    new_ts = prev_ts + pd.Timedelta(seconds=new_interval)
                    new_sequence.append((activity, new_ts))
                else:
                    new_sequence.append((activity, timestamp))

            # Salva la nuova sequenza
            sequences[next_case_id] = new_sequence
            sequence_classes[next_case_id] = minority_class
            next_case_id += 1

    # Riconverti in formato dataframe
    balanced_rows = []
    for case_id, sequence in sequences.items():
        for activity, timestamp in sequence:
            row = {'case_id': case_id, 'activity': activity}
            if timestamp is not None:
                row['timestamp'] = timestamp
            balanced_rows.append(row)

    # Crea il nuovo dataframe
    balanced_df = pd.DataFrame(balanced_rows)

    # Verifica la nuova distribuzione
    balanced_classes = {case_id: cls for case_id, cls in sequence_classes.items()
                        if case_id in sequences}
    balanced_class_0 = sum(1 for cls in balanced_classes.values() if cls == 0)
    balanced_class_1 = sum(1 for cls in balanced_classes.values() if cls == 1)

    print(f"Nuova distribuzione: Classe 0: {balanced_class_0}, Classe 1: {balanced_class_1}")
    print(
        f"Nuovo rapporto: {min(balanced_class_0, balanced_class_1) / max(1, max(balanced_class_0, balanced_class_1)):.2f}")

    # Salva il CSV bilanciato
    balanced_df.to_csv(output_csv, index=False)
    print(f"Dataset bilanciato salvato in {output_csv}")


if __name__ == '__main__':
    # Ottieni il nome del file originale
    input_filename = f'{DATA_DIR}/ALL_20DRG_2022_2023_CLASS_Duration_ricovero_dimissioni_LAST_17Jan2025_padded_edited.csv'

    # Crea il nome del file di output
    base_name = os.path.basename(input_filename)
    name_without_ext = os.path.splitext(base_name)[0]
    output_filename = f'{DATA_DIR}/{name_without_ext}_balanced.csv'

    # Esegui il bilanciamento
    balance_dataset(input_filename, output_filename, balance_ratio=0.9)