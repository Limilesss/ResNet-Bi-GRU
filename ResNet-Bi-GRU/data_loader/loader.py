import os
import pandas as pd

def load_data(folder_path):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    combined_data = pd.DataFrame()

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, header=2, usecols=['IsLa [A]', 'IsLb [A]', 'IsLc [A]', 'UsLLa [V]', 'UsLLb [V]', 'UsLLc [V]'])
        df['Source'] = file
        combined_data = pd.concat([combined_data, df], ignore_index=True)

    return combined_data