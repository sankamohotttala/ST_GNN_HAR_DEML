import os
import shutil

def copy_highest_numbered_file(path1, path2):
    # Define the folder names
    folder_name = "images_independant"
    source_folder = os.path.join(path1, folder_name)
    destination_folder = os.path.join(path2, folder_name)

    # Ensure the source folder exists
    if not os.path.exists(source_folder):
        print(f"Source folder '{source_folder}' does not exist.")
        return

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Find the file with the highest number in the source folder
    highest_number = -1
    highest_file = None
    for file_name in os.listdir(source_folder):
        if file_name.startswith("conf") and file_name[4:-4].isdigit():
            number = int(file_name[4:-4])
            if number > highest_number:
                highest_number = number
                highest_file = file_name

    if highest_file:
        # Copy the file to the destination folder
        source_file = os.path.join(source_folder, highest_file)
        destination_file = os.path.join(destination_folder, highest_file)
        shutil.copy2(source_file, destination_file)
        print(f"Copied '{highest_file}' to '{destination_folder}'.")
    else:
        print(f"No valid 'conf{{number}}' files found in '{source_folder}'.")


def copy_all_except_events_file(path1, path2):
    # Define the folder names
    folder_name = "log"
    source_folder = os.path.join(path1, folder_name)
    destination_folder = os.path.join(path2, folder_name)

    # Ensure the source folder exists
    if not os.path.exists(source_folder):
        print(f"Source folder '{source_folder}' does not exist.")
        return

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Copy all files except those starting with 'events.out'
    for file_name in os.listdir(source_folder):
        if not file_name.startswith("events.out"):
            source_file = os.path.join(source_folder, file_name)
            destination_file = os.path.join(destination_folder, file_name)
            shutil.copy2(source_file, destination_file)
            print(f"Copied '{file_name}' to '{destination_folder}'.")
        else:
            print(f"Skipped '{file_name}'.")

# Example usage
def copy_files_only(path1, path2):
    # Define the folder names
    folder_name = "matplotlib_graphs"
    source_folder = os.path.join(path1, folder_name)
    destination_folder = os.path.join(path2, folder_name)

    # Ensure the source folder exists
    if not os.path.exists(source_folder):
        print(f"Source folder '{source_folder}' does not exist.")
        return

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Copy only files from the source folder to the destination folder
    for item in os.listdir(source_folder):
        source_item = os.path.join(source_folder, item)
        if os.path.isfile(source_item):  # Check if it's a file
            destination_file = os.path.join(destination_folder, item)
            shutil.copy2(source_item, destination_file)
            print(f"Copied file '{item}' to '{destination_folder}'.")
        else:
            print(f"Skipped folder '{item}'.")



def copy_highest_numbered_csv(path1, path2):
    # Define the folder names
    folder_name = "save_values_csv"
    source_folder = os.path.join(path1, folder_name)
    destination_folder = os.path.join(path2, folder_name)

    # Ensure the source folder exists
    if not os.path.exists(source_folder):
        print(f"Source folder '{source_folder}' does not exist.")
        return

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Find the CSV file with the highest number in the source folder
    highest_number = -1
    highest_file = None
    for file_name in os.listdir(source_folder):
        if file_name.startswith("data_") and file_name.endswith(".csv") and file_name[5:-4].isdigit():
            number = int(file_name[5:-4])
            if number > highest_number:
                highest_number = number
                highest_file = file_name

    if highest_file:
        # Copy the file to the destination folder
        source_file = os.path.join(source_folder, highest_file)
        destination_file = os.path.join(destination_folder, highest_file)
        shutil.copy2(source_file, destination_file)
        print(f"Copied '{highest_file}' to '{destination_folder}'.")
    else:
        print(f"No valid 'data_{{number}}.csv' files found in '{source_folder}'.")



# Perform all tasks using the defined functions
def perform_all_tasks(path1, path2):
    copy_highest_numbered_file(path1, path2)
    copy_all_except_events_file(path1, path2)
    copy_files_only(path1, path2)
    copy_highest_numbered_csv(path1, path2)

# Example usage
path1 = r'E:\SLIIT RA 2024\SICET\zST-GCN_CWBG_LOOPnew\results_cwbg_loop\cwbg_shared_acc'
path2 = r'E:\SLIIT RA 2024\SICET\github_for_journal\ST_GNN_HAR_DEML\hyperparam_comparison\cross_subject\cwbg_shared'
# Get all folders in path1
for folder_name in os.listdir(path1):
    folder_path = os.path.join(path1, folder_name)
    if os.path.isdir(folder_path):  # Check if it's a folder
        # Create a corresponding folder in path2
        new_folder_path = os.path.join(path2, folder_name)
        os.makedirs(new_folder_path, exist_ok=True)
        # Perform tasks for the current folder
        perform_all_tasks(folder_path, new_folder_path)
# perform_all_tasks(path1, path2)