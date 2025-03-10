import os

def rename_images(old_id, new_id):
    dataset_path = "dataset"  
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset folder '{dataset_path}' does not exist!")
        return

    # Get all files in the dataset folder
    files = os.listdir(dataset_path)
 
    renamed_count = 0
    for file in files:
        # Check if the file matches the format `User.<old_id>.<count>.jpg`
        if file.startswith(f"User.{old_id}.") and file.endswith(".jpg"):
            try:
                # Extract the iteration count
                count = file.split('.')[2]
                # New filename with the new ID
                new_name = f"User.{new_id}.{count}.jpg"
                # Rename the file
                old_path = os.path.join(dataset_path, file)
                new_path = os.path.join(dataset_path, new_name)
                os.rename(old_path, new_path)
                renamed_count += 1
            except Exception as e:
                print(f"[ERROR] Could not rename file {file}: {e}")

    if renamed_count > 0:
        print(f"[INFO] Successfully renamed {renamed_count} images from ID {old_id} to ID {new_id}.")
    else:
        print(f"[INFO] No images found with ID {old_id}.")

# Prompt for old ID and new ID
old_id = input("Enter the old user ID to replace: ")
new_id = input("Enter the new user ID: ")

rename_images(old_id, new_id)
