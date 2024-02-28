from PIL import Image
import os


def convert_images(source_folder, target_folder, conflict_folder_name='conflicts'):
    stats = {
        'Original_RGBA': 0,
        'Converted_to_RGB': 0,
        'Original_RGB': 0,
        'Other': 0,
        'Failed': 0
    }
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    conflict_folder = os.path.join(target_folder, conflict_folder_name)
    if not os.path.exists(conflict_folder):
        os.makedirs(conflict_folder)
    for file_name in os.listdir(source_folder):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(source_folder, file_name)
            target_path = os.path.join(target_folder, file_name)
            try:
                with Image.open(file_path) as img:
                    original_mode = img.mode
                    if original_mode == 'RGBA':
                        stats['Original_RGBA'] += 1
                        img = img.convert('RGB')
                        stats['Converted_to_RGB'] += 1
                    elif original_mode == 'RGB':
                        stats['Original_RGB'] += 1
                    if os.path.exists(target_path):
                        # Save to conflict folder instead
                        target_path = os.path.join(conflict_folder, file_name)
                    img.save(target_path)
                    print(f"Processed {file_name}: original mode was {original_mode}, saved to {target_path}")
            except Exception as e:
                stats['Failed'] += 1
                print(f"Failed to process {file_name}: {e}")
    print("\n--- Processing summary ---")
    for key, count in stats.items():
        print(f"{key} images: {count}")


if __name__ == "__main__":
    source_folder = r'G:\xiaowu-pic\133_select_new'
    target_folder = r'G:\xiaowu-pic\133_3channel'
    convert_images(source_folder, target_folder)
