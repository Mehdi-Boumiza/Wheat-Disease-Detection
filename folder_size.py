import os

BASE_DIR = "/Users/mehdiboumiza/Downloads/wheat project for prod/data to use" 

splits = ["train", "valid", "test"]



for split in splits:
    split_path = os.path.join(BASE_DIR, split)
    print(f"\n {split.upper()} \n")

    total_split = 0

    for class_name in sorted(os.listdir(split_path)):
        class_path = os.path.join(split_path, class_name)

        if not os.path.isdir(class_path):
            continue

        num_files = len(os.listdir(class_path))
        total_split += num_files

        print(f"{class_name}: {num_files}")

    print(f"total in {split}: {total_split}")

