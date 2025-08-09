from src.__00__paths import data_dir_list, model_dir

# Iterate all over the two lists to create directories
for path in data_dir_list + [model_dir]:
    path.mkdir(parents=True, exist_ok=True)
