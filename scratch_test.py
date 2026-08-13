from src.config import TRAIN_PATH, TEST_PATH
from src.data_loader import load_dataset, build_label_map, encode_labels

train_df = load_dataset(TRAIN_PATH)
test_df = load_dataset(TEST_PATH)

label_map = build_label_map(train_df)
print(label_map)

train_df = encode_labels(train_df, label_map)
test_df = encode_labels(test_df, label_map)

print(train_df.head())