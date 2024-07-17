import csv
import json
from datasets import load_dataset

# For filtering eval dataset in.

kfashion_images_group_reader = csv.reader(open("./data/kfashion_images_group.tsv", "r"), delimiter="\t")

eval_img_id_set = set()

for i, row in enumerate(kfashion_images_group_reader):
    if i == 0:
        continue
    img_id = int(row[1].split("/")[-1].split(".")[0])
    eval_img_id_set.add(img_id)



write_csv_file = open("./data/fashion_test_queries_reformated.csv", "w")
writer = csv.writer(write_csv_file, quoting=csv.QUOTE_MINIMAL)

eval_write_csv_file = open("./data/fashion_test_queries_reformated_eval.csv", "w")
eval_writer = csv.writer(eval_write_csv_file, quoting=csv.QUOTE_MINIMAL)

with open("./data/fashion_test_queries.csv") as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i == 0:
            continue
        row_no_newline = [r.replace("\n", " ") for r in row]
        writer.writerow(row_no_newline)
        img_id = int(row_no_newline[5].split("/")[-1].split(".")[0])
        if img_id in eval_img_id_set:
            eval_writer.writerow(row_no_newline)

write_csv_file.close()
eval_write_csv_file.close()