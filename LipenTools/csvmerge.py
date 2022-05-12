import os

IN_PATH = "./csv_in/"
OUT_PATH = "./merged.csv"

in_file_paths = [IN_PATH + in_file_name for in_file_name in os.listdir(IN_PATH)]
first = True

with open(OUT_PATH, 'w') as out_file:
    for in_file_path in in_file_paths:
        with open(in_file_path, 'r') as in_file:
            lines = in_file.readlines()
            if not first:
                lines = lines[1:]
            out_file.writelines(lines)
        if in_file_path != in_file_paths[-1]:
            out_file.write('\n')
        first = False

