import os
import shutil

HARD_IN_PATH = "./hard_in/"
ALL_IN_PATH = "./all.csv"
CORRECTED_PATH = "./corrected.csv"
REJECTED_PATH = "./rejected.csv"
ALL_PHOTOS_PATH = "./all_photos/"
REJECTED_PHOTOS_PATH = "./rejected_photos/"
REJECTED_PHOTOS_FLAT_PATH = "./rejected_photos_flat/"

hard_file_paths = [HARD_IN_PATH + hard_file_name for hard_file_name in os.listdir(HARD_IN_PATH)]
hard_file_paths.sort()

fleiss_ratings_num = len(os.listdir(HARD_IN_PATH))
fleiss_categories = 14
classes = 7
class_names = ["Triangle", "Rules", "Rubber", "Pencil", "Pen", "None", "Invalid"]


with open(hard_file_paths[0]) as hard_file:
    fleiss_samples_num = len(hard_file.readlines()) - 1
    
fleiss_pj = [0] * fleiss_categories
fleiss_Pi = [0] * fleiss_samples_num
fleiss_nij = [[0] * fleiss_categories for x in range(fleiss_samples_num)]

perfect_agreement, great_agreement, good_agreement, half_agreement, slight_agreement, poor_agreement = [0] * 6
rejected = [[] for x in range(classes)]
new_class_subclass = dict()
new_extraclass = dict()
hard_photo_paths = []
#perfect -> == 100%
#great -> 80% <= x < 100%
#good -> 50% < x < 80%
#half -> == 50%
#slight -> 20% <= x < 50%
#poor -> x < 20%

def get_category(class_no, subclass_no):
    class_no = int(class_no)
    subclass_no = int(subclass_no)
    if class_no == 0:
        return subclass_no
    elif class_no == 1 or class_no == 2:
        return class_no + 1
    elif class_no == 3:
        return 4 + subclass_no
    elif class_no == 4:
        return 8 + subclass_no
    elif class_no == 5:
        return 11 + subclass_no
    else:
        return 13
        
def get_class_subclass(category):
    if category == 13:
        return 6, 0
    elif category >= 11:
        return 5, category - 11
    elif category >= 8:
        return 4, category - 8
    elif category >= 4:
        return 3, category - 4
    elif category >= 2:
        return category - 1, 0
    else:
        return 0, category

def print_result(text_prefix, agreement):
    print(f"{text_prefix} agreement: {agreement} ({agreement / fleiss_samples_num * 100:.2f}%)")

def print_results():
    print("##################################################")
    print_result("Perfect (== 100%)", perfect_agreement)
    print_result("Great (80% <= x < 100%)", great_agreement)
    print_result("Good (50% < x < 80%)", good_agreement)
    print_result("Half (== 50%)", half_agreement)
    print_result("Slight (20% <= x < 50%)", slight_agreement)
    print_result("Poor (x < 20%)", poor_agreement)
    print("##################################################")
    print("Rejection results per class:")
    rejected_num = [len(rejected_class) for rejected_class in rejected]
    for class_no in range(classes):
        print(f"{class_names[class_no]}: {rejected_num[class_no]}")
    print(f"In total: {sum(rejected_num)}")
    print("##################################################")
    print("Rejected files:")
    for class_no in range(classes):
        print(f"{class_names[class_no]}: {rejected[class_no]}")
    print("##################################################")

for hard_file_no, hard_file_path in enumerate(hard_file_paths):
    with open(hard_file_path, 'r') as hard_file:
        lines = hard_file.readlines()
        lines = lines[1:]
        lines.sort()
        for index, line in enumerate(lines):
            file_path, class_no, subclass_no, extraclass_code, _, _ = line.split(';')
            category = get_category(class_no, subclass_no)
            fleiss_pj[category] += 1
            fleiss_nij[index][category] += 1 
            if file_path not in new_extraclass.keys():
                new_extraclass[file_path] = 0
            new_extraclass[file_path] |= int(extraclass_code)
            if hard_file_no == 0:
                hard_photo_paths.append(file_path)
fleiss_pj = [pj / (fleiss_ratings_num * fleiss_samples_num) for pj in fleiss_pj]
for i in range(0, fleiss_samples_num):
    fleiss_Pi[i] = sum([fleiss_nij_ij * (fleiss_nij_ij - 1) for fleiss_nij_ij in fleiss_nij[i]])
    fleiss_Pi[i] *= (1 / (fleiss_ratings_num * (fleiss_ratings_num - 1)))
    max_category_votes = max(fleiss_nij[i])
    max_voted_category = fleiss_nij[i].index(max_category_votes)
    invalid_to_all_ratio = fleiss_nij[i][13] / fleiss_ratings_num
    max_voted_class_subclass = get_class_subclass(max_voted_category)
    max_to_all_ratio = max_category_votes / fleiss_ratings_num
    
    if max_to_all_ratio == 1.0:
        perfect_agreement += 1
    elif max_to_all_ratio >= 0.8:
        great_agreement += 1
    elif max_to_all_ratio > 0.5:
        good_agreement += 1
    elif max_to_all_ratio == 0.5:
        half_agreement += 1
    elif max_to_all_ratio >= 0.2:
        slight_agreement += 1
    else:
        poor_agreement += 1
    
    if invalid_to_all_ratio > 0.3:
        rejected[6].append(hard_photo_paths[i])
    elif max_to_all_ratio < 0.5:
        rejected[max_voted_class_subclass[0]].append(hard_photo_paths[i])
    else:
        new_class_subclass[hard_photo_paths[i]] = max_voted_class_subclass

with open(ALL_IN_PATH, 'r') as all_in_file:
    with open(CORRECTED_PATH, 'w') as corrected_file:
        with open(REJECTED_PATH, 'w') as rejected_file:
            os.makedirs(REJECTED_PHOTOS_FLAT_PATH)
            fleiss_P_ = sum(fleiss_Pi) / fleiss_samples_num
            fleiss_Pe_ = sum([pj ** 2 for pj in fleiss_pj])
            fleiss_K = (fleiss_P_ - fleiss_Pe_) / (1- fleiss_Pe_)
            print("Fleiss Kappa: " + str(fleiss_K))
            print_results()
            all_file_lines = all_in_file.readlines()
            corrected_file.write(all_file_lines[0])
            rejected_flattened = []
            for rejected_x in rejected:
                rejected_flattened += rejected_x
            non_changed = 0
            changed = 0
            for all_file_line in all_file_lines[1:]:
                file_path, _, _, _, author, _ = all_file_line.split(';')
                if file_path in hard_photo_paths:
                    if file_path in rejected_flattened:
                        rejected_file.write(file_path + "\n")
                        rejected_photo_path = REJECTED_PHOTOS_PATH + file_path
                        rejected_photo_dir = os.path.dirname(rejected_photo_path)
                        if not os.path.exists(rejected_photo_dir):
                            os.makedirs(rejected_photo_dir)
                        os.replace(ALL_PHOTOS_PATH + file_path, rejected_photo_path)
                        shutil.copy(rejected_photo_path, REJECTED_PHOTOS_FLAT_PATH + os.path.basename(rejected_photo_path))
                    else:
                        corrected_line_new = ';'.join((file_path, str(new_class_subclass[file_path][0]), str(new_class_subclass[file_path][1]), 
                                                      str(new_extraclass[file_path]), author)) + ";\n"
                        corrected_file.write(corrected_line_new)
                else:
                    corrected_file.write(all_file_line)
                