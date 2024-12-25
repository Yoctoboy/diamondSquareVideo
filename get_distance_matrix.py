from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np
from itertools import combinations
from tqdm import tqdm
import pickle

def compute_distance_between_images(mat_a, mat_b): 
    return np.linalg.norm(mat_a - mat_b)

def get_distance_matrix(folder_name): 
    file_names = sorted([f for f in listdir(folder_name) if isfile(join(folder_name, f))])
    index_to_filenames = {i: filename for i, filename in enumerate(file_names)}
    files_amount = len(file_names)
    images_dict = {}
    distance_matrix = np.zeros((files_amount, files_amount))
    for filename in tqdm(file_names, desc="Loading image files"):
        im_frame = Image.open(f"{folder_name}/{filename}")
        np_frame = np.array(im_frame, dtype=np.int64)
        images_dict[filename] = np_frame
        
    # compute distance matrix
    for image_a_index, image_b_index in tqdm(list(combinations(range(files_amount), r=2)), desc="Computing distance matrix"):
        image_a_name, image_b_name = index_to_filenames[image_a_index], index_to_filenames[image_b_index]
        image_a, image_b = images_dict[image_a_name], images_dict[image_b_name]
        distance = compute_distance_between_images(image_a, image_b)
        distance_matrix[image_a_index][image_b_index] = distance
        distance_matrix[image_b_index][image_a_index] = distance
    return distance_matrix, index_to_filenames
    
    

distance_matrix, index_to_filenames = get_distance_matrix("images")
with open("distance_matrix.pkl", "wb") as f:
    pickle.dump(distance_matrix, f)

with open("index_to_filenames.pkl", "wb") as f:
    pickle.dump(index_to_filenames, f)

print("loading pickled stuff")
with open("distance_matrix.pkl", "rb") as f:
    print(pickle.load(f))

with open("index_to_filenames.pkl", "rb") as f:
    print(pickle.load(f))