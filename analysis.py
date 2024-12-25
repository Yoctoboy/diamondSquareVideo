import pickle

with open("distance_matrix.pkl", "rb") as f:
    distance_matrix = pickle.load(f)

with open("index_to_filenames.pkl", "rb") as f:
    index_to_filenames = pickle.load(f)


min_distance = min((min(y for y in row if y!=0) for row in distance_matrix))
def find_closest_images():
    for i in range(1000):
        for j in range(1000):
            if(distance_matrix[i][j] == min_distance):
                print(index_to_filenames[i], index_to_filenames[j])
                return
            
find_closest_images()
            