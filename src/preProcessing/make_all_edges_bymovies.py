import numpy as np
import os
import pandas as pd

data_dir = "./data"
raw_dir = data_dir + "/raw"
edges_dir = data_dir + "/edges"

data_files = [file_name for file_name in os.listdir(raw_dir) if "combined" in file_name]
dfs = []

for index, data_file in enumerate(data_files):
    file_path = raw_dir + "/" + data_file
    
    with open(file_path, 'r') as file:
        movies = []
        users = []
        ratings = []

        i = 0
        for line in file:
            line = line.strip("\n")
            if ":" in line:
                movie_id = int(line.strip(":"))
                
            else:
                movies.append(movie_id)
                separated_line = line.split(",")
                users.append(int(separated_line[0]))
                ratings.append(float(separated_line[1]))
        
            if i == 0:
                print(f"first: {movie_id}")
            
            i += 1
        
        print(f"last: {movie_id}")
            
        edges = np.array([movies, users, ratings]).T
        
        del movies
        del users
        del ratings
        
        df = pd.DataFrame(edges, columns=["movies", "users", "ratings"])
        dfs.append(df)
        
        del edges
        del df

all_dfs = pd.concat(dfs)
complete_edges = all_dfs.to_numpy()
np.save(edges_dir + "/all_edges_bymovies.npy", complete_edges)

print(all_dfs)