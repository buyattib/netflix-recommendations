import numpy as np
import os
import pandas as pd
from datetime import datetime

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
        dates = []

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
                isoformat_date_string = separated_line[2]
                date = datetime.fromisoformat(isoformat_date_string)
                dates.append(date.timestamp())
        
            if i == 0:
                print(f"first: {movie_id}")
            
            i += 1
        
        print(f"last: {movie_id}")
            
        edges = np.array([movies, users, ratings, dates]).T
        
        del movies
        del users
        del ratings
        del dates
        
        df = pd.DataFrame(edges, columns=["movies", "users", "ratings", "dates"])
        dfs.append(df)
        
        del edges
        del df

all_dfs = pd.concat(dfs)
all_dfs = all_dfs[["users", "movies", "ratings", "dates"]]
all_dfs.sort_values("users", inplace=True)
print(all_dfs)

all_dfs = all_dfs.to_numpy()
np.save(edges_dir + "/all_edges.npy", all_dfs)
