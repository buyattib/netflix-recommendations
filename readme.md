Proyecto final de redes.

En la carpeta src están las carpetas con codigos.

- preProcessing:
    -- make_all_edges_bymovies.py: Toma los archivos originales (data/raw) y lo convierte en un array de 3 columnas (movies, users, ratings).
    
    -- make_all_edges_byusers.py: Toma el array ordenado segun las peliculas (data/edges/all_edges_bymovies.npy) y lo ordena segun usuarios (data/edges/all_edges.npy)
    
    -- make_gt3_and_lt3_edges.py: Toma el array ordenado segun usuarios (data/edges/all_edges.npy) y lo separa en dos arrays de enlaces segun tengan ratings >= 3 (data/edges/gt3.npy) o < 3 (data/edges/lt3.npy)
    
    -- make_sample_network_gt3.py: Del array de enlaces para ratings >= 3 (data/edges/gt3.npy), toma los usuarios de grado < 1500 y luego hace un sampleo aleatorio de 5000 usuarios. Con esos 5000 usuarios de grado < 1500 arma la red con todos sus enlaces. Se guarda en data/networks/sample_gt3.graphmlz. Esta red tiene un atributo en los vertices llamado "original_name" que es el id o nombre original del dataset. Al guardar en .graphmlz se guarda un atributo en los vertices llamado "id" automaticamente, pero no se usa.
    
    -- make_sample_network_lt3.py: Del array de enlaces para ratings < 3 (data/edges/lt3.npy), toma los usuarios de grado < 1500 y luego hace un sampleo aleatorio de 5000 usuarios. Con esos 5000 usuarios de grado < 1500 arma la red con todos sus enlaces. Se guarda en data/networks/sample_lt3.graphmlz. Esta red tiene un atributo en los vertices llamado "original_name" que es el id o nombre original del dataset. Al guardar en .graphmlz se guarda un atributo en los vertices llamado "id" automaticamente, pero no se usa.
    
    -- make_train_test_gt3.py: Toma la red de 5000 usuarios de grado < 1500 (data/networks/sample_gt3.graphmlz) y de forma aleatoria elimina el 10% de enlaces. Arma la red con el 90% restante y la guarda en data/networks/gt3_sample_90perc_edges.graphmlz. El 10% de enlaces eliminados los guarda en data/networks/gt3_sample_10perc_edges.npy como un array. Al guardar en .graphmlz se guarda un atributo en los vertices llamado "id" automaticamente, pero no se usa.
    
    -- make_train_test_lt3.py: Toma la red de 5000 usuarios de grado < 1500 (data/networks/sample_lt3.graphmlz) y de forma aleatoria elimina el 10% de enlaces. Arma la red con el 90% restante y la guarda en data/networks/lt3_sample_90perc_edges.graphmlz. El 10% de enlaces eliminados los guarda en data/networks/lt3_sample_10perc_edges.npy como un array. Al guardar en .graphmlz se guarda un atributo en los vertices llamado "id" automaticamente, pero no se usa.

- weightedProjections:
    -- functions.py: Funciones para hacer las proyecciones pesadas.
    
    -- make_gt3_projections.py: Usa las funciones para armar las matrices de pesos de probs, heats y las hibridas usando la red gt3 sampleada con 5000 usuarios de grado < 1500 y con el 90% de enlaces. Se guardan las matrices en data/weightedProjections/gt3.

    -- make_lt3_projections.py: Usa las funciones para armar las matrices de pesos de probs, heats y las hibridas usando la red lt3 sampleada con 5000 usuarios de grado < 1500 y con el 90% de enlaces. Se guardan las matrices en data/weightedProjections/lt3.

- recommendations:
    - weightedProjections:
        -- functions.py: Funcion para hacer las recomendaciones. Toma la matriz de pesos a usar y la matriz de incidencia. Devuelve dos matrices de dimensiones iguales a la matriz de incidencia. En la fila i se encuentran las peliculas recomendadas para el usuario i, ordenadas de mayor a menor. Las dos matrices devueltas se diferencian en que una tiene todas las peliculas para cada usuario y otra solo las que no vio. Es decir, en la fila i estaran las peliculas ordenadas de mas a menos recomendada para el usuario i. En una matriz estaran todas y en la otra solo las que no vio.

        - gt3: directorio donde guardo los codigos para la red gt3 de 5000 usuarios con el 90% de enlaces
            -- heats_recommendations.py, probs_recommendations.py, hybrid_recommendations.py: Usa la funcion de recomendaciones aplicada a cada matriz de pesos. Las matrices de pesos usadas fueron las que se calcularon usando las redes sampleadas de 5k usuarios y el 90% de enlaces. Se guardan las recomendaciones en la carpeta data/recommendations/weightedProjections/gt3.

        - lt3: directorio donde guardo los codigos para la red lt3 de 5000 usuarios con el 90% de enlaces
            -- heats_recommendations.py, probs_recommendations.py, hybrid_recommendations.py: Usa la funcion de recomendaciones aplicada a cada matriz de pesos. Las matrices de pesos usadas fueron las que se calcularon usando las redes sampleadas de 5k usuarios y el 90% de enlaces. Se guardan las recomendaciones en la carpeta data/recommendations/weightedProjections/lt3.

