import torch
import pandas as pd
import numpy as np
from utils.data_preprocessing import binarize


def get_user_personal_data():
    """
    Solicita datos personales al usuario y los codifica en un tensor.

    Returns:
        torch.Tensor: Tensor con las características codificadas del usuario.
    """
    try:
        occupation_list = [
            "Otro o no especificado", "Académico/Educador", "Artista", "Administrativo/Clerical",
            "Estudiante universitario", "Atención al cliente", "Médico/Cuidado de la salud",
            "Ejecutivo/Gerencial", "Agricultor", "Ama de casa", "Estudiante de secundaria",
            "Abogado", "Programador", "Jubilado", "Ventas/Marketing", "Científico",
            "Autónomo", "Técnico/Ingeniero", "Oficios/Artesano", "Desempleado", "Escritor"
        ]

        print("\nPor favor, ingresa tus datos personales:")
        gender = input("Género (M/F): ").strip().upper()

        # Validar y solicitar edad
        age_input = input("Edad: ").strip()
        try:
            age = int(age_input)
        except ValueError:
            print("Edad inválida. Se asignará la edad promedio del dataset.")
            age = 25  # Edad promedio del dataset

        # Mostrar lista de ocupaciones
        print("\nSelecciona tu ocupación según el siguiente listado:")
        for idx, occupation in enumerate(occupation_list):
            print(f"{idx}: {occupation}")

        # Validar y solicitar ocupación
        occupation_input = input("\nOcupación (ID numérico entre 0 y 20): ").strip()
        try:
            occupation = int(occupation_input)
            if not (0 <= occupation <= 20):
                print("Ocupación fuera del rango. Se asignará 'Otro'.")
                occupation = 0  # 'Otro' como predeterminado
        except ValueError:
            print("Ocupación inválida. Se asignará 'Otro'.")
            occupation = 0  # 'Otro' como predeterminado

        # Codificar género
        gender_dict = {'M': 0, 'F': 1}
        gender_one_hot = [0, 0]
        if gender in gender_dict:
            gender_one_hot[gender_dict[gender]] = 1
        else:
            print("Género no reconocido. Se asignará 'M'.")
            gender_one_hot[0] = 1

        # Codificar edad en grupos (según MovieLens)
        age_groups = [1, 18, 25, 35, 45, 50, 56]
        age_one_hot = [0] * len(age_groups)
        for idx, group in enumerate(age_groups):
            if age <= group:
                age_one_hot[idx] = 1
                break
        else:
            age_one_hot[-1] = 1  # Mayor de 56

        # Codificar ocupación
        occupation_one_hot = [0] * 21
        occupation_one_hot[occupation] = 1

        # Combinar todas las características
        user_features = gender_one_hot + age_one_hot + occupation_one_hot
        user_features_tensor = torch.FloatTensor(user_features).unsqueeze(0)

        return user_features_tensor

    except Exception as e:
        print("Ocurrió un error al procesar tus datos personales.")
        raise e




def get_user_ratings(movie_titles, movie_id_to_index):
    """
    Solicita al usuario que califique películas seleccionadas aleatoriamente.

    Args:
        movie_titles (pd.DataFrame): DataFrame con los títulos de las películas.
        movie_id_to_index (dict): Diccionario para mapear MovieID a índices.

    Returns:
        user_ratings (torch.Tensor): Tensor con las valoraciones del usuario.
    """
    try:
        print("\nPor favor, califica las siguientes películas (1-5 estrellas). Si no has visto la película, presiona Enter para omitir.")
        selected_movies = movie_titles.sample(20, random_state=np.random.randint(0, 10000)).reset_index(drop=True)
        user_ratings = torch.zeros(len(movie_id_to_index))  # Inicializar con 0 (no valoradas)

        for idx, row in selected_movies.iterrows():
            movie_id = row['MovieID']
            movie_title = row['Title']
            rating_input = input(f"{idx+1}. {movie_title}: ").strip()
            if rating_input.isdigit():
                rating = int(rating_input)
                if 1 <= rating <= 5:
                    movie_index = movie_id_to_index.get(movie_id)
                    if movie_index is not None:
                        user_ratings[movie_index] = rating
                    else:
                        print(f"MovieID {movie_id} no está en el diccionario. Se omitirá esta valoración.")
                else:
                    print("Valoración no válida. Se omitirá esta película.")
            else:
                print("Película omitida.")

        return user_ratings.unsqueeze(0)
    except Exception as e:
        print("Ocurrió un error al procesar tus valoraciones de películas.")
        raise e
    

def recommend_movies_sae(sae, user_input, movie_id_to_index, index_to_movie_id, movies, num_recommendations=10):
    """
    Genera recomendaciones para el usuario basado en sus valoraciones.

    Args:
        sae (SAE): Modelo entrenado de SAE.
        user_input (torch.Tensor): Tensor con las valoraciones del usuario.
        movie_id_to_index (dict): Diccionario para mapear MovieID a índices.
        index_to_movie_id (dict): Diccionario para mapear índices a MovieID.
        movies (pd.DataFrame): DataFrame con los datos de las películas.
        num_recommendaciones (int): Número de películas a recomendar.

    Returns:
        recommended_movies (pd.DataFrame): DataFrame con las películas recomendadas.
    """
    sae.eval()
    with torch.no_grad():
        output = sae(user_input)
        # Ignorar películas ya valoradas
        output[user_input > 0] = 0
        # Obtener índices de las películas con mayor puntuación
        _, recommended_indices = torch.topk(output, k=num_recommendations)
        recommended_indices = recommended_indices[0].cpu().numpy()

    # Mapear índices a MovieIDs
    recommended_movie_ids = [index_to_movie_id[idx] for idx in recommended_indices]

    # Obtener títulos de las películas recomendadas
    recommended_movies = movies[movies['MovieID'].isin(recommended_movie_ids)]

    return recommended_movies
