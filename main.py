import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

from models.sae import SAE
from utils.data_preprocessing import load_ratings, load_movies, binarize
from Recommender import get_user_ratings, recommend_movies_sae

# ===============================
# 1. Configuración del Dispositivo
# ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ===============================
# 2. Cargar y Preprocesar los Datos
# ===============================

# 2.1. Cargar las valoraciones
ratings = load_ratings('data/ml-1m/ratings.dat')
print("Valoraciones cargadas exitosamente.")

# 2.2. Obtener el número de usuarios y películas
nb_users = ratings['UserID'].nunique()
nb_movies = ratings['MovieID'].nunique()
print(f"Número de usuarios: {nb_users}")
print(f"Número de películas: {nb_movies}")

# 2.3. Mapear MovieIDs a índices consecutivos
unique_movie_ids = ratings['MovieID'].unique()
movie_id_to_index = {movie_id: index for index, movie_id in enumerate(unique_movie_ids)}
index_to_movie_id = {index: movie_id for movie_id, index in movie_id_to_index.items()}
print("MovieIDs mapeados a índices consecutivos.")

# 2.4. Convertir los datos a una matriz de usuarios vs películas
data = ratings.copy()
data['UserID'] = data['UserID'] - 1  # Ajustar índices de usuarios para que comiencen en 0
data['MovieIndex'] = data['MovieID'].map(movie_id_to_index)
data = data.dropna(subset=['MovieIndex'])
data['MovieIndex'] = data['MovieIndex'].astype(int)

# Crear una matriz llena de ceros
user_movie_matrix = np.zeros((nb_users, nb_movies))

# Llenar la matriz con las valoraciones
for row in data.itertuples():
    user_movie_matrix[row.UserID, row.MovieIndex] = row.Rating

print("Matriz de usuarios vs películas creada.")

# Convertir a tensores
user_movie_tensor = torch.FloatTensor(user_movie_matrix)
print("Matriz convertida a tensor de PyTorch.")

# 2.5. Dividir en conjunto de entrenamiento y prueba
train_set, test_set = train_test_split(user_movie_tensor, test_size=0.2, random_state=42)
train_set = train_set.to(device)
test_set = test_set.to(device)
print("Conjuntos de entrenamiento y prueba creados.")

# ===============================
# 3. Definir el SAE y el Optimizador
# ===============================

# Definir la arquitectura del SAE
sae = SAE(nb_movies=nb_movies, layers=[128, 64]).to(device)
print("Autoencoder SAE definido.")

# Definir la función de pérdida y el optimizador
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(sae.parameters(), lr=0.001, weight_decay=1e-5)
print("Función de pérdida y optimizador configurados.")

# ===============================
# 4. Crear DataLoaders para Entrenamiento y Prueba
# ===============================

# Definir el tamaño del mini-batch
batch_size = 128

# Crear TensorDatasets
train_dataset = TensorDataset(train_set)
test_dataset = TensorDataset(test_set)

# Crear DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("DataLoaders para entrenamiento y prueba creados.")

# ===============================
# 5. Entrenamiento del SAE con Mini-Batches
# ===============================

nb_epoch = 70
train_loss_per_epoch = []
test_loss_per_epoch = []

print("Inicio del entrenamiento del SAE...")

for epoch in range(1, nb_epoch + 1):
    sae.train()
    train_loss = 0
    s = 0.
    
    for batch in train_loader:
        input = batch[0].to(device)
        target = input.clone()
        
        # Crear máscara para ignorar las películas no valoradas
        mask = target > 0
        
        # Forward pass
        output = sae(input)
        
        # Calcular la pérdida solo en las películas valoradas
        loss = criterion(output[mask], target[mask])
        
        # Backward pass y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        s += 1.
    
    average_train_loss = train_loss / s if s > 0 else 0
    train_loss_per_epoch.append(average_train_loss)
    
    # Evaluación en el conjunto de prueba
    sae.eval()
    test_loss = 0
    s_test = 0.
    with torch.no_grad():
        for batch in test_loader:
            input = batch[0].to(device)
            target = input.clone()
            mask = target > 0
            output = sae(input)
            loss = criterion(output[mask], target[mask])
            test_loss += loss.item()
            s_test += 1.
    
    average_test_loss = test_loss / s_test if s_test > 0 else 0
    test_loss_per_epoch.append(average_test_loss)
    
    print(f"Época: {epoch}, Pérdida de Entrenamiento: {average_train_loss:.4f}, Pérdida de Prueba: {average_test_loss:.4f}")

print("Entrenamiento completado.")

# ===============================
# 6. Visualización de la Pérdida
# ===============================

plt.figure(figsize=(10,5))
plt.plot(range(1, nb_epoch + 1), train_loss_per_epoch, marker='o', label='Pérdida de Entrenamiento')
plt.plot(range(1, nb_epoch + 1), test_loss_per_epoch, marker='x', label='Pérdida de Prueba')
plt.xlabel('Época')
plt.ylabel('Pérdida MSE')
plt.title('Pérdida de Entrenamiento y Prueba por Época')
plt.legend()
plt.grid(True)
plt.savefig('assets/sae_loss.png')  # Guardar la figura
plt.show()
print("Gráfico de pérdida generado y guardado.")

# ===============================
# 7. Guardar el Modelo Entrenado
# ===============================

model_path = 'models/trained_sae.pth'
torch.save(sae.state_dict(), model_path)
print(f'Modelo guardado en {model_path}')

# ===============================
# 8. Generar Recomendaciones para un Usuario
# ===============================

# Cargar títulos de películas
movies = load_movies('data/ml-1m/movies.dat')
print("Archivo 'movies.dat' cargado exitosamente.")

# Filtrar películas para incluir solo aquellas en movie_id_to_index
movies = movies[movies['MovieID'].isin(movie_id_to_index.keys())]
print("Películas filtradas según el conjunto de datos.")

# Obtener valoraciones del usuario
user_ratings_tensor = get_user_ratings(movies[['MovieID', 'Title']], movie_id_to_index).to(device)
print("Valoraciones del usuario obtenidas.")

# Crear entrada para el autoencoder
user_input = user_ratings_tensor.clone()

# Generar recomendaciones
recommended_movies = recommend_movies_sae(
    sae, user_input, movie_id_to_index, index_to_movie_id, movies, num_recommendations=10
)

print("\nLas siguientes películas son recomendadas para ti:")
for idx, row in recommended_movies.iterrows():
    print(f"- {row['Title']} ({row['Genres']})")
