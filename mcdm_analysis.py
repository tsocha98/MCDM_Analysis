import numpy as np
from pymcdm import methods
from pymcdm.normalizations import scalers
from pymcdm.weights import Entropy, AHP

# 1. Przygotowanie danych (macierz decyzyjna, wagi i typy kryteriów)
matrix = np.array([
    [200, 300, 50, 0.1],  # Alternatywa 1
    [250, 350, 60, 0.2],  # Alternatywa 2
    [300, 400, 70, 0.3],  # Alternatywa 3
])

# Wagi dla kryteriów (sumują się do 1)
weights = np.array([0.4, 0.3, 0.2, 0.1])

# Typy kryteriów: 'max' dla maksymalizowanych, 'min' dla minimalizowanych
criteria_types = ['max', 'max', 'min', 'min']

# 2. Normalizacja danych
normalized_matrix = scalers.MinMaxScaler().fit_transform(matrix)

# 3. Wyznaczenie rankingu za pomocą metod MCDM

# TOPSIS
topsis = methods.TOPSIS()
topsis_result = topsis.evaluate(normalized_matrix, weights, criteria_types)
print("Ranking TOPSIS:", topsis_result)

# SPOTIS
spotis = methods.SPOTIS()
spotis_result = spotis.evaluate(normalized_matrix, weights, criteria_types)
print("Ranking SPOTIS:", spotis_result)

# 4. (Opcjonalnie) Inne metody decyzyjne

# VIKOR
vikor = methods.VIKOR()
vikor_result = vikor.evaluate(normalized_matrix, weights, criteria_types)
print("Ranking VIKOR:", vikor_result)

# PROMETHEE
promethee = methods.PROMETHEE()
promethee_result = promethee.evaluate(normalized_matrix, weights, criteria_types)
print("Ranking PROMETHEE:", promethee_result)

# 5. Wyznaczanie wag - Entropy
entropy_weights = Entropy().fit_transform(normalized_matrix)
print("Wagi wyznaczone metodą Entropii:", entropy_weights)

# 6. Wyznaczanie wag - AHP
ahp_weights = AHP().fit_transform(normalized_matrix)
print("Wagi wyznaczone metodą AHP:", ahp_weights)

# 7. Porównanie wyników i wnioski
print("\nPorównanie wyników:")
print("Ranking TOPSIS:", topsis_result[0])
print("Ranking SPOTIS:", spotis_result[0])
print("Ranking VIKOR:", vikor_result[0])
print("Ranking PROMETHEE:", promethee_result[0])

