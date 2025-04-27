# Punkt 2: Korzystając z pakietu csv wczytać plik xyz zawierający chmurę punktów w przestrzeni 3D. Plik
# powinien zawierać punkty wygenerowane za pomocą skryptu opracowanego w ramach
# ćwiczenia laboratoryjnego nr 1 i zawierać 3 przypadki, reprezentujące powierzchnie płaską
# pionową, płaską poziomą oraz cylindryczną.

import csv
import numpy as np

def load_xyz(filename):
    points = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            if len(row) == 3:
                points.append([float(val) for val in row])
    return np.array(points)
