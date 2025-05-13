import joblib
from sklearn.ensemble import RandomForestClassifier

# Exempeldataset: [max_lutning, riskprocent, total längd i meter]
X = [
    [2.0, 5.0, 300],     # lätt
    [3.5, 10.0, 400],    # lätt
    [5.0, 25.0, 600],    # medel
    [6.5, 35.0, 800],    # medel
    [8.0, 60.0, 1200],   # svår
    [10.0, 80.0, 1500],  # svår
]

# Klassetiketter: 0 = lätt, 1 = medel, 2 = svår
y = [0, 0, 1, 1, 2, 2]

# Etikettkarta
label_map = {"lätt": 0, "medel": 1, "svår": 2}

# Träna modell
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Spara till fil
joblib.dump({"model": model, "label_map": label_map}, "ml_modell.pkl")

print("✅ Modell sparad som 'ml_modell.pkl'")
