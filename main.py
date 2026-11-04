import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

data = fetch_california_housing(as_frame=True)
df = data.frame

features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms']
X = df[features]
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# ЗАВДАННЯ 1:
# ==========================================
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)
r2_tree = r2_score(y_test, y_pred_tree)

print(f"Результат Завдання 1 (R² дерева рішень): {r2_tree:.4f}")
print("\n" + "="*50 + "\n")


# ==========================================
# ЗАВДАННЯ 2:
# ==========================================

tree_d4 = DecisionTreeRegressor(max_depth=4, random_state=42)
tree_d4.fit(X_train, y_train)
r2_tree_d4 = r2_score(y_test, tree_d4.predict(X_test))

rf_model = RandomForestRegressor(max_depth=4, random_state=42)
rf_model.fit(X_train, y_train)
r2_rf = r2_score(y_test, rf_model.predict(X_test))

results_2 = pd.DataFrame({
    'Модель': ['DecisionTreeRegressor (max_depth=4)', 'RandomForestRegressor (max_depth=4)'],
    'Тест R²': [r2_tree_d4, r2_rf]
})

print("Результати Завдання 2:")
print(results_2.to_string(index=False))
print("\n" + "="*50 + "\n")


# ==========================================
# ЗАВДАННЯ 3:
# ==========================================
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
r2_gb = r2_score(y_test, gb_model.predict(X_test))

results_3 = pd.DataFrame({
    'Модель': ['RandomForestRegressor', 'GradientBoostingRegressor'],
    'Тест R²': [r2_rf, r2_gb]
})

print("Результати Завдання 3:")
print(results_3.to_string(index=False))
