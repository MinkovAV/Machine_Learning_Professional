# ============================================================
# ЧАСТЬ 1
# ============================================================
import os
import zipfile
import requests
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# sklearn boosting
from sklearn.ensemble import GradientBoostingRegressor
# external libs
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

plt.style.use("ggplot")
sns.set(font_scale=1.0)

# ------------------------------------------------------------
# 1. Скачивание датасета
# ------------------------------------------------------------

URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
ZIP_FILE = "online_retail.zip"
DATA_DIR = "online_retail_data"

if not os.path.exists(ZIP_FILE):
    print("Скачиваю датасет...")
    r = requests.get(URL)
    with open(ZIP_FILE, "wb") as f:
        f.write(r.content)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
    zip_ref.extractall(DATA_DIR)

print("Файлы распакованы:")
print(os.listdir(DATA_DIR))

# ------------------------------------------------------------
# 2. Загрузка данных
# ------------------------------------------------------------

# Обычно внутри файл Online Retail.xlsx
file_path = os.path.join(DATA_DIR, "Online Retail.xlsx")

df = pd.read_excel(file_path)

print("\nРазмер датасета:", df.shape)

# ------------------------------------------------------------
# 3. Первичный обзор
# ------------------------------------------------------------

print("\nПервые строки:")
print(df.head())

print("\nИнформация:")
print(df.info())

print("\nПропуски:")
print(df.isnull().sum())

print("\nОписательная статистика:")
print(df.describe(include="all"))

# ------------------------------------------------------------
# 4. Подготовка данных
# ------------------------------------------------------------

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# создаём выручку по строке
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# признаки времени
df["Year"] = df["InvoiceDate"].dt.year
df["Month"] = df["InvoiceDate"].dt.month
df["Day"] = df["InvoiceDate"].dt.day
df["Hour"] = df["InvoiceDate"].dt.hour
df["Weekday"] = df["InvoiceDate"].dt.day_name()

# ------------------------------------------------------------
# 5. Анализ признаков
# ------------------------------------------------------------

print("\nКоличество стран:", df["Country"].nunique())
print("Количество клиентов:", df["CustomerID"].nunique())
print("Количество товаров:", df["StockCode"].nunique())

# ------------------------------------------------------------
# 6. Проверим отменённые закаы
# ------------------------------------------------------------

cancelled = df["InvoiceNo"].astype(str).str.startswith("C").sum()
print("\nКоличество отменённых заказов:", cancelled)

# ------------------------------------------------------------
# 7. Графики
# ------------------------------------------------------------

plt.figure(figsize=(10,5))
sns.histplot(df["Quantity"], bins=100)
plt.xlim(-50, 200)
plt.title("Распределение Quantity")
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(df["UnitPrice"], bins=100)
plt.xlim(0, 50)
plt.title("Распределение UnitPrice")
plt.show()


# 8. Выручка по мсяцам
monthly_revenue = df.groupby(["Year","Month"])["Revenue"].sum().reset_index()
monthly_revenue["Period"] = monthly_revenue["Year"].astype(str) + "-" + monthly_revenue["Month"].astype(str)

plt.figure(figsize=(14,5))
sns.lineplot(data=monthly_revenue, x="Period", y="Revenue", marker="o")
plt.xticks(rotation=60)
plt.title("Выручка по месяцам")
plt.show()


# 9. ТОП стран по выручке
country_rev = (
    df.groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,6))
sns.barplot(x=country_rev.values, y=country_rev.index)
plt.title("ТОП-10 стран по выручке")
plt.show()


# 10. ТОП товаров
top_products = (
    df.groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,6))
sns.barplot(x=top_products.values, y=top_products.index)
plt.title("ТОП-10 товаров по количеству продаж")
plt.show()


# 11. Корреляции числовых признаков
num_cols = ["Quantity", "UnitPrice", "Revenue", "Month", "Hour"]
corr = df[num_cols].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Корреляция признаков")
plt.show()


# 12. Поведение клиентов (RFM-lite)
customer_stats = df.groupby("CustomerID").agg({
    "InvoiceNo": "nunique",
    "Revenue": "sum",
    "Quantity": "sum"
}).rename(columns={
    "InvoiceNo": "Orders",
    "Revenue": "TotalRevenue",
    "Quantity": "TotalItems"
})

print("\nТоп клиентов по выручке:")
print(customer_stats.sort_values("TotalRevenue", ascending=False).head(10))



# ============================================================
# ЧАСТЬ 2
# ============================================================
# ------------------------------------------------------------
# 13. Базовая очистка
# ------------------------------------------------------------

# удалим полные дубликаты
df = df.drop_duplicates()

# переведём дату
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# удалим строки с пропусками в важных колонках
df = df.dropna(subset=["CustomerID", "Description"])

# удалим строки с невалидными значениями
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]

# ------------------------------------------------------------
# 14. Feature Engineering
# ------------------------------------------------------------

# целевая бизнес-метрика
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# признаки даты
df["Year"] = df["InvoiceDate"].dt.year
df["Month"] = df["InvoiceDate"].dt.month
df["Day"] = df["InvoiceDate"].dt.day
df["Hour"] = df["InvoiceDate"].dt.hour
df["Weekday"] = df["InvoiceDate"].dt.weekday
df["Weekend"] = df["Weekday"].isin([5, 6]).astype(int)

# чек отменённый или нет
df["IsCancelled"] = df["InvoiceNo"].astype(str).str.startswith("C").astype(int)

# длина описания товара
df["DescLength"] = df["Description"].astype(str).apply(len)

# средняя цена за единицу клиента
df["CustomerMeanPrice"] = df.groupby("CustomerID")["UnitPrice"].transform("mean")

# сколько раз клиент встречается в датасете
df["CustomerFreq"] = df.groupby("CustomerID")["InvoiceNo"].transform("count")

# ------------------------------------------------------------
# 15. Удаляем лишние признаки
# ------------------------------------------------------------

drop_cols = [
    "InvoiceDate",   # дата уже разложена
]

df = df.drop(columns=drop_cols)

# ------------------------------------------------------------
# 16. Label Encoding категориальных переменных
# ------------------------------------------------------------

cat_cols = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Country"
]

encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ------------------------------------------------------------
# 17. Формируем X и y
# ------------------------------------------------------------
# Пример: будем предсказывать Revenue

target = "Revenue"

X = df.drop(columns=[target])
y = df[target]

# ------------------------------------------------------------
# 18. Train / Test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------------------------------------------
# 19. Проверка
# ------------------------------------------------------------

print("Размер исходного датасета:", df.shape)
print("Train:", X_train.shape, y_train.shape)
print("Test :", X_test.shape, y_test.shape)

print("\nКолонки признаков:")
print(X.columns.tolist())

print("\nПример train:")
print(X_train.head())

# ============================================================
# ЧАСТЬ 3
# ============================================================

# ------------------------------------------------------------
# 20. Загрузка подготовленного датасета
# ------------------------------------------------------------

df = pd.read_excel("online_retail_data/Online Retail.xlsx")

# ------------------------------------------------------------
# 21. PREPROCESSING
# ------------------------------------------------------------

from sklearn.preprocessing import LabelEncoder

df = df.drop_duplicates()
df = df.dropna(subset=["CustomerID", "Description"])

df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

df["Revenue"] = df["Quantity"] * df["UnitPrice"]

df["Year"] = df["InvoiceDate"].dt.year
df["Month"] = df["InvoiceDate"].dt.month
df["Day"] = df["InvoiceDate"].dt.day
df["Hour"] = df["InvoiceDate"].dt.hour
df["Weekday"] = df["InvoiceDate"].dt.weekday
df["Weekend"] = df["Weekday"].isin([5,6]).astype(int)

df["DescLength"] = df["Description"].astype(str).apply(len)
df["CustomerFreq"] = df.groupby("CustomerID")["InvoiceNo"].transform("count")
df["CustomerMeanPrice"] = df.groupby("CustomerID")["UnitPrice"].transform("mean")

df = df.drop(columns=["InvoiceDate"])

cat_cols = ["InvoiceNo", "StockCode", "Description", "Country"]

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# ------------------------------------------------------------
# 22. TRAIN / TEST
# ------------------------------------------------------------

target = "Revenue"

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ------------------------------------------------------------
# 23. ФУНКЦИЯ ОЦЕНКИ
# ------------------------------------------------------------

def evaluate(model, name):
    pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    print(f"{name}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R2  : {r2:.4f}")
    print("-"*40)

    return [name, rmse, mae, r2]

# ============================================================
# Часть A. MODELS WITH DEFAULT PARAMS
# ============================================================

results = []

models = {
    "Sklearn GBM":
        GradientBoostingRegressor(random_state=42),

    "XGBoost":
        XGBRegressor(random_state=42, verbosity=0),

    "CatBoost":
        CatBoostRegressor(verbose=0, random_state=42),

    "LightGBM":
        LGBMRegressor(random_state=42)
}

print("="*50)
print("DEFAULT MODELS")
print("="*50)

for name, model in models.items():
    model.fit(X_train, y_train)
    results.append(evaluate(model, name))

# ------------------------------------------------------------
# Кто лидирует?
# ------------------------------------------------------------

res_df = pd.DataFrame(results, columns=["Model", "RMSE", "MAE", "R2"])
print("\nРейтинг default моделей:")
print(res_df.sort_values("RMSE"))

# ============================================================
# Часть B. HYPERPARAMETER TUNING
# ============================================================

tuned_results = []

# ------------------------------------------------------------
# 1. sklearn GBM
# ------------------------------------------------------------

params_sklearn = {
    "n_estimators": [100, 200],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.8, 1.0]
}

search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    params_sklearn,
    n_iter=5,
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

search.fit(X_train, y_train)
best_sklearn = search.best_estimator_

# ------------------------------------------------------------
# 2. XGBoost
# ------------------------------------------------------------

params_xgb = {
    "n_estimators": [200, 400],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [4, 6, 8],
    "subsample": [0.8, 1.0]
}

search = RandomizedSearchCV(
    XGBRegressor(random_state=42, verbosity=0),
    params_xgb,
    n_iter=5,
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

search.fit(X_train, y_train)
best_xgb = search.best_estimator_

# ------------------------------------------------------------
# 3. CatBoost
# ------------------------------------------------------------

params_cat = {
    "iterations": [300, 500],
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1]
}

search = RandomizedSearchCV(
    CatBoostRegressor(verbose=0, random_state=42),
    params_cat,
    n_iter=5,
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

search.fit(X_train, y_train)
best_cat = search.best_estimator_

# ------------------------------------------------------------
# 4. LightGBM
# ------------------------------------------------------------

params_lgbm = {
    "n_estimators": [200, 400],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [4, 6, 8],
    "subsample": [0.8, 1.0]
}

search = RandomizedSearchCV(
    LGBMRegressor(random_state=42),
    params_lgbm,
    n_iter=5,
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

search.fit(X_train, y_train)
best_lgbm = search.best_estimator_

# ============================================================
# PART C. EVALUATE TUNED MODELS
# ============================================================

print("\n" + "="*50)
print("TUNED MODELS")
print("="*50)

tuned_results.append(evaluate(best_sklearn, "Sklearn Tuned"))
tuned_results.append(evaluate(best_xgb, "XGBoost Tuned"))
tuned_results.append(evaluate(best_cat, "CatBoost Tuned"))
tuned_results.append(evaluate(best_lgbm, "LightGBM Tuned"))

final_df = pd.DataFrame(
    tuned_results,
    columns=["Model", "RMSE", "MAE", "R2"]
)

print("\nФинальный рейтинг:")
print(final_df.sort_values("RMSE"))

# ============================================================
# ИТОГОВЫЕ ВЫВОДЫ
# ============================================================

print("""
Сейчас лучший выбор: XGBoost
""")