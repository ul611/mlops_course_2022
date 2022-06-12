#!/usr/bin/env python
# coding: utf-8

# R2 score is used as metric.
# Предсказание цены дома в США (без использования деревьев и DL)
from collections import Counter
import itertools
import json
import time
from tqdm import tqdm

import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, Lasso

# from sklearn.metrics import r2_score
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

CORR_THRES = 0.8
N = 4
FOLDER = "./data/"
RANDOM_STATE = 42
SCROLL_PAUSE_TIME = 1

# #### Импортируем исходные данные

X_train = pd.read_csv(f"{FOLDER}x_train.csv")
X_test = pd.read_csv(f"{FOLDER}x_test.csv")
y_train = pd.read_csv(f"{FOLDER}y_train.csv")


# Преобразуем в логарифм. Далее будем предсказывать логарифм цены, и затем потенцируем предсказание

y_train_log = np.log(y_train)

# #### Соберем все данные в один датасет
# убираем явный выброс по критерию bedrooms == 33, при этом площадь всего 1620 sqft

X_train.query("bedrooms > 30")


X_train["dataset"] = "train"
X_test["dataset"] = "test"

data = pd.concat(
    [pd.concat([X_train.drop(2113), y_train_log.drop(2113)], axis=1), X_test]
)

# ##### Признаки
# так как мало значений для view, sqft_basement, yr_renovated, делаем их булевыми

for col in ["view", "sqft_basement", "yr_renovated"]:
    data[col + "_bool"] = (data[col] > 0).astype(int)
    data.drop(col, axis=1, inplace=True)

# переводим "квадратные" признаки в сторону квадрата (извлекаем корень) и логарифмируем

for col in data.columns:
    if col.startswith("sqft") and not col.endswith("bool"):
        data[col[2:]] = np.log(np.sqrt(data[col]))
        data.drop(col, axis=1, inplace=True)

# извлекаем дату продажи и переводим ее в дни, начиная с самого раннего дня продажи

data.date = pd.to_datetime(data.date.str.split("T", expand=True)[0], format="%Y%m%d")
min_data = data.date.min()
data.date = (data.date - min_data).apply(lambda x: x.days)

# ### Категориальная переменная zipcode
# Собираем общую информацию по каждому zipcode

d_zipcodes_info = {}
info_fields = [
    "Population",
    "Population Density",
    "Housing Units",
    "Median Home Value",
    "Land Area",
    "Water Area",
    "Occupied Housing Units",
    "Median Household Income",
    "Median Age",
]

driver = webdriver.Chrome()

for zipcode in tqdm(data.zipcode.unique()):
    url = f"https://www.unitedstateszipcodes.org/{zipcode}/"
    driver.get(url)

    time.sleep(SCROLL_PAUSE_TIME)
    list_p_element = driver.find_elements(By.XPATH, "//tr")

    d_zipcodes_info[zipcode] = {}
    fulltxt = []

    for i_start, el in enumerate(list_p_element):
        txt = el.text
        if txt.startswith("Population"):
            break

    i = -1
    for el in list_p_element:
        i += 1
        if i < i_start or i_start + 8 < i:
            continue
        txt = el.text
        if txt:
            fulltxt += [txt]

    for txt, field in zip(fulltxt, info_fields):
        value = float(
            txt.split(field)[1]
            .replace(":", "")
            .strip()
            .split()[0]
            .replace(",", "")
            .replace("$", "")
        )
        d_zipcodes_info[zipcode][field] = value

for field in info_fields:
    colname = "_".join(field.split())
    data[colname] = data.zipcode.apply(lambda x: d_zipcodes_info[x][field])

# write dict into file

d_zipcodes_modified_keys = {str(key): val for key, val in d_zipcodes_info.items()}

with open("d_zipcodes_info.json", "w") as fd:
    json.dump(d_zipcodes_modified_keys, fd)


# load dict from file

with open("d_zipcodes_info.json", "к") as fd:
    json.load(d_zipcodes_modified_keys, fd)

d_zipcodes_info = {int(key): val for key, val in d_zipcodes_info.items()}

# немного преобразований переменных

data["Population"] = np.sqrt(data["Population"])
data["pers_houses_occupied"] = data.Occupied_Housing_Units / data.Housing_Units

data.to_csv("data_all.csv", index=False)

# #### Делим данные на тест и трейн
# Переделать

X_train = data.query("~price.isna()").drop(
    [
        #'lat', 'long', 'zipcode',
        "dataset",
        "price",
        "combined",
    ],
    axis=1,
)
X_test = data.query("price.isna()").drop(
    [
        #'lat', 'long', 'zipcode',
        "dataset",
        "price",
        "combined",
    ],
    axis=1,
)
y_train = data.query("~price.isna()")[["price"]]


# #### Подготовка категориальных данных

cat_features = [
    "zipcode",
    #'view_bool', 'sqft_basement_bool', 'waterfront', 'yr_renovated_bool'
]

for col in cat_features:
    le = LabelEncoder()
    le.fit(X_train[col])
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# #### Удаляем коррелирующие признаки

correlated_features = pd.DataFrame(
    X_train.corr(method="pearson")
    .abs()
    .unstack()
    .reset_index()
    .query("level_0 != level_1")
    .sort_values(0, ascending=False)
)

correlated_features["pairs"] = correlated_features.apply(
    lambda x: tuple(sorted([x.level_0, x.level_1])), axis=1
)
correlated_features = (
    correlated_features.drop_duplicates(subset="pairs")
    .drop(["level_0", "level_1"], axis=1)
    .rename(columns={0: "corr_value"})
)

# уберем из тех пар, у которых корреляция > corr те признаки, которые чаще других встречаются в парах
# коррелирующих признаков

f_to_drop = []
most_correlated_df = correlated_features.query("corr_value > @CORR_THRES").copy()

# пока есть пары в датасете с самыми коррелированными фичами
while len(most_correlated_df.index):
    # список всех оставшихся коррелированных фичей в датасете
    most_corr = list(itertools.chain(*most_correlated_df.pairs))
    # самая часто встречающаяся фича
    f = np.unique(sorted(most_corr, key=lambda x: Counter(most_corr)[x]))[-1]
    f_to_drop += [f]
    most_correlated_df["f_in_pairs"] = most_correlated_df.pairs.apply(lambda x: f in x)
    most_correlated_df = most_correlated_df.query("not f_in_pairs")

X_train.drop(f_to_drop, axis=1, inplace=True)
X_test.drop(f_to_drop, axis=1, inplace=True)


# #### Делим датасет на трейн и валидацию

X_train_train, X_val, y_train_train, y_val = train_test_split(
    X_train, y_train, random_state=RANDOM_STATE
)

# #### Шкалируем переменные с помощью Standard Scaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_train_scaled = pd.DataFrame(
    scaler.transform(X_train_train), columns=X_train_train.columns
)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# #### Поиск лучших параметров


def find_best_params(X_train_scaled, y_train):

    # поиск лучших параметров для регрессии
    grid_search_lasso = GridSearchCV(
        Lasso(),
        {
            "max_iter": range(10, 150, 10),
            "alpha": np.logspace(-9, -5),
            "random_state": [RANDOM_STATE],
        },
        scoring="r2",
    )
    grid_search_lasso.fit(X_train_scaled, y_train)

    # поиск лучших параметров для knn
    grid_search_knn = GridSearchCV(
        KNeighborsRegressor(),
        {
            "metric": [
                "cosine",
                "euclidean",
                "manhattan",
                "chebyshev",
                "hamming",
                "canberra",
                "braycurtis",
            ],
            "weights": ["distance"],
            "n_neighbors": range(3, 8),
        },
        scoring="r2",
    )
    grid_search_knn.fit(X_train_scaled, y_train)
    return grid_search_lasso, grid_search_knn


# get a stacking ensemble of models

grid_search_lasso, grid_search_knn = find_best_params(X_train_scaled, y_train)


def get_stacking():
    # define the base models
    level0 = list()
    level0.append(("knn", KNeighborsRegressor(**grid_search_knn.best_params_)))
    level0.append(("lasso", Lasso(**grid_search_lasso.best_params_)))
    # define meta learner model
    level1 = LinearRegression()
    # define the stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
    return model


model = get_stacking()
# fit the model
model.fit(X_train_scaled, y_train)
# make a prediction
y_pred = model.predict(X_test_scaled)


submission = pd.DataFrame(np.exp(y_pred), columns=["price"]).reset_index()
submission.columns = ["Id", "price"]
submission.set_index("Id").to_csv(f"solution-{N}-Uliana.csv")
