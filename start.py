import src

RANDOM_STATE = 42
SCROLL_PAUSE_TIME = 1
CORR_THRES = 0.8
INPUT_FILE_X_TRAIN = "./data/raw/x_train.csv"
INPUT_FILE_X_TEST = "./data/raw/x_test.csv"
INPUT_FILE_Y_TRAIN = "./data/raw/y_train.csv"
PATH_ALL_DATA = "./data/interim/data_all.csv"
PATH_ALL_DATA_PROCESSED = "./data/interim/data_all_processed.csv"
PATH_ZIPCODES_INFO = "./data/external/d_zipcodes_info.json"
PATH_ALL_DATA_ADDED_FEATURES = "./data/interim/data_added_features.csv"
PATH_DATA_ALL_FEATURES = "./data/interim/data_with_external_features.csv"
PATH_DATA_ALL_CAT = "./data/interim/data_processed_categorical.csv"
PATH_DATA_ALL_CORR = "./data/interim/data_processed_correlated_X.csv"
PATH_DATA_Y = "./data/interim/y.csv"
PATH_DATA_ALL_CORR_TO_PREDICT = "./data/interim/data_processed_correlated_X_to_predict.csv"
PATH_DATA_FINAL_X = "./data/interim/data_final_X.csv"
PATH_DATA_FINAL_X_TO_PREDICT = "./data/processed/data_final_X_to_predict.csv"
PATH_DATA_FINAL_X_TRAIN = "./data/processed/X_train.csv"
PATH_DATA_FINAL_X_VAL = "./data/processed/X_val.csv"
PATH_DATA_FINAL_Y_TRAIN = "./data/processed/y_train.csv"
PATH_DATA_FINAL_Y_VAL = "./data/processed/y_val.csv"
PATH_BEST_PARAMS_LASSO = "./data/interim/lasso_best_params.json"
PATH_BEST_PARAMS_KNN = "./data/interim/knn_best_params.json"
PATH_MODEL = "./models/model"
PATH_SCORE = "./reports/score.csv"
PATH_SOLUTION = "./reports/solution.csv"

if __name__ == "__main__":
    src.read_merge_data(INPUT_FILE_X_TRAIN, INPUT_FILE_X_TEST, INPUT_FILE_Y_TRAIN, PATH_ALL_DATA)
    src.preprocess_data(PATH_ALL_DATA, PATH_ALL_DATA_PROCESSED)
    src.get_external_data(SCROLL_PAUSE_TIME, PATH_ALL_DATA_PROCESSED, PATH_ZIPCODES_INFO)
    src.modify_features(PATH_ALL_DATA_PROCESSED, PATH_ALL_DATA_ADDED_FEATURES)
    src.add_external_features(PATH_ALL_DATA_ADDED_FEATURES, PATH_ZIPCODES_INFO, PATH_DATA_ALL_FEATURES)
    src.process_categorical_features(PATH_DATA_ALL_FEATURES, PATH_DATA_ALL_CAT)
    src.process_correlated_features(CORR_THRES, PATH_DATA_ALL_CAT, PATH_DATA_ALL_CORR, PATH_DATA_Y, PATH_DATA_ALL_CORR_TO_PREDICT)
    src.scale_features(PATH_DATA_ALL_CORR, PATH_DATA_ALL_CORR_TO_PREDICT, PATH_DATA_FINAL_X, PATH_DATA_FINAL_X_TO_PREDICT)
    src.prepare_dataset(RANDOM_STATE, PATH_DATA_FINAL_X, PATH_DATA_Y, PATH_DATA_FINAL_X_TRAIN, PATH_DATA_FINAL_X_VAL, PATH_DATA_FINAL_Y_TRAIN, PATH_DATA_FINAL_Y_VAL)
    src.find_best_model_parameters(RANDOM_STATE, PATH_DATA_FINAL_X_TRAIN, PATH_DATA_FINAL_Y_TRAIN, PATH_BEST_PARAMS_LASSO, PATH_BEST_PARAMS_KNN)
    src.train_model(PATH_DATA_FINAL_X_TRAIN, PATH_DATA_FINAL_Y_TRAIN, PATH_BEST_PARAMS_LASSO, PATH_BEST_PARAMS_KNN, PATH_MODEL)
    src.evaluate_model(PATH_DATA_FINAL_X_TO_PREDICT, PATH_DATA_FINAL_X_VAL, PATH_DATA_FINAL_Y_VAL, PATH_MODEL, PATH_SCORE, PATH_SOLUTION)
