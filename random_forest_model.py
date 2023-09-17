import pandas as pd
import numpy as np
import copy
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

logging.getLogger().setLevel(logging.INFO)


class DataProcessor:
    def __init__(self, raw_data):
        self.raw_data = raw_data.copy()

    def exponential_smoothing(self, alpha=0.2):
        smoothed_series = self.raw_data["Adj Close"].ewm(alpha=alpha).mean()

        return smoothed_series

    def OnBalanceVolume(self):
        self.raw_data["Smoothed_adj_close"] = self.exponential_smoothing()
        self.raw_data["Daily_Price_Change"] = self.raw_data["Smoothed_adj_close"].diff()
        self.raw_data["Direction"] = self.raw_data["Daily_Price_Change"].apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )
        OBV = (self.raw_data["Direction"] * self.raw_data["Volume"]).cumsum()

        return OBV

    def StochasticOscillator(self, K_high=14, K_low=3):
        self.raw_data["L14"] = self.raw_data["Low"].rolling(window=K_high).min()
        self.raw_data["H14"] = self.raw_data["High"].rolling(window=K_high).max()
        self.raw_data["%K"] = 100 * (
            (self.raw_data["Smoothed_adj_close"] - self.raw_data["L14"])
            / (self.raw_data["H14"] - self.raw_data["L14"])
        )
        self.raw_data["%D"] = self.raw_data["%K"].rolling(window=K_low).mean()

        return self.raw_data[["%K", "%D"]]

    def MACD(self, short_period=12, long_period=26, signal_period=9):
        MA_Fast = (
            self.raw_data["Smoothed_adj_close"]
            .ewm(span=short_period, min_periods=long_period)
            .mean()
        )
        MA_Slow = (
            self.raw_data["Smoothed_adj_close"]
            .ewm(span=long_period, min_periods=long_period)
            .mean()
        )
        MACD = MA_Fast - MA_Slow
        Signal = MACD.ewm(span=signal_period, adjust=False).mean()
        MACD = pd.concat([MACD.rename("MACD"), Signal.rename("Signal")], axis=1)

        return MACD

    def categorical_price(self, forecast_period):
        price = np.sign(
            np.log(
                self.raw_data["Smoothed_adj_close"].shift(-forecast_period)
                / self.raw_data["Smoothed_adj_close"]
            )
        )

        return price

    def prepare_data(self, forecast_period: range = range(1, 31)):
        self.raw_data = copy.deepcopy(self.raw_data)
        self.raw_data["Smoothed_adj_close"] = self.exponential_smoothing()
        self.raw_data["OBV"] = self.OnBalanceVolume()
        self.raw_data[["%K", "%D"]] = self.StochasticOscillator()
        self.raw_data[["MACD", "Signal"]] = self.MACD()
        for day in forecast_period:
            self.raw_data[f"Categorical_p_{day}"] = self.categorical_price(day)

        return self.raw_data

    def RFClassifier(
        self,
        independent_variables: list,
        dependent_variable: str,
        horizon: int,
        ticker: str,
    ):
        columns_to_check = ["Adj Close", "Volume", "Low", "High"]
        counts = [self.raw_data[column].isna().sum() for column in columns_to_check]
        null_close, null_volume, null_low, null_high = counts
        logging.info(f"Initiating model for {ticker} with prediction horizon {horizon}")
        [
            logging.warning(
                f"There are {var_value} {var_name} values in the pre_processed dataset"
            )
            for var_name, var_value in [
                ("null_close", null_close),
                ("null_volume", null_volume),
                ("null_low", null_low),
                ("null_high", null_high),
            ]
        ]
        self.raw_data = self.raw_data.dropna()
        X = self.raw_data[independent_variables]
        y = self.raw_data[dependent_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        """Accuracy metrics"""
        predicted_probs = model.predict_proba(X_test)[:, 1]
        AUC = metrics.roc_auc_score(y_test, predicted_probs)
        Accuracy = metrics.accuracy_score(y_test, prediction)
        Precision = metrics.average_precision_score(y_test, prediction)
        Recall = metrics.recall_score(y_test, prediction)
        F1_score = metrics.f1_score(y_test, prediction)
        Confusion_matrix = metrics.confusion_matrix(
            y_test, prediction, labels=model.classes_
        )
        cm_display = metrics.ConfusionMatrixDisplay(
            confusion_matrix=Confusion_matrix, display_labels=model.classes_
        )
        feature_names = X_train.columns
        feature_importances = model.feature_importances_
        feature_importance_dict = dict(zip(feature_names, feature_importances))
        logging.info(
            f"Completed prediction for {ticker} with horizon {horizon}. Writing results to dict."
        )
        result = {
            "Metrics": {
                "Accuracy": Accuracy,
                "Precision": Precision,
                "Recall": Recall,
                "F1_score": F1_score,
                "AUC": AUC,
            },
            "ConfusionMatrix": Confusion_matrix,
            "FeatureImportance": feature_importance_dict
        }

        return result