import pandas as pd
from random_forest_model import DataProcessor
import yfinance as yf


def aggregate_metrics(model_outcome: list):
    metrics_list = []

    for idx, result in enumerate(model_outcome, start=1):
        metrics = result["Metrics"]
        metrics["Horizon"] = idx
        metrics_list.append(metrics)

    df = pd.DataFrame(metrics_list)
    df.set_index("Horizon", inplace=True)

    return df


def aggreate_confusion_metrics(model_outcome: list):
    conf_metrics_list = []
    for idx, result in enumerate(model_outcome, start=1):
        confusion_matrix = result["ConfusionMatrix"]
        sum_down = sum(confusion_matrix[0])
        sum_up = sum(confusion_matrix[1])
        metrics_dict = {
            "correct_down": confusion_matrix[0][0] / sum_down,
            "incorrect_down": confusion_matrix[0][1] / sum_down,
            "correct_up": confusion_matrix[1][1] / sum_up,
            "incorrect_up": confusion_matrix[1][0] / sum_up,
            "Horizon": idx,
        }
        conf_metrics_list.append(metrics_dict)
    df = pd.DataFrame(conf_metrics_list)
    df.set_index("Horizon", inplace=True)

    return df


def aggregate_feature_importance(model_outcome: list):
    feature_importance_list = []

    for idx, result in enumerate(model_outcome, start=1):
        feature_importance = result.get("FeatureImportance")
        if feature_importance is not None:
            feature_importance["Horizon"] = idx
            feature_importance_list.append(feature_importance)

    df = pd.DataFrame(feature_importance_list)
    df.set_index("Horizon", inplace=True)

    return df


def write_metrics(forecast: list, path_to_write_data: str, ticker: str):
    accuracy_metrics = aggregate_metrics(forecast)
    conf_metrics = aggreate_confusion_metrics(forecast)
    feature_importance = aggregate_feature_importance(forecast)
    metrics = pd.concat([accuracy_metrics, conf_metrics, feature_importance], axis=1)

    full_path = path_to_write_data + "/" + ticker + ".csv"

    metrics.to_csv(full_path)


tickers = ["BX", "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "TSLA"]
start = "2000-06-02"
end = "2016-04-26"
forecasting_range = range(1, 31)
data = {}
for company in tickers:
    data[company] = []
    company_data = yf.download(company, start=start, end=end)[
        ["Adj Close", "Volume", "High", "Low", "Close"]
    ]
    data_processor = DataProcessor(company_data)
    pre_processed_aapl_data = data_processor.prepare_data(
        forecast_period=forecasting_range
    )
    for horizon in forecasting_range:
        result = data_processor.RFClassifier(
            dependent_variable=f"Categorical_p_{horizon}",
            independent_variables=["OBV", "%K", "%D", "MACD", "Signal"],
            horizon=horizon,
            ticker=company,
        )
        data[company].append(result)
    write_metrics(
        forecast=data[company],
        path_to_write_data="yourpath",
        ticker=company,
    )
