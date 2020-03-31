import numpy as np
import csv
from typing import Dict, Any
from collections import Counter


def load_data(path):
    """
    read a csv in from a dict
    """
    result = {}
    reader = csv.DictReader(open(path))
    for row in reader:
        for k, v in row.items():
            result.setdefault(k, []).append(v)
    return result


def _test_value(x: Any) -> bool:
    """
    test if a column is continuous or not
    """
    try:
        float(x)
        return True
    except ValueError:
        return False


def cleanup(d: Dict[str, str]) -> Dict[str, np.ndarray]:
    """
    turn default continuous columns into floats, replace non alphanumeric with np.nan
    """
    res = {}
    for k, v in d.items():
        if _test_value(v[0]):
            res[k] = np.array([float(x) if _test_value(x) else np.nan for x in v])
        else:
            res[k] = np.array(v)
    return res


def convert_dollars_percs(x: np.ndarray) -> np.ndarray:
    """replace dollar signs and percentages"""
    out = [x[i].replace("$", "").replace("%", "") for i in range(x.shape[0])]
    out = np.array([float(z) if _test_value(z) else np.nan for z in out])
    return out


def impute_cats(
    d: Dict[str, np.ndarray], c: Dict[str, Counter]
) -> Dict[str, np.ndarray]:
    """mode impute cats"""
    for k in c.keys():
        d[k][np.isnan(d[k])] = c[k].most_common(1)[0][0]
    return d


def load_dataset(path: str) -> Dict[str, np.ndarray]:
    data = load_data(path)
    data = cleanup(data)
    for k in ["x32", "x37"]:
        data[k] = convert_dollars_percs(data[k])
    # cats are continent, month, day
    cats = [k for k, v in data.items() if not _test_value(v[0])]
    cont_dict = {"": np.nan}
    for idx, k in enumerate(list(set(data["x24"]))[1:]):
        cont_dict[k] = idx
    data["x24"] = np.array([cont_dict[v] for v in data["x24"]])
    day_dict = dict(
        zip(["monday", "tuesday", "wednesday", "thurday", "friday"], range(1, 6))
    )
    day_dict[""] = np.nan
    data["x30"] = np.array([day_dict[v] for v in data["x30"]])
    months = [
        "January",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "July",
        "Aug",
        "sept.",
        "Oct",
        "Nov",
        "Dev",
    ]
    month_dict = dict(zip(months, range(1, 13)))
    month_dict[""] = np.nan
    data["x29"] = np.array([month_dict[v] for v in data["x29"]])
    cat_dict = {k: Counter(data[k]) for k in cats}
    data = impute_cats(data, cat_dict)
    return data


x = load_dataset("../data/final_project.csv")
print(x)
