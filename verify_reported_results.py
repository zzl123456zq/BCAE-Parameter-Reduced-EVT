"""Check released result files and saved notebook outputs against the paper."""

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# Values in Tables IV and V: (Acc, F1, FDR, FAR) for Groups A--D.
PIPELINE_EXPECTED = {
    "VAE": [
        (0.8904, 0.8837, 0.8680, 0.0889),
        (0.8984, 0.9155, 0.8970, 0.0993),
        (0.8821, 0.8917, 0.8737, 0.1074),
        (0.8887, 0.9162, 0.8831, 0.0989),
    ],
    "DAGMM": [
        (0.9151, 0.9087, 0.8806, 0.0530),
        (0.9485, 0.9579, 0.9537, 0.0596),
        (0.9030, 0.9082, 0.8643, 0.0488),
        (0.9437, 0.9584, 0.9409, 0.0503),
    ],
    "GANomaly": [
        (0.9336, 0.9317, 0.9433, 0.0754),
        (0.9188, 0.9325, 0.9134, 0.0727),
        (0.8838, 0.8907, 0.8525, 0.0771),
        (0.8986, 0.9232, 0.8845, 0.0703),
    ],
    "USAD": [
        (0.9072, 0.8991, 0.8613, 0.0505),
        (0.9465, 0.9560, 0.9471, 0.0544),
        (0.9131, 0.9187, 0.8849, 0.0518),
        (0.9250, 0.9439, 0.9153, 0.0535),
    ],
    "OpenMax": [
        (0.7342, 0.7775, 0.9678, 0.4816),
        (0.7513, 0.8136, 0.8841, 0.4597),
        (0.5389, 0.5915, 0.6012, 0.5388),
        (0.6181, 0.7122, 0.6859, 0.5319),
    ],
    "OSSC+EVT": [
        (0.9275, 0.9253, 0.9345, 0.0789),
        (0.9248, 0.9382, 0.9301, 0.0836),
        (0.9213, 0.9296, 0.9352, 0.0960),
        (0.8379, 0.8746, 0.8204, 0.1232),
    ],
    "D3R": [
        (0.8433, 0.8376, 0.8417, 0.1552),
        (0.7986, 0.8163, 0.7293, 0.0914),
        (0.8401, 0.8533, 0.8374, 0.1565),
        (0.8581, 0.8936, 0.8653, 0.1578),
    ],
    "GEL": [
        (0.8562, 0.8345, 0.7548, 0.0501),
        (0.7997, 0.8194, 0.7400, 0.1054),
        (0.7774, 0.8014, 0.7321, 0.1506),
        (0.7362, 0.7730, 0.6519, 0.0773),
    ],
    "TCDL": [
        (0.9228, 0.9173, 0.8921, 0.0488),
        (0.9048, 0.9187, 0.8759, 0.0492),
        (0.9474, 0.9525, 0.9512, 0.0575),
        (0.9696, 0.9781, 0.9883, 0.0719),
    ],
    "Proposed": [
        (0.9741, 0.9734, 0.9863, 0.0372),
        (0.9622, 0.9687, 0.9540, 0.0248),
        (0.9534, 0.9579, 0.9543, 0.0476),
        (0.9571, 0.9689, 0.9702, 0.0719),
    ],
}

# Values in the TEP comparison table: (Acc, F1, FDR, FAR).
TEP_EXPECTED = {
    "VAE": (0.8025, 0.7866, 0.9098, 0.2690),
    "DAGMM": (0.9070, 0.8891, 0.9320, 0.1097),
    "GANomaly": (0.8152, 0.7768, 0.8038, 0.1772),
    "USAD": (0.8665, 0.8252, 0.7880, 0.0812),
    "OpenMax": (0.9032, 0.8891, 0.9699, 0.1414),
    "OSSC+EVT": (0.7487, 0.7376, 0.8829, 0.3407),
    "D3R": (0.9186, 0.8991, 0.9071, 0.0737),
    "GEL": (0.7487, 0.6727, 0.6456, 0.1825),
    "TCDL": (0.6741, 0.7102, 0.9984, 0.5422),
    "Proposed": (0.9230, 0.9083, 0.9526, 0.0967),
}

NP_EXPECTED = {2: 0.9041, 3: 0.9083, 4: 0.9083, 5: 0.8571, 6: 0.8571}

PIPELINE_AUC_EXPECTED = {
    "A": 0.9975,
    "B": 0.9966,
    "C": 0.9925,
    "D": 0.9927,
}

COMPLEXITY_EXPECTED = {
    "DAGMM": (0.1342, 0.18, 0.0081, 21.98),
    "USAD": (0.0853, 0.28, 0.0036, 19.16),
    "OSSC": (0.0721, 0.22, 0.0048, 18.86),
    "D3R": (0.4099, 0.46, 0.0275, 51.12),
    "GEL": (0.2568, 0.36, 0.0186, 34.42),
    "Proposed": (0.0207, 0.42, 0.0100, 23.36),
}


def read_rows(filename):
    with (RESULTS / filename).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def assert_values(actual, expected, context):
    if len(actual) != len(expected) or any(
        abs(value - reference) > 5e-7
        for value, reference in zip(actual, expected)
    ):
        raise AssertionError(f"{context}: expected {expected}, got {actual}")


def check_pipeline_results():
    rows = read_rows("reported_metrics.csv")
    indexed = {(row["method"], row["group"]): row for row in rows}
    groups = "ABCD"
    for method, expected_rows in PIPELINE_EXPECTED.items():
        for group, expected in zip(groups, expected_rows):
            row = indexed[(method, group)]
            actual = tuple(float(row[key]) for key in ("accuracy", "f1", "fdr", "far"))
            assert_values(actual, expected, f"Pipeline {method}, Group {group}")


def check_tep_results():
    rows = {row["method"]: row for row in read_rows("tep_reported_metrics.csv")}
    for method, expected in TEP_EXPECTED.items():
        row = rows[method]
        actual = tuple(float(row[key]) for key in ("accuracy", "f1", "fdr", "far"))
        assert_values(actual, expected, f"TEP {method}")


def check_np_results():
    rows = {int(row["np"]): row for row in read_rows("tep_np_sensitivity.csv")}
    for np_value, expected_f1 in NP_EXPECTED.items():
        assert_values((float(rows[np_value]["f1"]),), (expected_f1,), f"TEP N_p={np_value}")


def check_complexity_results():
    rows = {row["method"]: row for row in read_rows("computational_efficiency.csv")}
    keys = (
        "parameters_m",
        "training_time_s_per_epoch",
        "inference_latency_ms_per_sample",
        "peak_gpu_memory_mb",
    )
    for method, expected in COMPLEXITY_EXPECTED.items():
        actual = tuple(float(rows[method][key]) for key in keys)
        assert_values(actual, expected, f"Complexity {method}")


def check_notebook_outputs():
    notebook = json.loads((ROOT / "main.ipynb").read_text(encoding="utf-8"))
    output_text = "\n".join(
        "".join(output.get("text", []))
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
    )
    for group, (acc, f1, fdr, far) in zip("ABCD", PIPELINE_EXPECTED["Proposed"]):
        auc = PIPELINE_AUC_EXPECTED[group]
        pattern = (
            rf"Acc:{acc:.4f}, P:[0-9.]+, R:[0-9.]+, F1:{f1:.4f}, "
            rf"FDR:{fdr:.4f}, FAR:{far:.4f}, AUC:{auc:.4f}"
        )
        if re.search(pattern, output_text) is None:
            raise AssertionError(f"Saved notebook output is missing Proposed Group {group}.")

    for token in (
        "t_best=0.05",
        "gamma=5.588",
        "sigma=0.008018",
        "F1:0.9083",
        "Paper-reported reference: Params=0.0207 M, Train=0.42 s/epoch, "
        "Latency=0.0100 ms/sample, GPU memory=23.36 MB",
    ):
        if token not in output_text:
            raise AssertionError(f"Saved TEP output is missing {token}.")

    for np_value, expected_f1 in NP_EXPECTED.items():
        pattern = rf"N_p {np_value}.*?F1:{expected_f1:.4f}"
        if re.search(pattern, output_text, flags=re.DOTALL) is None:
            raise AssertionError(f"Saved notebook output is missing N_p={np_value}.")


def main():
    check_pipeline_results()
    check_tep_results()
    check_np_results()
    check_complexity_results()
    check_notebook_outputs()
    print("Reported-result consistency check passed.")


if __name__ == "__main__":
    main()
