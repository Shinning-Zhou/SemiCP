import csv
import os
import torch
import numpy as np


filenames = ["Seed", "Dataset", "Shift_dataset", "Model", "Weight", "Top1", "Top5",
             "setting", "score", "random", "method", "alpha", "class_num", "group_num",
             "predicted_coverage", "predicted_size", "coverage_gap",
             "labeled_num", "unlabeled_num", "test_num",
             ]

def write_result_semicp(file, seed, dataset_name, shift_dataset_name, model, weight, top1, top5,
                        setting, score, random, method, alpha, class_num, group_num,
                        coverage_rate, average_size, CovGap,
                        labeled_num, unlabeled_num, test_num
                        ):
    data = {
        "Seed": seed,
        "Dataset": dataset_name,
        "Shift_dataset": shift_dataset_name,
        "Model": model,
        "Weight": weight,
        "Top1": round(top1, 2) if top1 != None else None,
        "Top5": round(top5, 2) if top5 != None else None,
        "setting": setting,
        "score": score,
        "random": random,
        "method": method,
        "alpha": alpha,
        "class_num": class_num,
        "group_num": group_num,
        "predicted_coverage": round(coverage_rate, 4),
        "predicted_size": round(average_size, 4),
        "coverage_gap": round(CovGap, 4),
        "labeled_num": labeled_num,
        "unlabeled_num": unlabeled_num,
        "test_num": test_num,
    }

    with open(file, mode='a', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        csv_writer.writerow(data)


def write_per_class_csv(
    csv_path: str,
    seed: int,
    dataset_name: str,
    model_name: str,
    weight: str,
    setting: str,
    score_name: str,
    alpha: float,
    class_num: int,
    labeled_num: int,
    unlabeled_num: int,
    test_num: int,
    results: dict
):
    """
    保存每个 class 一行：
      class_id, class_count, class_acc,
      cov_standard, cov_oracle, cov_semicp,
      size_standard, size_oracle, size_semicp
    并附带整体信息（写在每行重复，方便 groupby）
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    header = [
        "seed", "dataset", "model", "weight", "setting", "score", "alpha",
        "labeled_num", "unlabeled_num", "test_num",
        "class_id", "class_count", "class_acc",
        "cov_standard", "cov_oracle", "cov_semicp",
        "size_standard", "size_oracle", "size_semicp",
        "overall_acc",
        "overall_cov_standard", "overall_cov_oracle", "overall_cov_semicp",
        "overall_size_standard", "overall_size_oracle", "overall_size_semicp",
    ]

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()

        acc_pc = results["_test_acc"]["per_class_acc"].detach().cpu().numpy()
        acc_cnt = results["_test_acc"]["per_class_cnt"].detach().cpu().numpy()
        overall_acc = results["_test_acc"]["overall_acc"]

        cov_std = results["standard"]["per_class_coverage"].detach().cpu().numpy()
        cov_orc = results["oracle"]["per_class_coverage"].detach().cpu().numpy()
        cov_semi = results["semicp"]["per_class_coverage"].detach().cpu().numpy()

        size_std = results["standard"]["per_class_avg_size"].detach().cpu().numpy()
        size_orc = results["oracle"]["per_class_avg_size"].detach().cpu().numpy()
        size_semi = results["semicp"]["per_class_avg_size"].detach().cpu().numpy()

        overall_cov_std = results["standard"]["overall_coverage"]
        overall_cov_orc = results["oracle"]["overall_coverage"]
        overall_cov_semi = results["semicp"]["overall_coverage"]

        overall_size_std = results["standard"]["overall_avg_size"]
        overall_size_orc = results["oracle"]["overall_avg_size"]
        overall_size_semi = results["semicp"]["overall_avg_size"]

        for c in range(class_num):
            row = {
                "seed": seed,
                "dataset": dataset_name,
                "model": model_name,
                "weight": weight,
                "setting": setting,
                "score": score_name,
                "alpha": alpha,
                "labeled_num": labeled_num,
                "unlabeled_num": unlabeled_num,
                "test_num": test_num,

                "class_id": c,
                "class_count": int(acc_cnt[c]),
                "class_acc": float(acc_pc[c]) if not np.isnan(acc_pc[c]) else "",
                "cov_standard": float(cov_std[c]) if not np.isnan(cov_std[c]) else "",
                "cov_oracle": float(cov_orc[c]) if not np.isnan(cov_orc[c]) else "",
                "cov_semicp": float(cov_semi[c]) if not np.isnan(cov_semi[c]) else "",
                "size_standard": float(size_std[c]) if not np.isnan(size_std[c]) else "",
                "size_oracle": float(size_orc[c]) if not np.isnan(size_orc[c]) else "",
                "size_semicp": float(size_semi[c]) if not np.isnan(size_semi[c]) else "",

                "overall_acc": overall_acc,
                "overall_cov_standard": overall_cov_std,
                "overall_cov_oracle": overall_cov_orc,
                "overall_cov_semicp": overall_cov_semi,
                "overall_size_standard": overall_size_std,
                "overall_size_oracle": overall_size_orc,
                "overall_size_semicp": overall_size_semi,
            }
            w.writerow(row)
