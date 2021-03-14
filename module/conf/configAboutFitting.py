# configは辞書化しておく。
def outputConfig(train, test, target, folds):
    confFitting = {}

    # Fitするときに"y"として使う列の列名配列
    confFitting["target_cols"] = target.columns.values.tolist()
    # Fitするときに"X"として使う列の列名配列
    # kfold, id等はここで削除。
    feature_cols = [c for c in folds.columns if c not in confFitting["target_cols"]]
    confFitting["feature_cols"] = [c for c in feature_cols if c not in ['kfold', 'sig_id']]
    # 特徴量、ターゲットのサイズ
    confFitting["num_features"] = len(confFitting["feature_cols"])
    confFitting["num_targets"] = len(confFitting["target_cols"])

    return confFitting