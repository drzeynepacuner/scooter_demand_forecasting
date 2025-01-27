from src.model_selection.time_series_cv import evaluate_model_candidates

def forward_feature_selection(data, all_features, model_candidates, n_splits=3):
    selected = []
    remaining = set(all_features)

    best_rmse_global = None
    best_mae_global = None
    best_mape_global = None
    best_model_global = None
    best_params_global = None

    while remaining:
        feature_to_add = None
        best_local_rmse = float('inf')
        best_local_mae = None
        best_local_mape = None
        best_local_model = None
        best_local_params = None

        for f in remaining:
            trial_feats = selected + [f]
            r, m, mp, model_, params_ = evaluate_model_candidates(data, trial_feats, model_candidates, n_splits)
            if r < best_local_rmse:
                best_local_rmse = r
                best_local_mae = m
                best_local_mape = mp
                best_local_model = model_
                best_local_params = params_
                feature_to_add = f

        if (best_rmse_global is None) or (best_local_rmse < best_rmse_global):
            selected.append(feature_to_add)
            remaining.remove(feature_to_add)
            best_rmse_global = best_local_rmse
            best_mae_global = best_local_mae
            best_mape_global = best_local_mape
            best_model_global = best_local_model
            best_params_global = best_local_params

            print(f"Added feature: {feature_to_add}")
            print(f"  => CV RMSE: {best_rmse_global:.3f}, MAE: {best_mae_global:.3f}, MAPE: {best_mape_global:.2f}%")
            print(f"  => Best model: {best_model_global[0]}, params: {best_params_global}")
        else:
            print("No improvement, stopping.")
            break

    return (selected, best_rmse_global, best_mae_global, best_mape_global, best_model_global, best_params_global)
