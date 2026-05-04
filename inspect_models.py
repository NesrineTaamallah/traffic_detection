"""
inspect_models.py
=================
Lance ce script UNE FOIS pour voir exactement quelles features
attendent tes modèles pkl.

Usage :
    python inspect_models.py
"""

import joblib
from pathlib import Path

MODELS_DIR = Path(r"C:\Users\nesri\OneDrive\Desktop\network_traffic_detection\models")

files = {
    "binary":     "best_binary_model.pkl",
    "multiclass": "xgb_hierarchical_multiclass.pkl",
    "scaler":     "scaler_hierarchical.pkl",
    "label_enc":  "label_encoder_hierarchical.pkl",
}

for name, fname in files.items():
    path = MODELS_DIR / fname
    print(f"\n{'='*60}")
    print(f"  {name.upper()} → {fname}")
    print(f"{'='*60}")
    try:
        obj = joblib.load(path)
        print(f"  Type : {type(obj).__name__}")

        # Features attendues
        for attr in ["feature_names_in_", "feature_names_", "feature_name_"]:
            if hasattr(obj, attr):
                feats = list(getattr(obj, attr))
                print(f"  {attr} ({len(feats)} features) :")
                for i, f in enumerate(feats):
                    print(f"    [{i:3d}] {f}")
                break

        # Scaler
        if hasattr(obj, "feature_names_in_"):
            pass  # déjà affiché
        elif hasattr(obj, "n_features_in_"):
            print(f"  n_features_in_ : {obj.n_features_in_}")

        # LabelEncoder
        if hasattr(obj, "classes_"):
            print(f"  classes_ : {list(obj.classes_)}")

        # Pipeline
        if hasattr(obj, "steps"):
            print(f"  Pipeline steps : {[s[0] for s in obj.steps]}")
            for step_name, step_obj in obj.steps:
                for attr in ["feature_names_in_", "feature_names_"]:
                    if hasattr(step_obj, attr):
                        feats = list(getattr(step_obj, attr))
                        print(f"    [{step_name}] {attr} ({len(feats)}) :")
                        for i, f in enumerate(feats):
                            print(f"      [{i:3d}] {f}")

    except Exception as e:
        print(f"  ERREUR : {e}")

print("\n\nCOLLECTE des features uniques attendues par binary + scaler :")
try:
    binary = joblib.load(MODELS_DIR / "best_binary_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler_hierarchical.pkl")

    feat_sets = []
    for obj in [binary, scaler]:
        for attr in ["feature_names_in_", "feature_names_"]:
            if hasattr(obj, attr):
                feat_sets.append(set(getattr(obj, attr)))
                break
        if hasattr(obj, "steps"):
            for _, step in obj.steps:
                for attr in ["feature_names_in_", "feature_names_"]:
                    if hasattr(step, attr):
                        feat_sets.append(set(getattr(step, attr)))

    if feat_sets:
        all_feats = sorted(feat_sets[0])
        print(f"\nFeatures requises ({len(all_feats)}) :")
        print(all_feats)
except Exception as e:
    print(f"ERREUR : {e}")