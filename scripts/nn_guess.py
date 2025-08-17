import os
import numpy as np
import pandas as pd
import joblib
import keras  # ensure registry available

from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

#this script uses the trained model to make suggestions to the user for the best behavior change from a set of possible changes

# ---------------------------
# custom layers to compute mean on non-padded data
# ---------------------------
@keras.saving.register_keras_serializable(package="lifeadapt")
class TimeMaskFromLength(tf.keras.layers.Layer):
    def __init__(self, time_steps, **kwargs):
        super().__init__(**kwargs)
        self.time_steps = int(time_steps)
    def call(self, lengths):
        m = tf.sequence_mask(lengths, maxlen=self.time_steps)
        return tf.cast(tf.expand_dims(m, -1), tf.float32)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"time_steps": self.time_steps})
        return cfg

@keras.saving.register_keras_serializable(package="lifeadapt")
class MaskedMean(tf.keras.layers.Layer):
    def call(self, inputs):
        x, m = inputs
        x = x * m
        s = tf.reduce_sum(x, axis=1)
        c = tf.reduce_sum(m, axis=1) + 1e-8
        return s / c
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])

# ----------------------------
# Config
# ----------------------------
COMBINED_CSV = "combined_output.csv"
MODEL_PATH = "model.keras"
STRUCT_SCALER_PATH = "structscaler.pkl"
DEM_SCALER_PATH   = "demscaler.pkl"
Y_SCALER_PATH     = "yscaler.pkl"
GENDER_OHE_PATH   = "gender_ohe.pkl"

R_MAP = {
    "active_num":    "r1",
    "engaged_num":   "r2",
    "fatigue_num":   "r3",
    "anxious_num":   "r4",
    "depressed_num": "r5",
    "irritable_num": "r6",
    "social_num":    "r7",
}
STRUCT_COLS = ["acc", "out", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "score"]

TUNABLE_FIELDS = ["active_num", "engaged_num", "anxious_num", "social_num"]
STEP = 1.0  # +/- 1 Likert point

# ----------------------------
# Helpers
# ----------------------------

#create day column that counts days rather than date-time format
def ensure_day(df: pd.DataFrame) -> pd.DataFrame:
    if "day" in df.columns:
        return df
    dfx = df.copy()
    if "date" in dfx.columns:
        dfx["date"] = pd.to_datetime(dfx["date"], errors="coerce")
        dfx = dfx.sort_values(["pid", "date"])
    else:
        dfx = dfx.sort_values(["pid"])
    dfx["day"] = dfx.groupby("pid").cumcount()
    return dfx

#pads or trims data to fit specified time series length
def pad_or_trim_to_T(seq_2d: np.ndarray, T_target: int):
    seq_2d = np.asarray(seq_2d, dtype=np.float32)
    Ti, F = seq_2d.shape
    X = np.zeros((1, T_target, F), dtype=np.float32)
    T_copy = min(Ti, T_target)
    if T_copy > 0:
        X[0, :T_copy, :] = seq_2d[:T_copy, :]
    L = np.array([T_copy], dtype=np.int32)
    return X, L

#applies scaler to the non-padded data
def fit_like_transform_struct(X: np.ndarray, lengths: np.ndarray, scaler):
    B, T, F = X.shape
    mask = (np.arange(T)[None, :] < lengths[:, None])
    X_flat = X.reshape(-1, F)
    mask_flat = mask.reshape(-1)
    X_flat[mask_flat] = scaler.transform(X_flat[mask_flat])
    return X_flat.reshape(B, T, F)

#input data to model, convert output out of scaler and return the mean
def predict_mean_target(model, Xs_scaled, Xe, Xd_scaled, y_scaler, lengths):
    lengths = np.asarray(lengths, dtype=np.int32).reshape(-1)
    y_pred_split = model.predict([Xs_scaled, Xe, Xd_scaled, lengths], verbose=0)
    y_pred_scaled = np.hstack([p.reshape(-1, 1) for p in y_pred_split])
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    return float(np.mean(y_pred, axis=1)[0]), y_pred[0]

#clean data formatting and clip EMA values
def build_s_frame_from_combined(df_pid: pd.DataFrame) -> pd.DataFrame:
    s = pd.DataFrame(index=df_pid.index)
    s["acc"]  = pd.to_numeric(df_pid["acc"],  errors="coerce").astype(np.float32)
    s["out"]  = pd.to_numeric(df_pid["out"],  errors="coerce").astype(np.float32)
    for csv_col, rname in R_MAP.items():
        s[rname] = pd.to_numeric(df_pid[csv_col], errors="coerce").astype(np.float32)
    s["score"] = pd.to_numeric(df_pid["score"], errors="coerce").astype(np.float32)

    s = s.ffill().bfill()
    for r in ["r1","r2","r3","r4","r5","r6","r7"]:
        s[r] = s[r].clip(1.0, 5.0)
    s = s.fillna(0.0)
    return s[STRUCT_COLS].copy()

#formats data to be ready to input into model
def build_arrays_from_combined_fixedT(df_pid: pd.DataFrame, model):
    df_pid = df_pid.sort_values("day").reset_index(drop=True)

    T_model = int(model.inputs[0].shape[1])      # e.g., 41
    embed_dim = int(model.inputs[1].shape[-1])   # e.g., 1536

    s_slice = build_s_frame_from_combined(df_pid)        # (Ti, 10)
    Xs_seq = s_slice.to_numpy(np.float32)

    Xe_seq = np.zeros((len(df_pid), embed_dim), dtype=np.float32)  # no embeddings here

    age = float(pd.to_numeric(df_pid["age"].iloc[0], errors="coerce"))
    edu = float(pd.to_numeric(df_pid["education"].iloc[0], errors="coerce"))
    gender = str(df_pid["gender"].iloc[0])
    if np.isnan(age) or np.isnan(edu):
        raise ValueError("Missing age/education for this pid; cannot scale demographics.")
    Xd_raw = np.array([[age, edu]], dtype=np.float32)

    Xs, lengths = pad_or_trim_to_T(Xs_seq, T_model)
    Xe, _       = pad_or_trim_to_T(Xe_seq, T_model)

    if np.isnan(Xs).any():
        raise ValueError("NaNs remain in structured inputs after imputation.")
    return Xs, Xe, Xd_raw, lengths, gender, s_slice, T_model

# Main function, tries all the possible changes and chooses the one that most improves TMB scores
def select_best_single_change_for_pid_from_combined(df_all: pd.DataFrame, pid: str):
    model = load_model(MODEL_PATH, compile=False)  # custom layers are registered above
    struct_scaler = joblib.load(STRUCT_SCALER_PATH)
    dem_scaler    = joblib.load(DEM_SCALER_PATH)
    y_scaler      = joblib.load(Y_SCALER_PATH)

    # Use the exact OHE saved during training if available (prevents width mismatch)
    if os.path.exists(GENDER_OHE_PATH):
        ohe = joblib.load(GENDER_OHE_PATH)
    else:
        genders_all = df_all[["gender"]].astype(object).to_numpy()
        try:
            ohe = OneHotEncoder(sparse_output=False, drop="if_binary").fit(genders_all)
        except TypeError:
            ohe = OneHotEncoder(sparse=False, drop="if_binary").fit(genders_all)

    df_pid = df_all[df_all["pid"] == pid].copy()
    if df_pid.empty:
        raise ValueError(f"No rows for pid={pid}")
    df_pid = ensure_day(df_pid).sort_values("day").reset_index(drop=True)

    Xs, Xe, Xd_raw, lengths, gender, s_slice, T_model = build_arrays_from_combined_fixedT(df_pid, model)

    ghot = ohe.transform(np.array([[gender]], dtype=object)).astype(np.float32)
    Xd = np.concatenate([Xd_raw, ghot], axis=1)
    expected = getattr(dem_scaler, "n_features_in_", None)
    if expected is not None and Xd.shape[1] != expected:
        raise ValueError(
            f"Demographics width mismatch: got {Xd.shape[1]}, dem_scaler expects {expected}. "
            f"Ensure gender_ohe.pkl from training is present."
        )
    Xd_scaled = dem_scaler.transform(Xd)

    Xs_scaled = fit_like_transform_struct(Xs.copy(), lengths, struct_scaler)

    base_mean, base_vec = predict_mean_target(model, Xs_scaled, Xe, Xd_scaled, y_scaler, lengths)

    best = {"field": None, "delta": 0.0, "mean": base_mean, "improvement": 0.0, "y_vec": base_vec}

    for field in TUNABLE_FIELDS:
        vals = pd.to_numeric(df_pid[field], errors="coerce").astype(np.float32)
        vals = vals.ffill().bfill().fillna(0.0).to_numpy()
        for delta in (+STEP, -STEP):
            s_mod = s_slice.copy()
            rname = R_MAP[field]
            s_mod[rname] = np.clip(vals + delta, 1.0, 5.0).astype(np.float32)

            Xs_mod_seq = s_mod.to_numpy(np.float32)
            Xs_mod, _ = pad_or_trim_to_T(Xs_mod_seq, T_model)

            # normalize the new value
            Xs_mod_scaled = fit_like_transform_struct(Xs_mod.copy(), lengths, struct_scaler)

            mean_val, y_vec = predict_mean_target(model, Xs_mod_scaled, Xe, Xd_scaled, y_scaler, lengths)
            improvement = mean_val - base_mean
            if improvement > best["improvement"]:
                best = {"field": field, "delta": float(delta), "mean": float(mean_val),
                        "improvement": float(improvement), "y_vec": y_vec}

    return {
        "pid": pid,
        "baseline_mean": float(base_mean),
        "best_field": best["field"],
        "best_delta": best["delta"],
        "best_mean": best["mean"],
        "improvement": best["improvement"],
        #"baseline_vector": base_vec.tolist(),
        #"best_vector": best["y_vec"].tolist(),
    }

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    df = pd.read_csv(COMBINED_CSV, engine="python")

    need = {"pid", "age", "education", "gender", "acc", "out", "score"} | set(R_MAP.keys())
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: combined_output.csv missing columns: {sorted(missing)}")

    pids = df["pid"].drop_duplicates().tolist()

    results = []
    #for pid in ["p001"]:
    for pid in pids:
        try:
            res = select_best_single_change_for_pid_from_combined(df, pid)
            print(res)
            results.append(res)
        except Exception as e:
            print(f"[WARN] Skipping {pid}: {e}")

    if results:
        pd.DataFrame(results).to_csv("suggestions.csv", index=False)
    else:
        print("No results produced.")
