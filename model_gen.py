import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import keras  # Keras 3 (for serialization registry)

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, TimeDistributed, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import pearsonr


### script for generating the neural network model
### has functionality for k-fold validation, training the model on all data to save it, and testing the trained model for reasonable output

# ---------------------------
# Seed random seed and mode
# ---------------------------
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
mode = "train"  # "train" | "test" | "cv"

# ---------------------------
# Config
# ---------------------------
COMBINED_CSV = "combined_output.csv"
EMBED_PKL    = "daily_prompts_and_embeddings.pkl"  # optional; zeros if missing
TMB_CSV      = "tmb.csv"                # targets per pid
TIME_STEPS   = 41                                   # fixed T expected by model

#mapping some column names different keys
R_MAP = {
    "active_num":    "r1",
    "engaged_num":   "r2",
    "fatigue_num":   "r3",
    "anxious_num":   "r4",
    "depressed_num": "r5",
    "irritable_num": "r6",
    "social_num":    "r7",
}
#list of column names to use in training
STRUCT_COLS = ["acc", "out", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "score"]

#focus TMB scores to use as y values
TMB_TARGETS = [
    "TMB Choice Reaction Time",
    "TMB Digit Symbol Matching",
    "TMB Matrix Reasoning",
]

# ---------------------------
# custom layers to compute mean on non-padded data
# ---------------------------
@keras.saving.register_keras_serializable(package="lifeadapt")
class TimeMaskFromLength(tf.keras.layers.Layer):
    def __init__(self, time_steps, **kwargs):
        super().__init__(**kwargs)
        self.time_steps = int(time_steps)

    def call(self, lengths):
        m = tf.sequence_mask(lengths, maxlen=self.time_steps)  # (B, T)
        return tf.cast(tf.expand_dims(m, -1), tf.float32)      # (B, T, 1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"time_steps": self.time_steps})
        return cfg

@keras.saving.register_keras_serializable(package="lifeadapt")
class MaskedMean(tf.keras.layers.Layer):
    def call(self, inputs):
        x, m = inputs  # x:(B,T,C), m:(B,T,1)
        x = x * m
        s = tf.reduce_sum(x, axis=1)           # (B, C)
        c = tf.reduce_sum(m, axis=1) + 1e-8    # (B, 1)
        return s / c

    def compute_output_shape(self, input_shape):
        # input_shape[0] is (B,T,C)
        return (input_shape[0][0], input_shape[0][2])

# ---------------------------
# Utils
# ---------------------------

#creates day index from date information
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

#make sure all time series have the same length (by padding or trimming)
def pad_or_trim_to_T(seq_2d: np.ndarray, T_target: int):
    seq_2d = np.asarray(seq_2d, dtype=np.float32)
    Ti, F = seq_2d.shape
    X = np.zeros((1, T_target, F), dtype=np.float32)
    T_copy = min(Ti, T_target)
    if T_copy > 0:
        X[0, :T_copy, :] = seq_2d[:T_copy, :]
    L = np.array([T_copy], dtype=np.int32)
    return X, L

#fits data to StandardScaler scaling (mean of 0, std of 1)
def fit_transform_struct(X: np.ndarray, lengths: np.ndarray):
    B, T, F = X.shape
    mask = (np.arange(T)[None, :] < lengths[:, None])
    X_flat = X.reshape(-1, F)
    mask_flat = mask.reshape(-1)
    scaler = StandardScaler()
    X_flat[mask_flat] = scaler.fit_transform(X_flat[mask_flat])
    return X_flat.reshape(B, T, F), scaler

#applies specified scaler (defaults to standardscaler) to data
def transform_struct(X: np.ndarray, lengths: np.ndarray, scaler: StandardScaler):
    B, T, F = X.shape
    mask = (np.arange(T)[None, :] < lengths[:, None])
    X_flat = X.reshape(-1, F)
    mask_flat = mask.reshape(-1)
    X_flat[mask_flat] = scaler.transform(X_flat[mask_flat])
    return X_flat.reshape(B, T, F)

# ---------------------------
# Data loading
# ---------------------------
def load_data():
    df = pd.read_csv(COMBINED_CSV, engine="python")
    need = {"pid", "age", "education", "gender", "acc", "out", "score"} | set(R_MAP.keys())
    missing = need - set(df.columns)
    if missing:
        raise RuntimeError(f"{COMBINED_CSV} is missing columns: {sorted(missing)}")

    df = ensure_day(df).sort_values(["pid", "day"]).reset_index(drop=True)

    # Optional embeddings
    have_embeddings = os.path.exists(EMBED_PKL)
    if have_embeddings:
        df_embed = pd.read_pickle(EMBED_PKL)
        if "day" not in df_embed.columns:
            df_embed = ensure_day(df_embed)
        df_embed = df_embed[["pid", "day", "embedding"]].drop_duplicates(["pid", "day"])
        df = df.merge(df_embed, on=["pid", "day"], how="left")

    # Structured slice with impute & clipping
    s = pd.DataFrame(index=df.index)
    s["acc"] = pd.to_numeric(df["acc"], errors="coerce").astype(np.float32)
    s["out"] = pd.to_numeric(df["out"], errors="coerce").astype(np.float32)
    for src, tgt in R_MAP.items():
        s[tgt] = pd.to_numeric(df[src], errors="coerce").astype(np.float32)
    s["score"] = pd.to_numeric(df["score"], errors="coerce").astype(np.float32)

    s[STRUCT_COLS] = s.groupby(df["pid"])[STRUCT_COLS].apply(lambda g: g.ffill().bfill()).reset_index(level=0, drop=True)
    for r in ["r1","r2","r3","r4","r5","r6","r7"]:
        s[r] = s[r].clip(1.0, 5.0)
    s[STRUCT_COLS] = s[STRUCT_COLS].fillna(0.0)

    pid_order = list(df["pid"].drop_duplicates())
    X_struct_list, X_embed_list, lengths_list = [], [], []

    # infer embed dim if present
    embed_dim = None
    if have_embeddings:
        sample = df["embedding"].dropna()
        if len(sample):
            embed_dim = int(np.array(sample.iloc[0], dtype=np.float32).shape[0])
    if embed_dim is None:
        embed_dim = 1536

    for pid in pid_order:
        mask_pid = (df["pid"] == pid)
        s_pid = s.loc[mask_pid, STRUCT_COLS].to_numpy(np.float32)
        Xs, Ls = pad_or_trim_to_T(s_pid, TIME_STEPS)

        if have_embeddings:
            emb_vals = df.loc[mask_pid, "embedding"].tolist()
            seq = []
            for v in emb_vals:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    seq.append(np.zeros((embed_dim,), dtype=np.float32))
                else:
                    arr = np.array(v, dtype=np.float32)
                    if arr.shape[0] != embed_dim:
                        arr = np.zeros((embed_dim,), dtype=np.float32)
                    seq.append(arr)
            Xe_pid = np.vstack([e[None, :] for e in seq])
        else:
            Xe_pid = np.zeros((s_pid.shape[0], embed_dim), dtype=np.float32)

        Xe, _ = pad_or_trim_to_T(Xe_pid, TIME_STEPS)

        X_struct_list.append(Xs)
        X_embed_list.append(Xe)
        lengths_list.append(Ls)

    X_struct = np.concatenate(X_struct_list, axis=0)
    X_embed  = np.concatenate(X_embed_list, axis=0)
    lengths  = np.concatenate(lengths_list, axis=0)

    # Demographics
    dem_rows = []
    for pid in pid_order:
        row = df.loc[df["pid"] == pid].iloc[0]
        age = float(pd.to_numeric(row["age"], errors="coerce"))
        edu = float(pd.to_numeric(row["education"], errors="coerce"))
        gen = str(row["gender"])
        dem_rows.append([pid, age, edu, gen])
    dem_df = pd.DataFrame(dem_rows, columns=["pid", "age", "education", "gender"]).set_index("pid")
    dem_num = dem_df[["age", "education"]].to_numpy(np.float32)
    gender_labels = dem_df["gender"].astype(object).to_numpy().reshape(-1, 1)

    # Targets
    ydf = pd.read_csv(TMB_CSV)
    y_raw = ydf.set_index("pid").loc[pid_order][TMB_TARGETS].to_numpy(np.float32)

    return X_struct, X_embed, (dem_num, gender_labels), y_raw, lengths

# ---------------------------
# Model
# ---------------------------
def build_model(struct_dim, embed_dim, dem_dim, time_steps, hidden_dim=64, output_dim=3):
    input_struct = Input(shape=(time_steps, struct_dim), name="struct_input")
    input_embed  = Input(shape=(time_steps, embed_dim),  name="embed_input")
    input_dem    = Input(shape=(dem_dim,),               name="dem_input")
    input_len    = Input(shape=(), dtype="int32",        name="seq_len")

    s = TimeDistributed(Dense(hidden_dim, activation='relu'))(input_struct)
    s = Dropout(0.2)(s)
    e = TimeDistributed(Dense(hidden_dim, activation='relu'))(input_embed)
    e = Dropout(0.2)(e)
    aligned = Concatenate(axis=-1)([s, e])  # (B, T, 2*hidden)

    mask_bt = TimeMaskFromLength(time_steps, name="time_mask")(input_len)     # (B,T,1)
    pooled  = MaskedMean(name="masked_mean")([aligned, mask_bt])              # (B,C)

    combined = Concatenate(axis=-1)([pooled, input_dem])                      # (B, C+dem_dim)
    h = Dense(hidden_dim, activation='relu')(combined)
    outputs = [Dense(1, name=f"task_{i+1}")(h) for i in range(output_dim)]

    model = Model(inputs=[input_struct, input_embed, input_dem, input_len], outputs=outputs)
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

# ---------------------------
# Demographics helpers
# ---------------------------

#apply one-hot encoder to gender and scaling to other demographic values
def fit_demographics(dem_num, gender_labels):
    try:
        ohe = OneHotEncoder(sparse_output=False, drop="if_binary")
    except TypeError:
        ohe = OneHotEncoder(sparse=False, drop="if_binary")
    gender_onehot = ohe.fit_transform(gender_labels)
    X_dem_raw = np.concatenate([dem_num, gender_onehot], axis=1)
    dscaler = StandardScaler()
    X_dem_scaled = dscaler.fit_transform(X_dem_raw)
    return X_dem_scaled, (ohe, dscaler)

#apply an already existing one-hot encoder and scaler to data
def transform_demographics(dem_num, gender_labels, ohe, dscaler):
    gender_onehot = ohe.transform(gender_labels)
    X_dem_raw = np.concatenate([dem_num, gender_onehot], axis=1)
    return dscaler.transform(X_dem_raw)

# ---------------------------
# CV / Train / Test
# ---------------------------

#perform k-fold cross validation
def run_cross_val(Xs, Xe, dem_pack, y, lengths, model_fn, n_splits=5, epochs=40, batch_size=16):
    dem_num, gender_labels = dem_pack
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    pearson_rs = []

    T, Ds, De = Xs.shape[1], Xs.shape[2], Xe.shape[2]
    D_out = y.shape[1]

    for fold, (tr, va) in enumerate(kf.split(Xs)):
        Xs_tr_raw, Xs_va_raw = Xs[tr], Xs[va]
        Xe_tr, Xe_va         = Xe[tr], Xe[va]
        dem_num_tr, dem_num_va = dem_num[tr], dem_num[va]
        gen_tr, gen_va       = gender_labels[tr], gender_labels[va]
        y_tr_raw, y_va_raw   = y[tr], y[va]
        L_tr, L_va           = lengths[tr], lengths[va]

        Xs_tr, struct_scaler = fit_transform_struct(Xs_tr_raw.copy(), L_tr)
        Xs_va = transform_struct(Xs_va_raw.copy(), L_va, struct_scaler)

        Xd_tr, (ohe, dem_scaler) = fit_demographics(dem_num_tr, gen_tr)
        Xd_va = transform_demographics(dem_num_va, gen_va, ohe, dem_scaler)

        y_scaler = StandardScaler()
        y_tr = y_scaler.fit_transform(y_tr_raw)
        y_va = y_scaler.transform(y_va_raw)

        y_tr_split = [y_tr[:, i] for i in range(D_out)]
        y_va_split = [y_va[:, i] for i in range(D_out)]

        model = model_fn(Ds, De, Xd_tr.shape[1], T, output_dim=D_out)
        tf.random.set_seed(42)
        model.fit(
            [Xs_tr, Xe_tr, Xd_tr, L_tr], y_tr_split,
            validation_data=([Xs_va, Xe_va, Xd_va, L_va], y_va_split),
            epochs=epochs, batch_size=batch_size, verbose=0,
            callbacks=[EarlyStopping(patience=7, restore_best_weights=True)]
        )

        y_pred_split = model.predict([Xs_va, Xe_va, Xd_va, L_va], verbose=0)
        y_pred = np.hstack([yp.reshape(-1, 1) for yp in y_pred_split])

        r = []
        for i in range(D_out):
            a, b = y_va[:, i], y_pred[:, i]
            r.append(pearsonr(a, b)[0] if (np.std(a) > 0 and np.std(b) > 0) else np.nan)
        pearson_rs.append(r)

        print(f"Fold {fold+1}:")
        for i in range(D_out):
            print(f"  Task {i+1} - Pearson r = {r[i]:.4f}")

    r_avg = np.nanmean(np.array(pearson_rs), axis=0)
    print("\nAverage performance:")
    for i in range(D_out):
        print(f"  Task {i+1} - Pearson r = {r_avg[i]:.4f}")
    return r_avg


#train the model on all data and save it
def run_train(Xs, Xe, dem_pack, y, lengths, model_fn, epochs=60, batch_size=16, val_split=0.0):
    dem_num, gender_labels = dem_pack
    T, Ds, De = Xs.shape[1], Xs.shape[2], Xe.shape[2]
    D_out = y.shape[1]

    Xs_scaled, struct_scaler = fit_transform_struct(Xs.copy(), lengths)
    Xd_scaled, (ohe, dem_scaler) = fit_demographics(dem_num, gender_labels)

    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)
    y_split = [y_scaled[:, i] for i in range(D_out)]

    model = model_fn(Ds, De, Xd_scaled.shape[1], T, output_dim=D_out)
    tf.random.set_seed(42)

    fit_kwargs = dict(x=[Xs_scaled, Xe, Xd_scaled, lengths], y=y_split, epochs=epochs, batch_size=batch_size, verbose=1)
    if val_split and val_split > 0.0:
        fit_kwargs.update(dict(validation_split=val_split, callbacks=[EarlyStopping(patience=7, restore_best_weights=True)]))
    model.fit(**fit_kwargs)

    joblib.dump(struct_scaler, "structscaler.pkl")
    joblib.dump(dem_scaler,    "demscaler.pkl")
    joblib.dump(y_scaler,      "yscaler.pkl")
    joblib.dump(ohe,           "gender_ohe.pkl")
    model.save("model.keras")
    print("Saved: model.keras, structscaler.pkl, demscaler.pkl, yscaler.pkl, gender_ohe.pkl")
    return model

#test already trained model to make sure it has reasonable output
def run_test(Xs, Xe, dem_pack, y, lengths):
    dem_num, gender_labels = dem_pack
    model = load_model("model.keras", compile=False)

    struct_scaler = joblib.load("structscaler.pkl")
    dem_scaler    = joblib.load("demscaler.pkl")
    y_scaler      = joblib.load("yscaler.pkl")
    try:
        ohe = joblib.load("gender_ohe.pkl")
    except Exception:
        try:
            ohe = OneHotEncoder(sparse_output=False, drop="if_binary").fit(gender_labels)
        except TypeError:
            ohe = OneHotEncoder(sparse=False, drop="if_binary").fit(gender_labels)

    Xs_scaled = transform_struct(Xs.copy(), lengths, struct_scaler)
    Xd_scaled = transform_demographics(dem_num, gender_labels, ohe, dem_scaler)
    y_true    = y

    y_pred_split = model.predict([Xs_scaled, Xe, Xd_scaled, lengths], verbose=0)
    y_pred_scaled = np.hstack([p.reshape(-1,1) for p in y_pred_split])
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    D_out = y_true.shape[1]
    pearsons, maes = [], []
    for i in range(D_out):
        a, b = y_true[:, i], y_pred[:, i]
        r = pearsonr(a, b)[0] if (np.std(a) > 0 and np.std(b) > 0) else np.nan
        pearsons.append(r)
        maes.append(np.mean(np.abs(a - b)))

    print("Test metrics on all data:")
    for i in range(D_out):
        print(f"  Task {i+1}: r = {pearsons[i]:.4f}, MAE = {maes[i]:.4f}")

    return {"pearson_r": pearsons, "mae": maes}, y_true, y_pred

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    X_struct, X_embed, dem_pack, y, lengths = load_data()
    if mode == "train":
        run_train(X_struct, X_embed, dem_pack, y, lengths, model_fn=build_model)
    elif mode == "test":
        run_test(X_struct, X_embed, dem_pack, y, lengths)
    else:
        run_cross_val(X_struct, X_embed, dem_pack, y, lengths, model_fn=build_model)
