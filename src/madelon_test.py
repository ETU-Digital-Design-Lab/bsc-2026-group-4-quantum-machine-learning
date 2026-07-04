import time
import warnings
import pandas as pd
import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ==========================================
# 1. CPU C++ SİMÜLATÖRÜ (lightning.qubit)
# ==========================================
def get_device(wires):
    return qml.device("lightning.qubit", wires=wires)

# ==========================================
# 2. VERİ YÜKLEME VE ÖN İŞLEME (SADECE MADELON)
# ==========================================
def load_data(sample_limit=150):
    print("   [!] Madelon veri seti OpenML'den indiriliyor (Bu işlem 10-15 saniye sürebilir)...")
    # NIPS 2003 karmaşık veri seti (500 öznitelik)
    data = fetch_openml(name='madelon', version=1, parser='auto')
    X = np.array(data.data, dtype=float)
    y = np.array(data.target, dtype=int)

    # Standart Binary (0 ve 1) formata dönüştürür
    y = (y == np.unique(y)[0]).astype(int)
    
    # Kuantum simülasyon sürelerini makul tutmak için örneklem limiti
    if len(X) > sample_limit:
        X, _, y, _ = train_test_split(X, y, train_size=sample_limit, stratify=y, random_state=42)
        
    return X, y

def prepare_features(X_tr, X_te, enc_type, q):
    if enc_type == "Amplitude":
        dim = 2**q
        if X_tr.shape[1] > dim:
            pca = PCA(n_components=dim)
            X_tr, X_te = pca.fit_transform(X_tr), pca.transform(X_te)
        elif X_tr.shape[1] < dim:
            pad = dim - X_tr.shape[1]
            X_tr = np.pad(X_tr, ((0,0), (0,pad)), 'constant')
            X_te = np.pad(X_te, ((0,0), (0,pad)), 'constant')
        X_tr, X_te = normalize(X_tr, norm='l2'), normalize(X_te, norm='l2')
    else:
        # Angle Encoding Mantığı
        pca = PCA(n_components=q)
        X_tr, X_te = pca.fit_transform(X_tr), pca.transform(X_te)
        scaler = MinMaxScaler((0, np.pi))
        X_tr, X_te = scaler.fit_transform(X_tr), scaler.transform(X_te)
        
    return np.array(X_tr, requires_grad=False), np.array(X_te, requires_grad=False)

# ==========================================
# 3. PENNYLANE KUANTUM DEVRELERİ
# ==========================================
def encode_data(x, enc_type, q):
    if enc_type == "Amplitude":
        qml.AmplitudeEmbedding(x, wires=range(q), normalize=True, pad_with=0.0)
    else:
        qml.AngleEmbedding(x, wires=range(q))

def qsvm_kernel(X_train, X_test, enc_type, q):
    dev = get_device(q)
    
    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        encode_data(x1, enc_type, q)
        qml.adjoint(encode_data)(x2, enc_type, q)
        return qml.probs(wires=range(q))
    
    K_train = np.zeros((len(X_train), len(X_train)))
    for i in range(len(X_train)):
        for j in range(i, len(X_train)):
            val = kernel_circuit(X_train[i], X_train[j])[0]
            K_train[i, j] = K_train[j, i] = val
            
    K_test = np.zeros((len(X_test), len(X_train)))
    for i in range(len(X_test)):
        for j in range(len(X_train)):
            K_test[i, j] = kernel_circuit(X_test[i], X_train[j])[0]
            
    return K_train, K_test

def run_vqc(X_train, y_train, X_test, enc_type, q, iters=10):
    dev = get_device(q)
    
    @qml.qnode(dev, diff_method="adjoint")
    def vqc_circuit(weights, x):
        encode_data(x, enc_type, q)
        qml.StronglyEntanglingLayers(weights, wires=range(q))
        return qml.expval(qml.PauliZ(0))

    y_tr_vqc = np.where(y_train == 0, -1, 1)
    
    def cost(weights, X, Y):
        predictions = np.array([vqc_circuit(weights, x) for x in X])
        return np.mean((predictions - Y)**2)
    
    np.random.seed(42)
    weights = np.random.random(qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=q), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.1)
    
    batch_size = min(16, len(X_train))
    for i in range(iters):
        batch_idx = np.random.randint(0, len(X_train), (batch_size,))
        weights = opt.step(lambda w: cost(w, X_train[batch_idx], y_tr_vqc[batch_idx]), weights)
        
    preds = np.array([vqc_circuit(weights, x) for x in X_test])
    return np.where(preds >= 0.0, 1, 0)

# ==========================================
# 4. BENCHMARK VE KIYASLAMA MOTORU
# ==========================================
def execute_benchmark():
    models = ["VQC", "QSVM", "Classical_SVM(RBF)"]
    encodings = ["Angle", "Amplitude"]
    qubits_list = [3, 4]
    validation_methods = ["Train/Test Split", "2-Fold CV"]
    
    results = []
    
    print("=" * 80)
    print("🚀 SADECE MADELON: QML & CML BENCHMARK MOTORU BAŞLADI")
    print("Cihaz: PennyLane (lightning.qubit) CPU Hızlandırıcı Aktif")
    print("=" * 80)

    print("\n🔄 Veri Seti İşleniyor: Madelon (500 Öznitelik)")
    X, y = load_data()
    X = StandardScaler().fit_transform(X)
    
    for val_method in validation_methods:
        splits = []
        if val_method == "Train/Test Split":
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            splits.append((X_tr, X_te, y_tr, y_te))
        else:
            skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
            for tr_idx, te_idx in skf.split(X, y):
                splits.append((X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]))

        for q in qubits_list:
            for enc in encodings:
                for m in models:
                    if m == "Classical_SVM(RBF)" and (q != qubits_list[0] or enc != encodings[0]):
                        continue
                    
                    enc_label = "N/A" if m.startswith("Classical") else enc
                    q_label = "N/A" if m.startswith("Classical") else q
                    
                    print(f" └─ {val_method[:6]} | {m:<18} | {enc_label:<9} | {q_label} Q... ", end="", flush=True)
                    
                    accs, times = [], []
                    for X_tr, X_te, y_tr, y_te in splits:
                        start = time.time()
                        
                        if m == "Classical_SVM(RBF)":
                            clf = SVC(kernel="rbf")
                            clf.fit(X_tr, y_tr)
                            preds = clf.predict(X_te)
                        else:
                            X_tr_q, X_te_q = prepare_features(X_tr, X_te, enc, q)
                            
                            if m == "VQC":
                                preds = run_vqc(X_tr_q, y_tr, X_te_q, enc, q, iters=15)
                            elif m == "QSVM":
                                K_tr, K_te = qsvm_kernel(X_tr_q, X_te_q, enc, q)
                                clf = SVC(kernel="precomputed")
                                clf.fit(K_tr, y_tr)
                                preds = clf.predict(K_te)
                        
                        accs.append(accuracy_score(y_te, preds))
                        times.append(time.time() - start)
                        
                    avg_acc = np.mean(accs)
                    avg_time = np.sum(times) if val_method == "2-Fold CV" else np.mean(times)
                    
                    print(f"✔ [Acc: {avg_acc:.4f} | Süre: {avg_time:.2f}s]")
                    
                    results.append({
                        "Dataset": "Madelon", "Validation": val_method, "Model": m,
                        "Encoding": enc_label, "Qubits": q_label, 
                        "Accuracy": avg_acc, "Time(s)": avg_time
                    })

    df = pd.DataFrame(results)
    df.to_csv("qml_benchmark_madelon_only.csv", index=False)
    analyze_results(df)

# ==========================================
# 5. TABLOLAMA VE ANALİZ BÖLÜMÜ
# ==========================================
def analyze_results(df):
    print("\n" + "=" * 80)
    print("📊 MADELON ÖZEL ANALİZ RAPORU (500 Öznitelik Sıkıştırması)")
    print("=" * 80)

    print("\n📌 1. K-FOLD DOĞRULAMASININ PERFORMANS VE ZAMAN ETKİSİ")
    val_perf = df.groupby("Validation")[["Accuracy", "Time(s)"]].mean()
    print(val_perf.to_string(formatters={"Accuracy": "{:.4f}".format, "Time(s)": "{:.2f}s".format}))

    print("\n📌 2. VQC vs QSVM vs KLASİK SVM (MODEL FARKLARI)")
    mod_perf = df.groupby("Model")[["Accuracy", "Time(s)"]].mean()
    print(mod_perf.to_string(formatters={"Accuracy": "{:.4f}".format, "Time(s)": "{:.2f}s".format}))

    print("\n📌 3. ANGLE vs AMPLITUDE KODLAMA ETKİSİ (Sadece QML Modelleri)")
    qml_df = df[df["Encoding"] != "N/A"]
    enc_perf = qml_df.groupby("Encoding")[["Accuracy", "Time(s)"]].mean()
    print(enc_perf.to_string(formatters={"Accuracy": "{:.4f}".format, "Time(s)": "{:.2f}s".format}))

    print("\n📌 4. QUBIT SAYISININ ÖLÇEKLENME ETKİSİ (Sadece QML Modelleri)")
    q_perf = qml_df.groupby("Qubits")[["Accuracy", "Time(s)"]].mean()
    print(q_perf.to_string(formatters={"Accuracy": "{:.4f}".format, "Time(s)": "{:.2f}s".format}))

if __name__ == "__main__":
    execute_benchmark()
