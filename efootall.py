# ============================================================
# K-MEANS CLUSTERING eFootball PES 2020
# ============================================================

import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# LOAD & AMBIL 500 PEMAIN RATING TERTINGGI
df_full = pd.read_csv('deets-updated.csv') 
df = df_full.nlargest(500, 'overall_rating').reset_index(drop=True)

# 5 FITUR (atribut) UTAMA
fitur = ['speed', 'dribbling', 'finishing', 'defensive_awareness', 'overall_rating']

# FILTER POSISI
def kelompok_posisi(pos):
    if pos in ['CF', 'SS', 'LWF', 'RWF']:            return 'Penyerang'
    elif pos in ['CMF', 'AMF', 'DMF', 'LMF', 'RMF']: return 'Gelandang'
    elif pos in ['CB', 'LB', 'RB']:                   return 'Bek'
    elif pos == 'GK':                                  return 'Kiper'
    else:                                              return 'Lainnya'

df['kelompok_posisi'] = df['registered_position'].apply(kelompok_posisi)

# ============================================================
# NORMALISASI MANUAL (ganti StandardScaler)
# rumus: z = (x - mean) / std
# ============================================================
X = df[fitur].values.astype(float)
mean_val        = np.mean(X, axis=0) #cari rata-rata nilai per kolom
std_val         = np.std(X, axis=0)
std_val[std_val == 0] = 1
df_scaled       = (X - mean_val) / std_val

# ============================================================
# K-MEANS MANUAL (ganti KMeans dari sklearn)
# ============================================================
def kmeans_manual(X, k,  max_iter=100, random_state=42):
    np.random.seed(random_state)
    centroid = X[np.random.choice(len(X), k, replace=False)].copy()
    labels   = np.zeros(len(X), dtype=int)

    for _ in range(max_iter):
        labels_lama = labels.copy()

        # ASSIGN: tiap pemain ke centroid terdekat
        for i in range(len(X)):
            jarak    = [np.sqrt(np.sum((X[i] - centroid[j]) ** 2)) for j in range(k)]
            labels[i] = np.argmin(jarak)

        # UPDATE: centroid baru = rata-rata anggota cluster
        for j in range(k):
            anggota = X[labels == j]
            if len(anggota) > 0:
                centroid[j] = np.mean(anggota, axis=0)

        # KONVERGEN: berhenti jika tidak ada perubahan
        if np.array_equal(labels, labels_lama):
            break

    return labels, centroid

# ============================================================
# SILHOUETTE SCORE MANUAL (ganti silhouette_score dari sklearn)
# rumus: s = (b - a) / max(a, b)
# ============================================================
def silhouette_score_manual(X, labels):
    scores = []
    for i in range(len(X)):
        cluster_i = labels[i]
        same      = X[labels == cluster_i] #ambil pemain yang clusternya sama dengan pemaain i
        a = np.mean([np.sqrt(np.sum((X[i] - same[j])**2)) for j in range(len(same)) if not np.array_equal(X[i], same[j])]) if len(same) > 1 else 0
        b_list = []
        for oc in np.unique(labels):
            if oc != cluster_i:
                other = X[labels == oc]
                b_list.append(np.mean([np.sqrt(np.sum((X[i] - other[j])**2)) for j in range(len(other))]))
        b = min(b_list) if b_list else 0
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0)
    return np.mean(scores)

# ============================================================
# CARI K OPTIMAL
# ============================================================
best_k, best_sil = 2, 0
sample_idx = np.random.choice(len(df_scaled), 100, replace=False)

print("K Optimal:")
for k in range(2, 8):
    labels_k, centroids_k = kmeans_manual(df_scaled, k) 
    sil   = silhouette_score_manual(df_scaled[sample_idx], labels_k[sample_idx])
    tanda = " ← terbaik" if sil > best_sil else ""
    if sil > best_sil:
        best_sil = sil
        best_k   = k
        best_labels    = labels_k
        best_centroids = centroids_k 
    print(f"  K={k}  Silhouette={sil:.4f}{tanda}")

# SIMPAN HASIL TERBAIK
df['cluster'] = best_labels

# NAMA CLUSTER OTOMATIS
hasil = df.groupby('cluster')[fitur].mean().round(1)
nama_cluster = {}
for c in sorted(df['cluster'].unique()):
    row = hasil.loc[c]
    if row['finishing'] >= 80 and row['speed'] >= 80:
        nama = 'Penyerang Cepat'
    elif row['finishing'] >= 80:
        nama = 'Striker / Target Man'
    elif row['defensive_awareness'] >= 75:
        nama = 'Pemain Bertahan'
    elif row['dribbling'] >= 80:
        nama = 'Playmaker / Dribbler'
    else:
        nama = 'Pemain Serba Bisa'
    nama_cluster[c] = nama

# HASIL CLUSTER
print(f"\nHasil Cluster (K={best_k}):")
print("-" * 55)
for c in sorted(df['cluster'].unique()):
    jumlah = len(df[df['cluster'] == c])
    print(f"  Cluster {c} ({nama_cluster[c]}) — {jumlah} pemain")

# CONTOH PEMAIN TIAP CLUSTER
print("\nContoh Pemain Tiap Cluster:")
print("-" * 55)
for c in sorted(df['cluster'].unique()):
    print(f"\n  Cluster {c} — {nama_cluster[c]}:")
    contoh = df[df['cluster'] == c].nlargest(3, 'overall_rating')[
        ['name', 'registered_position', 'overall_rating']
    ]
    for _, row in contoh.iterrows():
        print(f"    {row['name']:<25} {row['registered_position']:<6} Rating: {row['overall_rating']:.0f}")

# SISTEM REKOMENDASI
def rekomendasi(nama_pemain, top_n=3):
    if nama_pemain not in df['name'].values:
        mirip = df[df['name'].str.contains(nama_pemain, case=False, na=False)]['name'].head(3).tolist()
        print(f"\n   '{nama_pemain}' tidak ditemukan. Mungkin: {mirip}")
        return

    pemain = df[df['name'] == nama_pemain].iloc[0]
    c      = pemain['cluster']
    posisi = pemain['kelompok_posisi']

    print(f"\n  Pengganti untuk: {nama_pemain}")
    print(f"  Posisi: {pemain['registered_position']} | Cluster: {c} ({nama_cluster[c]}) | Rating: {pemain['overall_rating']:.0f}")
    print(f"  {'Nama':<25} {'Posisi':<8} {'Rating':<8} {'Kemiripan'}")
    print(f"  {'-' * 50}")

    kandidat = df[
        (df['cluster'] == c) &
        (df['kelompok_posisi'] == posisi) &
        (df['name'] != nama_pemain)
    ].copy()

    if kandidat.empty:
        print("  Tidak ada kandidat.")
        return

    centroid = best_centroids[c]
    k_scaled = (kandidat[fitur].values.astype(float) - mean_val) / std_val
    kandidat['jarak']   = [np.sqrt(np.sum((k_scaled[i] - centroid) ** 2)) for i in range(len(k_scaled))]
    kandidat['mirip_%'] = (100 - kandidat['jarak'] / kandidat['jarak'].max() * 100).round(1)

    for _, row in kandidat.nsmallest(top_n, 'jarak').iterrows():
        print(f"  {row['name']:<25} {row['registered_position']:<8} {row['overall_rating']:<8.0f} {row['mirip_%']}%")

print("\nSistem Rekomendasi Pemain Pengganti:")
print("=" * 55)
rekomendasi('NEYMAR')
rekomendasi('L. MESSI')
rekomendasi('K. DE BRUYNE')