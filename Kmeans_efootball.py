# ============================================================
# K-MEANS CLUSTERING eFootball PES 2020
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# LOAD & AMBIL 500 PEMAIN RATING TERTINGGI
df_full = pd.read_csv('deets-updated.csv')
df = df_full.nlargest(500, 'overall_rating').reset_index(drop=True)

# 5 FITUR UTAMA
fitur = ['speed', 'dribbling', 'finishing', 'defensive_awareness', 'overall_rating']

# FILTER POSISI
def kelompok_posisi(pos):
    if pos in ['CF', 'SS', 'LWF', 'RWF']:   return 'Penyerang'
    elif pos in ['CMF', 'AMF', 'DMF', 'LMF', 'RMF']: return 'Gelandang'
    elif pos in ['CB', 'LB', 'RB']:           return 'Bek'
    elif pos == 'GK':                          return 'Kiper'
    else:                                      return 'Lainnya'

df['kelompok_posisi'] = df['registered_position'].apply(kelompok_posisi)

# NORMALISASI
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[fitur].dropna())

# CARI K OPTIMAL
best_k, best_sil = 2, 0
print("K Optimal:")
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(df_scaled)
    sil = silhouette_score(df_scaled, km.labels_)
    tanda = " ← terbaik" if sil > best_sil else ""
    if sil > best_sil:
        best_sil = sil
        best_k = k
    print(f"  K={k}  Silhouette={sil:.4f}{tanda}")

# JALANKAN K-MEANS
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(df_scaled)

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
def rekomendasi(nama_pemain, top_n=5):
    if nama_pemain not in df['name'].values:
        mirip = df[df['name'].str.contains(nama_pemain, case=False, na=False)]['name'].head(3).tolist()
        print(f"\n   '{nama_pemain}' tidak ditemukan. Mungkin: {mirip}")
        return

    pemain       = df[df['name'] == nama_pemain].iloc[0]
    c            = pemain['cluster']
    posisi       = pemain['kelompok_posisi']

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

    centroid            = kmeans.cluster_centers_[c]
    kandidat_scaled     = scaler.transform(kandidat[fitur].fillna(0))
    kandidat['jarak']   = np.linalg.norm(kandidat_scaled - centroid, axis=1)
    kandidat['mirip_%'] = (100 - kandidat['jarak'] / kandidat['jarak'].max() * 100).round(1)

    for _, row in kandidat.nsmallest(top_n, 'jarak').iterrows():
        print(f"  {row['name']:<25} {row['registered_position']:<8} {row['overall_rating']:<8.0f} {row['mirip_%']}%")

print("\nSistem Rekomendasi Pemain Pengganti:")
print("=" * 55)
rekomendasi('NEYMAR', top_n=3)
rekomendasi('L. MESSI', top_n=3)
rekomendasi('K. DE BRUYNE',top_n=3)