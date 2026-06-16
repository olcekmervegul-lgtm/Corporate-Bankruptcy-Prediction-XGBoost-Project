import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import pickle

print("--- 1. ADIM: VERİ SEÇİMİ VE OTOMATİK BOŞLUK TEMİZLEME ---")
# data_split.py ile oluşturduğumuz eğitim verisini okuyoruz
df = pd.read_excel("train_test_data.xlsx")

# Sütun isimlerinin başındaki ve sonundaki gizli boşlukları siliyoruz
df.columns = df.columns.str.strip()

# Seninle birlikte belirlediğimiz 15 rasyo + hedef değişken
selected_features = [
    'Bankrupt?',
    'Current Ratio',
    'Quick Ratio',
    'Cash/Total Assets',
    'Debt ratio %',
    'Borrowing dependency',
    'Liability to Equity',
    'Equity to Liability',
    'Cash Flow to Liability',
    'Cash Flow to Total Assets',
    'Operating Funds to Liability',
    'Net Income to Total Assets',
    'Retained Earnings to Total Assets',
    'Working Capital to Total Assets',
    'Current Liability to Assets'
]

# Veriyi sadece bu 15 sütun kalacak şekilde süzüyoruz
df_filtered = df[selected_features]
print(f"Başarılı! Veri seti süzüldü. Mevcut Boyut: {df_filtered.shape[0]} Şirket, {df_filtered.shape[1]} Özellik.")


print("\n--- 2. ADIM: HEDEF DEĞİŞKEN (BANKRUPT?) DAĞILIMI ---")
counts = df_filtered['Bankrupt?'].value_counts()
percentages = df_filtered['Bankrupt?'].value_counts(normalize=True) * 100

sağlam_sayısı = counts.get(0, 0)
iflas_sayısı = counts.get(1, 0)
sağlam_yüzde = percentages.get(0, 0.0)
iflas_yüzde = percentages.get(1, 0.0)

print(f"Sağlam Şirket Sayısı (0): {sağlam_sayısı} (%{sağlam_yüzde:.2f})")
print(f"İflas Eden Şirket Sayısı (1): {iflas_sayısı} (%{iflas_yüzde:.2f})")

# Süzülmüş temiz veriyi yeni bir dosyaya kaydediyoruz
df_filtered.to_excel("final_filtered_data.xlsx", index=False)
print("Filtrelenmiş temiz veri 'final_filtered_data.xlsx' olarak kaydedildi!")


print("\n--- 3. ADIM: GÖRSELLEŞTİRME VE EDA GRAFİKLERİ ---")
plt.rcParams['figure.dpi'] = 150
sns.set_theme(style="whitegrid")

# GRAFİK 1: Hedef Değişken Dağılımı (Bar Plot)
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='Bankrupt?', data=df_filtered, palette='Set2')
plt.title('Hedef Değişken Dağılımı (Sınıf Dengesizliği)', fontsize=12, fontweight='bold')
plt.xlabel('Şirket Durumu (0: Sağlam, 1: İflas)', fontsize=10)
plt.ylabel('Şirket Sayısı', fontsize=10)

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)

plt.tight_layout()
plt.savefig('eda_target_distribution.png')
plt.close()
print("1. Grafik (eda_target_distribution.png) kaydedildi!")

# GRAFİK 2: Seçtiğimiz 15 Rasyonun Korelasyon Isı Haritası (Correlation Heatmap)
plt.figure(figsize=(12, 10))
corr_matrix = df_filtered.drop(columns=['Bankrupt?']).corr()

sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5, annot_kws={"size": 7})
plt.title('Finansal Rasyolar Arası Korelasyon Isı Haritası', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig('eda_correlation_heatmap.png')
plt.close()
print("2. Grafik (eda_correlation_heatmap.png) kaydedildi!")


print("\n--- 4. ADIM: TRAIN / TEST SPLIT (VERİYİ BÖLME) ---")
# Hedef değişken (y) ve bağımsız değişkenleri (X) ayırıyoruz [cite: 51, 54]
X = df_filtered.drop(columns=['Bankrupt?'])
y = df_filtered['Bankrupt?']

# %80 Eğitim, %20 Test olacak şekilde bölüyoruz [cite: 60]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Eğitim Seti Boyutu: {X_train.shape[0]} şirket")
print(f"Test Seti Boyutu: {X_test.shape[0]} şirket")


print("\n--- 5. ADIM: XGBOOST MODELİNİN EĞİTİLMESİ ---")
# Sınıf dengesizliğini çözmek için ağırlık hesaplıyoruz (Sağlam / İflas)
scale_weight = sağlam_sayısı / iflas_sayısı

# XGBoost Sınıflandırıcısını tanımlıyoruz [cite: 61]
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=scale_weight,  # Azınlık sınıfı olan iflasları koruma formülü
    random_state=42,
    eval_metric='logloss'
)

# Modeli eğitiyoruz
model.fit(X_train, y_train)
print("XGBoost Modeli Başarıyla Eğitildi!")


print("\n--- 6. ADIM: MODEL PERFORMANSI VE METRİKLER ---")
# Test setini modele verip tahmin alıyoruz
y_pred = model.predict(X_test)

# Başarı skorlarını ekrana basıyoruz [cite: 63]
accuracy = accuracy_score(y_test, y_pred)
print(f"Modelin Genel Doğruluk Oranı (Accuracy): %{accuracy*100:.2f}")

print("\nDetaylı Performans Raporu (Classification Report):")
print(classification_report(y_test, y_pred))


print("\n--- 7. ADIM: MODELİN GELECEKTE KULLANIM İÇİN KAYDEDİLMESİ ---")
# Eğittiğimiz bu zeki modeli bilgisayara dosya olarak yazıyoruz
with open("xgboost_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model 'xgboost_model.pkl' ismiyle başarıyla kaydedildi!")
print("\nTüm süreç tamamlandı!")