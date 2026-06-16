import pandas as pd

# 1. Excel dosyasını yüklüyoruz (Dosya adın boşluklu olduğu için tırnak içine aynen yazıyoruz)
file_name = "data.xls"
df = pd.read_excel(file_name)

print("Veri başarıyla yüklendi!")
print(f"Toplam Satır Sayısı: {df.shape[0]}, Toplam Sütun Sayısı: {df.shape[1]}")

# 2. Sunumda 'canlı demo' yapmak için en sondan 300 şirketi ayırıyoruz
demo_df = df.tail(300)

# 3. Geri kalan ~6500 satırı model eğitimi (train/test) için ayırıyoruz
train_test_df = df.head(len(df) - 300)

# 4. Bu iki yeni yapıyı bilgisayara ayrı ayrı kaydediyoruz ki karışmasınlar
demo_df.to_excel("demo_data_300.xlsx", index=False)
train_test_df.to_excel("train_test_data.xlsx", index=False)

print("Veriler bölündü ve 'demo_data_300.xlsx' ile 'train_test_data.xlsx' olarak kaydedildi!")