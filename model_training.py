import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import pickle

print("1.STEP")
df = pd.read_excel("train_test_data.xlsx")


df.columns = df.columns.str.strip()

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


df_filtered = df[selected_features]
print(f"Success! Dataset filtered. Current Shape: {df_filtered.shape[0]} Firms, {df_filtered.shape[1]} Features.")


print("2.STEP")
counts = df_filtered['Bankrupt?'].value_counts()
percentages = df_filtered['Bankrupt?'].value_counts(normalize=True) * 100

sağlam_sayısı = counts.get(0, 0)
iflas_sayısı = counts.get(1, 0)
sağlam_yüzde = percentages.get(0, 0.0)
iflas_yüzde = percentages.get(1, 0.0)

print(f"Healthy Firms (0): {healthy_count} ({healthy_pct:.2f}%)")
print(f"Bankrupt Firms (1): {bankrupt_count} ({bankrupt_pct:.2f}%)")


df_filtered.to_excel("final_filtered_data.xlsx", index=False)
print("Filtered clean dataset successfully exported as 'final_filtered_data.xlsx")


print("3.STEP")
plt.rcParams['figure.dpi'] = 150
sns.set_theme(style="whitegrid")


plt.figure(figsize=(6, 4))
ax = sns.countplot(x='Bankrupt?', data=df_filtered, palette='Set2')
plt.title('Target Variable Distribution (Class Imbalance Analysis)', fontsize=12, fontweight='bold')
plt.xlabel('Corporate Status (0: Healthy, 1: Bankrupt)', fontsize=10)
plt.ylabel('Firm Count', fontsize=10)

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)

plt.tight_layout()
plt.savefig('eda_target_distribution.png')
plt.close()
print("Graph 1 (eda_target_distribution.png) generated and saved.")


plt.figure(figsize=(12, 10))
corr_matrix = df_filtered.drop(columns=['Bankrupt?']).corr()

sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5, annot_kws={"size": 7})
plt.title('Finansal Rasyolar Arası Korelasyon Isı Haritası', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig('eda_correlation_heatmap.png')
plt.close()
print("Graph 2 (eda_correlation_heatmap.png) generated and saved.")


print("4.STEP")

X = df_filtered.drop(columns=['Bankrupt?'])
y = df_filtered['Bankrupt?']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Training Set Partition Shape: {X_train.shape[0]} firms")
print(f"Validation Set Partition Shape: {X_test.shape[0]} firms")


print("5.STEP")

scale_weight = healthy_count / bankrupt_count


model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=scale_weight,  
    random_state=42,
    eval_metric='logloss'
)


model.fit(X_train, y_train)
print("XGBoost Predictive Architecture Successfully Trained!")


print("6.STEP")

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Model Accuracy: {accuracy*100:.2f}%")

print("\nDetailed Performance Diagnostics (Classification Report):")
print(classification_report(y_test, y_pred))


print("7.STEP")

with open("xgboost_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Trained model binary successfully serialized as 'xgboost_model.pkl'")
print("\n--- End-to-End Pipeline Execution Completed Successfully ---")
