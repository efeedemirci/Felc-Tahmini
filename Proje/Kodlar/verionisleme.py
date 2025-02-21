import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('stroke.csv', sep=',')

print("\nVeri Bilgisi:")
df.info()

# 1. Tutarsız değerleri düzenleme
# Çalışma türü sütunundaki değerlerdeki tutarsızlıkları giderme
df['work_type'] = df['work_type'].replace({
    'Never_worked': 'Never Worked',
    'Govt_job': 'Government Job'
})

# Sigara içme durumu sütunundaki değerlerdeki tutarsızlıkları giderme
df['smoking_status'] = df['smoking_status'].replace({
    'Formerly Smoked': 'formerly smoked',
    'Currently Smokes': 'smokes',
    'Non-smoker': 'never smoked'
})

# Evli olup olmama durumundaki değerlerdeki tutarsızlıkları giderme
df['ever_married'] = df['ever_married'].replace({
    'Yes': 'Married',
    'No': 'Single',
})


# 2. Eksik ve gereksiz değerlerin temizlenmesi
# Eksik değerleri kaldırma
df = df.dropna()
df.info()
# Tekrarlayan satırları kaldırma
df = df.drop_duplicates()
df.info()
# "Unknown" değerine sahip satırları kaldırma
df = df[df['smoking_status'] != 'Unknown']
# "children" değerine sahip satırları kaldırma
df = df[df['work_type'] != 'children']
# "Other" olarak belirtilen cinsiyet değerlerini kaldırma
df = df[df['gender'] != 'Other']

excludes =['gender','ever_married','work_type','Residence_type','smoking_status','hypertension','stroke','heart_disease']

# Aykırı değerlerin IQR yöntemi ile analizi ve çıkarılması
def iqr_hesaplama(data, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []

    for column in data.columns:
        if column not in exclude_cols:
            Q1 = data[column].quantile(0.25)  # İlk çeyrek
            Q3 = data[column].quantile(0.75)  # Üçüncü çeyrek
            IQR = Q3 - Q1  # IQR hesaplama
            lower_bound = Q1 - 1.5 * IQR  # Alt sınır
            upper_bound = Q3 + 1.5 * IQR  # Üst sınır
            data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
            plt.figure(figsize=(10, 6))
            plt.boxplot(df[column], vert=False, patch_artist=True)
            plt.title(f'{column} değişkeninin kutu grafiği')
            plt.xlabel('Değerler')
            plt.grid(True)
            plt.show()
    # Sadece sınırlar içinde kalanları seçiyoruz
    return data

# Aykırı değerleri çıkarma
df = iqr_hesaplama(df,excludes)

# BMI değerini mantıklı bir aralık olan 10 ile 60 arasında filtreleme
df = df[(df['bmi'] >= 10) & (df['bmi'] <= 60)]

# 3. Categorical verileri düzenleme ve dönüştürme
df['smoking_status'] = df['smoking_status'].replace({'formerly smoked': 3, 'smokes': 2, 'never smoked': 1})
df['gender'] = df['gender'].replace({'Female': 1, 'Male': 0})
df['ever_married'] = df['ever_married'].replace({'Single':1,'Married':2,'Divorced':3})
df['work_type'] = df['work_type'].replace({'Private':1,'Self-employed':2,'Government Job':3,'Never Worked':4})
df['Residence_type'] = df['Residence_type'].replace({'Urban':1,'Rural':2})

print(df.head(5))

for col in df.columns:
    df[col].plot(kind='hist', bins=100, alpha=0.6)
    plt.title(f'{col} veri dağılımı')
    plt.xlabel(f'{col}')
    plt.show()

# Cramér's V hesaplama fonksiyonu
def cramers_V(var1,var2) :
  crosstab =np.array(pd.crosstab(var1,var2, rownames=None, colnames=None)) # Cross table building
  stat = chi2_contingency(crosstab)[0] # Keeping of the test statistic of the Chi2 test
  obs = np.sum(crosstab) # Number of observations
  mini = min(crosstab.shape)-1 # Take the minimum value between the columns and the rows of the cross table
  return np.sqrt(stat/(obs*mini))

rows = []
for var1 in df:
    col = []
    for var2 in df:
        cramers = cramers_V(df[var1], df[var2])  # Cramer's V test
        col.append(round(cramers, 2))  # Keeping of the rounded value of the Cramer's V
    rows.append(col)

cramers_results = np.array(rows)
cramers_df = pd.DataFrame(cramers_results, columns=df.columns, index=df.columns)

# Check correlation with heatmap

fig, ax = plt.subplots(figsize=(18,8))
corr_matrix = cramers_df.corr()
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_matrix, annot=True, mask=mask)

plt.title( 'Cramer\'s V Korelasyonu Isı Haritası', fontsize=20, fontweight='bold', fontfamily='serif', pad=15);
plt.xticks( rotation=0)
plt.show()

corr_matrix = df.corr()
target = corr_matrix['stroke']
print(target)

df = df.drop('Residence_type', axis=1)

scaler = MinMaxScaler()
df[['age','avg_glucose_level','bmi']] = scaler.fit_transform(df[['age','avg_glucose_level','bmi']])

df.info()
#df.to_csv('preprocessed_stroke.csv', index=False)