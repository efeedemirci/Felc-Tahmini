import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

df = pd.read_csv('preprocessed_stroke.csv',sep=',')

x_train, x_test, y_train, y_test = train_test_split(df.drop('stroke',axis=1),df['stroke'],test_size=0.2,random_state=None)

stroke_counts = y_train.value_counts()
print("Orijinal Sınıf Dağılımı:\n",stroke_counts)

# SMOTE uygulama
smote = SMOTE(random_state=None)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

stroke_counts_resampled = y_train_resampled.value_counts()
print("SMOTE Sonrası Sınıf Dağılımı:\n",stroke_counts_resampled)

knn = KNeighborsClassifier()

param_dist = {'n_neighbors': np.arange(1,31),
              'weights':['uniform','distance'],
              'metric': ['euclidean','manhattan']}

# RandomizedSearchCV ile hiperparametre araması yap
random_search = RandomizedSearchCV(knn, param_dist, cv=5, random_state=None, n_jobs=-1)
random_search.fit(x_train_resampled, y_train_resampled)

# En iyi hiperparametreleri yazdır
print("En iyi hiperparametreler: ", random_search.best_params_)

# Eğitim ve test doğruluklarını saklamak için listeler
train_accuracies = []
test_accuracies = []

# Her n_neighbors değeri için modeli eğitip doğrulukları kaydedin
for n in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=n, weights=random_search.best_params_['weights'],metric=random_search.best_params_['metric'])
    knn.fit(x_train_resampled, y_train_resampled)

    train_accuracy = accuracy_score(y_train_resampled, knn.predict(x_train_resampled))
    test_accuracy = accuracy_score(y_test, knn.predict(x_test))

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# Doğrulukları görselleştir
plt.figure(figsize=(10, 6))
plt.plot(range(1, 31), train_accuracies, label='Eğitim Doğruluğu', marker='o')
plt.plot(range(1, 31), test_accuracies, label='Test Doğruluğu', marker='o')
plt.title('KNN Modeli için Doğruluk Karşılaştırması')
plt.xlabel('Komşu Sayısı (n_neighbors)')
plt.ylabel('Doğruluk')
plt.xticks(range(1, 31))
plt.legend()
plt.grid()
plt.show()

# En iyi modeli kullanarak tahmin yap
y_pred = random_search.predict(x_test)

print("Test seti doğruluğu: ", accuracy_score(y_test, y_pred))
print("Karışıklık Matrisi: ", confusion_matrix(y_test, y_pred))
print("Sınıflandırma Raporu: ", classification_report(y_test, y_pred))

plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('KNN (RandomSearch) Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.show()

y_pred_prob = random_search.predict_proba(x_test)[:, 1]
# ROC eğrisi hesaplama
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# ROC Eğrisini çiz
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label=f"ROC Eğrisi (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Rastgele tahmin çizgisi
plt.title('ROC Eğrisi')
plt.xlabel('Yanlış Pozitif Oranı (FPR)')
plt.ylabel('Doğru Pozitif Oranı (TPR)')
plt.legend(loc='lower right')
plt.grid()
plt.show()