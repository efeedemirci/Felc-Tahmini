import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc

df = pd.read_csv('preprocessed_stroke.csv',sep=',')

x = df.drop(columns=['stroke'])
y = df['stroke']

plt.figure(figsize=(6, 4))
sns.countplot(x=y, palette='viridis')
plt.title("SMOTE Öncesi Dağılım")
plt.show()

smote = SMOTE(random_state=None)
X_resampled, y_resampled = smote.fit_resample(x, y)

plt.figure(figsize=(6, 4))
sns.countplot(x=y_resampled, palette='viridis')
plt.title("SMOTE Sonrası Dağılım")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=None, stratify=y_resampled)

param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear', 'saga'],
    'max_iter': [100, 200, 500]
}

log_reg = LogisticRegression()
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("En iyi parametreler:", best_params)

log_reg_best = LogisticRegression(**best_params, random_state=None)
log_reg_best.fit(X_train, y_train)

y_pred = log_reg_best.predict(X_test)
print("Test seti doğruluğu: ", accuracy_score(y_test, y_pred))
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print("ROC-AUC Skoru:", roc_auc_score(y_test, y_pred))

plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.show()

train_acc = accuracy_score(y_train, log_reg_best.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)
print(f"Eğitim Doğruluğu: {train_acc}")
print(f"Test Doğruluğu: {test_acc}")

# Eğitim ve test doğruluklarını saklamak için listeler
train_accuracies = []
test_accuracies = []

# Her n_neighbors değeri için modeli eğitip doğrulukları kaydedin
for n in range(1, 31):
    log_reg = LogisticRegression(penalty=grid_search.best_params_['penalty'], C=grid_search.best_params_['C'], solver=grid_search.best_params_['solver'], max_iter=grid_search.best_params_['max_iter'])
    log_reg.fit(X_train, y_train)

    train_accuracy = accuracy_score(y_train, log_reg.predict(X_train))
    test_accuracy = accuracy_score(y_test, log_reg.predict(X_test))

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# Doğrulukları görselleştir
plt.figure(figsize=(10, 6))
plt.plot(range(1, 31), train_accuracies, label='Eğitim Doğruluğu', marker='o')
plt.plot(range(1, 31), test_accuracies, label='Test Doğruluğu', marker='o')
plt.title('Logistic Regression Modeli için Doğruluk Karşılaştırması')
plt.xlabel('Komşu Sayısı (n_neighbors)')
plt.ylabel('Doğruluk')
plt.xticks(range(1, 31))
plt.legend()
plt.grid()
plt.show()

y_pred_prob = log_reg_best.predict_proba(X_test)[:, 1]
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