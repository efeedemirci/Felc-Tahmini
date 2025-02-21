import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Veri setini yükle
df = pd.read_csv('preprocessed_stroke.csv')

# Veri setini eğitim ve test olarak ayır
x_train, x_test, y_train, y_test = train_test_split(df.drop('stroke', axis=1), df['stroke'], test_size=0.2, random_state=42)

# SMOTE ile sınıf dengesizliğini gider
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

# Hiperparametre aralığını tanımla
param_space = {
    'n_estimators': Integer(50, 300),  # Ağaç sayısı
    'max_depth': Integer(5, 30),  # Maksimum derinlik
    'min_samples_split': Integer(2, 10),  # Dallanma için minimum örnek sayısı
    'min_samples_leaf': Integer(1, 10),  # Yaprak düğümü için minimum örnek sayısı
    'criterion': Categorical(['gini', 'entropy'])  # Bölme kriteri
}

# Bayes optimizasyonu
bayes_search_rf = BayesSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    search_spaces=param_space,
    n_iter=50,  # Deneme sayısı
    cv=5,  # Cross-validation katman sayısı
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

# Optimizasyonu yap
bayes_search_rf.fit(x_train_resampled, y_train_resampled)

# En iyi hiperparametreleri yazdır
print("En iyi hiperparametreler: ", bayes_search_rf.best_params_)
print("En iyi doğruluk skoru (Cross-Validation): ", bayes_search_rf.best_score_)

# En iyi model ile test verisi üzerinde değerlendirme
best_rf = bayes_search_rf.best_estimator_

# Test seti doğruluğu
y_pred = best_rf.predict(x_test)
print("Test seti doğruluğu: ", accuracy_score(y_test, y_pred))

# Karışıklık Matrisi ve Sınıflandırma Raporu
print("Karışıklık Matrisi: \n", confusion_matrix(y_test, y_pred))
print("Sınıflandırma Raporu: \n", classification_report(y_test, y_pred))

# Karışıklık Matrisi görselleştirme
plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest (BayesSearchCV) Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.show()

y_pred_prob = best_rf.predict_proba(x_test)[:, 1]
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