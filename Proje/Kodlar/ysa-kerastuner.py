import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner.tuners import RandomSearch
import tensorflow as tf

df = pd.read_csv('preprocessed_stroke.csv')

# Veri setini eğitim ve test olarak ayır
x_train, x_test, y_train, y_test = train_test_split(df.drop('stroke', axis=1), df['stroke'], test_size=0.2,random_state=42)

# SMOTE ile sınıf dengesizliğini gider
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

# YSA için kategorik sınıfları dönüştür
y_train_resampled = to_categorical(y_train_resampled)
y_test = to_categorical(y_test)


# Model oluşturma fonksiyonu
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units_layer1', min_value=32, max_value=256, step=32),
                    activation=hp.Choice('activation_layer1', values=['relu', 'tanh']),
                    input_dim=x_train_resampled.shape[1]))
    model.add(Dropout(rate=hp.Float('dropout_layer1', min_value=0.0, max_value=0.5, step=0.1)))

    # İkinci katman (isteğe bağlı)
    if hp.Boolean('use_second_layer'):
        model.add(Dense(units=hp.Int('units_layer2', min_value=32, max_value=128, step=32),
                        activation=hp.Choice('activation_layer2', values=['relu', 'tanh'])))
        model.add(Dropout(rate=hp.Float('dropout_layer2', min_value=0.0, max_value=0.5, step=0.1)))

    # Çıkış katmanı
    model.add(Dense(units=y_train_resampled.shape[1], activation='softmax'))

    # Modeli derle
    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# Keras Tuner ile hiperparametre araması
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=20,
    executions_per_trial=1,
    directory='my_dir',
    project_name='ysa_hyperparameter_tuning'
)

# Erken durdurma
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Optimizasyon
tuner.search(
    x_train_resampled, y_train_resampled,
    epochs=50,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# En iyi hiperparametreleri al
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("En iyi hiperparametreler:")
print(f"Birinci katman birim sayısı: {best_hps.get('units_layer1')}")
print(f"Birinci katman aktivasyon fonksiyonu: {best_hps.get('activation_layer1')}")
print(f"Birinci katman dropout oranı: {best_hps.get('dropout_layer1')}")
if best_hps.get('use_second_layer'):
    print(f"İkinci katman birim sayısı: {best_hps.get('units_layer2')}")
    print(f"İkinci katman aktivasyon fonksiyonu: {best_hps.get('activation_layer2')}")
    print(f"İkinci katman dropout oranı: {best_hps.get('dropout_layer2')}")
print(f"Optimizer: {best_hps.get('optimizer')}")

# En iyi model ile eğitimi tamamla
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(
    x_train_resampled, y_train_resampled,
    epochs=50,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Test seti değerlendirme
y_pred = best_model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

print("Test seti doğruluğu: ", accuracy_score(y_test_classes, y_pred_classes))
print("Karışıklık Matrisi: \n", confusion_matrix(y_test_classes, y_pred_classes))
print("Sınıflandırma Raporu: \n", classification_report(y_test_classes, y_pred_classes))

# Karışıklık Matrisi görselleştirme
plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test_classes, y_pred_classes), annot=True, fmt='d', cmap='Blues')
plt.title('Yapay Sinir Ağları (Keras Tuner) Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.show()

y_pred_prob = best_model.predict(x_test)[:, 1]
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