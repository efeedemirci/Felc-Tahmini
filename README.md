Bu projede felç riskini tahmin edebilecek bir makine öğrenmesi modeli geliştirdim. Projede iki ayrı veri seti birleştirildi ve şu adımlar izlendi:

- **Veri ön işleme:** Label encoding, IQR yöntemi, Cramér's V korelasyonu ile anlamsız/sapmalı veriler elendi.
- **Modelleme:** Logistic Regression, XGBoost ve Random Forest gibi algoritmalar kullanıldı.
- **Optimizasyon:** BayesSearchCV, GridSearchCV ve RandomSearch ile hiperparametreler optimize edildi.

Proje, hem veri temizliği hem de model performansı artırma açısından bütüncül bir yaklaşım sergilemektedir.
