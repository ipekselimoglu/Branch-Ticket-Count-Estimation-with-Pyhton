import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from joblib import dump
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

# Pandas ayarlarını yapılandırma
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

path = "C:/Users/ipeks/PycharmProjects/pythonProject/Bitirme/Datasets/siramatik_model_input.csv"
data = pd.read_csv(path)
print(data.columns)
data["PROD_DATE"] = pd.to_datetime(data['PROD_TIME']).dt.date
data.head(15)
groupby_list = ['BRNCH_CODE', 'Year', 'Month', 'Day', 'Hour', 'Day_of_week', 'Season',
                'Time_of_day_PROD', 'PROD_DATE']

group_dataframe = pd.DataFrame(data.groupby(groupby_list).agg({"TICKET_ID": "count",
                                                               "WAITING_PERIOD(second)": "mean"}))

group_dataframe.head(10)
group_dataframe.to_csv("data.csv")

path2 = "C:/Users/ipeks/PycharmProjects/pythonProject/Bitirme/datasets/data.csv"
model_data = pd.read_csv(path2)
model_data.columns = ['BRNCH_CODE', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'DAY_OF_WEEK', 'SEASON',
                      'TIME_OF_DAY_PROD', 'PROD_DATE', 'SUM_TICKET_COUNT', 'WAITING_PERIOD(second)']

model_data.head()

# LabelEncoder nesnesini oluşturma
label_encoder = LabelEncoder()
# Kategorik değişkeni label encoding ile dönüştürün
model_data['BRNCH_CODE'] = label_encoder.fit_transform(model_data['BRNCH_CODE'])
model_data.groupby('BRNCH_CODE').count()

# Kategorik değişkeni label encoding ile dönüştürme
encoded_labels = label_encoder.fit_transform(model_data['BRNCH_CODE'])
unique_labels = model_data['BRNCH_CODE'].unique()

# Label değerlerini içeren bir DataFrame oluşturma
label_df = pd.DataFrame({
    'BRNCH_CODE': unique_labels,
    'Encoded_Label': encoded_labels
})

print(label_df)
print("\nAtama Bilgisi:")
for label, code in zip(unique_labels, label_encoder.classes_):
    print(f"'{label}' --> {code}")


def label_encode_categorical_columns(data, columns):
    """
    Verilen veri setinde belirtilen kategorik sütunları etiket kodlayarak dönüştürür.

    Args:
        data (pd.DataFrame): Veri seti
        columns (list): Etiket kodlanacak sütun adlarının listesi

    Returns:
        pd.DataFrame: Etiket kodlanmış veri seti
    """
    # LabelEncoder nesnesini oluşturun
    label_encoder = LabelEncoder()

    # Her bir sütun için etiket kodlama işlemini uygulayın
    for column in columns:
        data[column] = label_encoder.fit_transform(data[column])

    return data


# Fonksiyonu kullanarak kategorik sütunları etiket kodlayarak dönüştürme
columns_to_encode = ['DAY_OF_WEEK', 'SEASON', 'TIME_OF_DAY_PROD']
model_data = label_encode_categorical_columns(model_data, columns_to_encode)
print(model_data.head())

#####################################################################################
# MODELE HAZIRLIK
model_data2 = model_data.copy()
model_data = model_data2
model_data = model_data[model_data["BRNCH_CODE"] == 6]
model_data.head(15).to_excel("dddd.xlsx")

# Bağımsız ve bağımlı değişkenleri seçin
X = model_data.drop(['BRNCH_CODE', 'YEAR', 'SUM_TICKET_COUNT', 'PROD_DATE'],
                    axis=1)  # Bağımsız değişkenler
y = model_data['SUM_TICKET_COUNT']  # Bağımlı değişken

# Veriyi eğitim ve test setlerine bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#######################################################################################3
# RANDOM FOREST REGREESSOR İLE MODEL
def train_evaluate_rf_model(X_train, X_test, y_train, y_test, param_grid):
    """
    Random Forest Regressor modelini eğitir ve performansını değerlendirir.

    Args:
        X_train, X_test: Eğitim ve test veri setleri (bağımsız değişkenler)
        y_train, y_test: Eğitim ve test veri setleri (bağımlı değişkenler)
        param_grid: Model parametre aralığı

    Returns:
        RandomForestRegressor: En iyi model
        dict: Model performansını içeren sözlük
        stratified_kfold
    """
    # Grid Search ile en iyi parametreleri bulma
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # En iyi modeli seçme
    best_rf_model = grid_search.best_estimator_

    # En iyi modelle eğitim yapma
    best_rf_model.fit(X_train, y_train)

    # Model performansını değerlendirme
    rf_train_predictions = best_rf_model.predict(X_train)
    rf_test_predictions = best_rf_model.predict(X_test).round()

    rf_train_mse = mean_squared_error(y_train, rf_train_predictions)
    rf_test_mse = mean_squared_error(y_test, rf_test_predictions)

    rf_train_r2 = r2_score(y_train, rf_train_predictions)
    rf_test_r2 = r2_score(y_test, rf_test_predictions)

    rf_train_rmse = np.sqrt(rf_train_mse)
    rf_test_rmse = np.sqrt(rf_test_mse)

    rf_performance = {
        'Train MSE': rf_train_mse,
        'Test MSE': rf_test_mse,
        'Train R^2': rf_train_r2,
        'Test R^2': rf_test_r2,
        'Train RMSE': rf_train_rmse,
        'Test RMSE': rf_test_rmse
    }

    print("Random Forest Regressor Performance:")
    for metric, value in rf_performance.items():
        print(f"{metric}: {value}")

    return best_rf_model, rf_performance


# Random Forest Regresyon modeli için uygun parametrelerin aralığı
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Fonksiyonu kullanarak modeli eğitme ve performansı değerlendirme
best_rf_model, rf_performance = train_evaluate_rf_model(X_train, X_test, y_train, y_test, param_grid)

# Eğitim ve test setleri için tahmin edilen değerler
train_predictions = best_rf_model.predict(X_train).round()
test_predictions = best_rf_model.predict(X_test).round()

# Gerçek ve tahmin edilen değerleri bir veri çerçevesinde birleştirme
train_results = pd.DataFrame({'Actual': y_train, 'Predicted': train_predictions})
test_results = pd.DataFrame({'Actual': y_test, 'Predicted': test_predictions})
X_train["predict"] = train_results["Predicted"]
X_train["actual"] = train_results["Actual"]
X_train.head()
# Eğitim seti için gerçek ve tahmin edilen değerleri yazdırma
print("Eğitim Seti:")
print(train_results.head(20))

# Test seti için gerçek ve tahmin edilen değerleri yazdırma
print("\nTest Seti:")
print(test_results.head(20))

# Tahmin edilen ve gerçek değerler arasındaki farkı hesaplayan yeni bir sütun ekleme
X_train['Difference'] = abs(test_results['Predicted'] - test_results['Actual'])

# Gerçek ve tahmin edilen değerlerin grafiğini çizme
plt.figure(figsize=(10, 6))

# Eğitim seti için grafiği çizme
plt.subplot(1, 2, 1)
plt.scatter(y_train, train_predictions, color='blue', label='Gerçek vs Tahmin (Eğitim Seti)')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Gerçek vs Tahmin (Eğitim Seti)')
plt.legend()

# Test seti için grafiği çizme
plt.subplot(1, 2, 2)
plt.scatter(y_test, test_predictions, color='red', label='Gerçek vs Tahmin (Test Seti)')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Gerçek vs Tahmin (Test Seti)')
plt.legend()

plt.tight_layout()
plt.show()


def feature_importance(model, X):
    """
    Verilen modeldeki değişkenlerin önem sırasını yazdırır.

    Args:
        model: Eğitilmiş model
        X: Bağımsız değişkenlerin olduğu veri seti

    Returns:
        None
    """
    # Modeldeki değişkenlerin önem sırası
    feature_importance = model.feature_importances_

    # Önem sırasına göre değişkenlerin indeksleri
    sorted_indices = np.argsort(feature_importance)[::-1]

    # Önem sırasına göre değişken adları
    sorted_features = X.columns[sorted_indices]

    # Önem sırasına göre değişkenleri ve önemleri
    print("Değişkenlerin Önem Sırası:")
    for i, feature in enumerate(sorted_features):
        print(f"{i + 1}. {feature}: {feature_importance[sorted_indices[i]]}")


# Fonksiyonu çağırarak değişkenlerin önem sırası
feature_importance(best_rf_model, X)


def calculate_and_plot_correlation_matrix(X):
    """
    Verilen veri setindeki değişkenler arasındaki korelasyon matrisini hesaplar ve görselleştirir.

    Args:
        X: Değişkenlerin olduğu veri seti

    Returns:
        None
    """
    # Değişkenler arasındaki korelasyon matrisini hesaplama
    correlation_matrix = X.corr()

    # Korelasyon matrisinin görselleştirilmesi
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Değişkenler Arasındaki Korelasyon Matrisi')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()


# Korelasyon matrisini hesaplayıp görselleştirme
calculate_and_plot_correlation_matrix(X)


def model_complexity(model):
    """
    Model karmaşıklığını değerlendirir.

    Args:
        model: Eğitilmiş bir makine öğrenimi modeli

    Returns:
        complexity: Model karmaşıklığı değeri
    """
    complexity = 0

    # Decision Trees sayısını al
    if hasattr(model, 'n_estimators'):
        complexity += model.n_estimators

    # Max depth'ı al
    if hasattr(model, 'max_depth'):
        complexity += model.max_depth if model.max_depth is not None else 0

    # Min samples split ve min samples leaf'i al
    if hasattr(model, 'min_samples_split'):
        complexity += model.min_samples_split

    if hasattr(model, 'min_samples_leaf'):
        complexity += model.min_samples_leaf

    return complexity


# Eğitilmiş en iyi Random Forest modeli için karmaşıklığı değerlendirme
rf_complexity = model_complexity(best_rf_model)
print("Random Forest Model Karmaşıklığı:", rf_complexity)

# Modeli kaydetme
dump(best_rf_model, 'random_forest_model.joblib')

##########################################################################
# LİNEAR REGRESYON İLE MODEL

# Doğrusal regresyon modelini oluşturma
model = LinearRegression()

# Modeli eğitme
model.fit(X_train, y_train)


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Verilen modelin performansını değerlendirir.

    Args:
        model: Eğitilmiş model
        X_train, X_test: Eğitim ve test veri setleri (bağımsız değişkenler)
        y_train, y_test: Eğitim ve test veri setleri (bağımlı değişkenler)

    Returns:
        dict: Model performansını içeren sözlük
    """
    # Eğitim setinde modelin tahminleri
    train_predictions = model.predict(X_train)

    # Test setinde modelin tahminleri
    test_predictions = model.predict(X_test)

    # Eğitim setinde ortalama karesel hata (MSE)
    train_mse = mean_squared_error(y_train, train_predictions)

    # Test setinde ortalama karesel hata (MSE)
    test_mse = mean_squared_error(y_test, test_predictions)

    # Eğitim setinde R-kare (R-squared) değerini
    train_r2 = r2_score(y_train, train_predictions)

    # Test setinde R-kare (R-squared) değerini
    test_r2 = r2_score(y_test, test_predictions)

    # Eğitim setinde kök ortalama karesel hata (RMSE)
    train_rmse = np.sqrt(train_mse)

    # Test setinde kök ortalama karesel hata (RMSE)
    test_rmse = np.sqrt(test_mse)

    # Model performansını içeren sözlüğü
    performance = {
        'Train MSE': train_mse,
        'Test MSE': test_mse,
        'Train R^2': train_r2,
        'Test R^2': test_r2,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse
    }

    return performance


# Modelin performansını
model_performance = evaluate_model(model, X_train, X_test, y_train, y_test)
print(model_performance)

##################################################################
# ARIMA

# Veriyi zaman serisi olarak düzenleme
arima_data = pd.DataFrame(data.groupby("PROD_DATE").agg({"TICKET_ID": "count"}))
arima_data.columns = ["SUM_TICKET_COUNT"]
arima_data.index = pd.to_datetime(arima_data.index)  # 'PROD_DATE' sütununu datetime nesnelerine dönüştürür
arima_data.sort_index(inplace=True)  # Zaman serisini tarihe göre sıralama


# ARIMA modelini kurma ve tahminler yapma fonksiyonu
def fit_arima_model(data, order=(5, 1, 0), start=1, end=10):
    # ARIMA modelini kurma
    model = ARIMA(data, order=order)
    fit_model = model.fit()

    # Tahminler yapma
    predictions = fit_model.predict(start=start, end=end, dynamic=False)

    return predictions, fit_model


# Modelin performansını değerlendirme fonksiyonu
def evaluate_model(predictions, actual_values):
    # Hata metriklerini hesaplama
    mse = mean_squared_error(actual_values, predictions)
    r2 = r2_score(actual_values, predictions)

    return mse, r2


# ARIMA modelini kurma ve tahminler yapma
predictions, fit_model = fit_arima_model(arima_data['SUM_TICKET_COUNT'])

# Tahminlerin ve gerçek değerlerin bir veri çerçevesine eklenmesi
results = pd.DataFrame({'Tahmin': predictions, 'Gerçek': arima_data['SUM_TICKET_COUNT'].iloc[:len(predictions)]})

# Modelin performansını değerlendirme
mse, r2 = evaluate_model(predictions, arima_data['SUM_TICKET_COUNT'].iloc[:len(predictions)])
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Tahminleri ve gerçek değerleri görselleştirme
plt.plot(arima_data.index[:len(predictions)], predictions, label='Tahmin')
plt.plot(arima_data.index[:len(predictions)], arima_data['SUM_TICKET_COUNT'].iloc[:len(predictions)], label='Gerçek')
plt.xlabel('Tarih')
plt.ylabel('Tahmin Edilen ve Gerçek Değerler')
plt.title('ARIMA Model Tahminleri')
plt.legend()
plt.show()
###############################################################################
# XGBoost MODELI
xgb_model = XGBRegressor()

# Cross-validation ile model performansını değerlendirME
cv_scores_mse = cross_val_score(xgb_model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_scores_r2 = cross_val_score(xgb_model, X, y, cv=5, scoring='r2')

# Cross-validation sonuçlarını ekrana yazdırma
print("Cross-Validation MSE Scores:", -cv_scores_mse)  # Negatif MSE değerleri olduğu için işaretini değiştiriyoruz
print("Cross-Validation R^2 Scores:", cv_scores_r2)

# Cross-validation ile elde edilen MSE'nin ortalamasını alarak genel model performansını
print("Mean Cross-Validation MSE:", -cv_scores_mse.mean())

# Cross-validation ile elde edilen R^2'nin ortalamasını alarak genel model performansını
print("Mean Cross-Validation R^2:", cv_scores_r2.mean())
