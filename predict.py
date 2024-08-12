import pandas as pd
from joblib import load

# Modeli yükleme
loaded_model = load('random_forest_model.joblib')

""""
örnek_input: 
brnch_code :6
tarih : 2023-03-29 09-00-00
actual değeri 12
"""


def get_input_features():
    """
    Kullanıcıdan modelin bağımsız değişkenlerini input olarak alan bir fonksiyon.

    Returns:
        DataFrame: Kullanıcı girişlerine göre oluşturulan DataFrame
    """
    while True:
        print("Lütfen aşağıdaki değerleri giriniz:")

        # Kullanıcıdan giriş alınması
        branch_code = input("Şube kodunu giriniz: ")
        prod_date_input = input("Üretim tarihini (YYYY-MM-DD HH:MM:SS) giriniz: ")

        try:
            # Giriş değerlerini uygun formata dönüştürme
            prod_date = pd.to_datetime(prod_date_input)
            day_of_week = prod_date.day_name()
            month = prod_date.month

            # Gün ismini kategorik bir değişken olarak kodlamak için kullanılacak sözlük

            day_of_week_dict = {
                'Monday': 1,
                'Tuesday': 3,
                'Wednesday': 4,
                'Thursday': 2,
                'Friday': 0,
                'Saturday': 6,
                'Sunday': 7
            }

            # Mevsim adlarını mevsim kodlarına dönüştüren sözlük
            season_dict = {
                'Winter': 2,
                'Spring': 0,
                'Summer': 1,
                'Fall': 3
            }

            # Giriş değerlerini modele uygun formata dönüştürme
            input_data = pd.DataFrame({
                'MONTH': [month],
                'DAY': [prod_date.day],
                'HOUR': [prod_date.hour],
                'DAY_OF_WEEK': [day_of_week_dict[day_of_week]],  # Gün ismini kategorik değişkene çevirme
                'SEASON': [season_dict['Winter' if month in [12, 1, 2] else (
                    'Spring' if month in [3, 4, 5] else (
                        'Summer' if month in [6, 7, 8] else 'Fall'))]],
                'TIME_OF_DAY_PROD': [prod_date.hour],
                'WAITING_PERIOD(second)': [250]  # ORTALAMA bekleme süresi
            })

            return input_data
        except ValueError:
            print(
                "Hatalı formatta giriş yaptınız. Lütfen tarihi ve saati doğru formatta giriniz (YYYY-MM-DD HH:MM:SS) ve tekrar deneyin.")


# Kullanıcıdan giriş alarak DataFrame'i elde etme
input_features = get_input_features()
print(input_features)

# Tahmin yapma
prediction = loaded_model.predict(input_features)

print("Tahmin edilen bilet sayısı:", prediction[0].round())
