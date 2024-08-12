import numpy as np
import pandas as pd
import missingno as msno
from matplotlib import pyplot as plt

# Pandas ayarlarını yapılandırma
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


# Veri kümesini yükleme işlevi
def load_dataset():
    """
    Veri kümesini bir Excel dosyasından yükler.
    """
    path = "C:/Users/ipeks/PycharmProjects/pythonProject/Bitirme/Datasets/Proje1_copy.xlsx"
    data = pd.read_excel(path)
    data.columns = ["BRNCH_CODE", "TICKET_ID", "PROD_TIME", "CALL_TIME", "FINISH_TIME", "ERROR"]
    return data


# Veri kümesini yükleme
df_ = load_dataset()
df = df_.copy()
df.shape
df.head()
df.tail()


def data_info(dataframe):
    # Veri kümesinin başlangıç bilgilerini görüntüleme
    print("Veri kümesinin başlangıç bilgileri:")
    print(dataframe.head())  # İlk beş satır
    print("Veri kümesinin şekli:", dataframe.shape)  # Veri kümesinin şekli
    print(dataframe.describe().T)  # Temel istatistiksel bilgiler
    print(dataframe.info())  # Veri kümesi hakkında bilgiler

    # Eksik değer bilgilerini görüntüleme
    print("Eksik değer bilgileri:")
    print("Eksik değer sayısı:\n", dataframe.isnull().sum())  # Eksik değerlerin sayısı
    print("Eksik olmayan değer sayısı:\n", dataframe.notnull().sum())  # Eksik olmayan değerlerin sayısı
    print("Herhangi bir eksik değer var mı?", dataframe.isnull().values.any())  # Herhangi bir eksik değer var mı?
    print("Tüm eksik değerleri içermeyen satırların sayısı:\n",
          dataframe[dataframe.notnull().all(axis=1)].shape)  # Tüm eksik değerleri içermeyen satırların sayısı
    print(dataframe.isnull().sum().sort_values(ascending=False))  # Eksik değerleri sırala


# Fonksiyonu çağırarak veri kümesi bilgilerini görüntüleme
data_info(df)


# Eksik değer analizi
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


# Eksik değer tablosunu oluşturma ve gösterme
missing_values_table(df)
missing_values_table(df, True)

# Eksik değer görselleştirmesi
msno.bar(df)
plt.show()

# null değerlerin silinmesi
df.shape
df.dropna(inplace=True)
df.shape
data_info(df)


def check_non_numeric_values_in_column(data, column_name):
    """
    Belirli bir sütunda sayı formatına uymayan değerleri kontrol eder.

    Args:
        data (DataFrame): Veri kümesi
        column_name (str): Kontrol edilecek sütun adı

    Returns:
        DataFrame: Sayı formatına uymayan değerler içeren DataFrame
    """
    non_numeric_values = data[~data[column_name].astype(str).str.match(r'^\d+$')]
    return non_numeric_values


def remove_non_numeric_values_in_column(data, column_name):
    """
    Belirli bir sütundaki sayı formatına uymayan değerleri kaldırır.

    Args:
        data (DataFrame): Veri kümesi
        column_name (str): Kaldırılacak sütun adı

    Returns:
        DataFrame: Güncellenmiş veri kümesi
        int: Silinen kayıt sayısı
    """
    original_length = len(data)
    # Sayı formatına uymayan satırları filtreleme işlemi
    data = data[data[column_name].apply(lambda x: str(x).isdigit())]
    # Silinen kayıt sayısını hesapla
    removed_count = original_length - len(data)
    return data, removed_count


# 'TICKET_ID' sütununda sayı formatına uymayan değerleri kontrol etme
non_numeric_values = check_non_numeric_values_in_column(df, 'TICKET_ID')

if non_numeric_values.empty:
    print("Belirtilen sütunun tüm değerleri sayı formatına uyuyor.")
else:
    print("Sayı formatına uymayan değerler var:")
    print(non_numeric_values)

# Sayı formatına uymayan değerleri 'TICKET_ID' sütunundan kaldırdık
# Güncellenen fonksiyon çağrıldı ve kaç kayıt silindiği yazdırdık
df, removed_count = remove_non_numeric_values_in_column(df, 'TICKET_ID')
print("\nSayı formatına uymayan değerler silindi. Toplam", removed_count, "kayıt silindi.")
df.shape


##############################
def remove_invalid_timestamps_in_columns(data, column_indices):
    """
    Belirli sütunlardaki datetime formatına uymayan değerleri kaldırır.

    Args:
        data (DataFrame): Veri kümesi
        column_indices (list of int): Kaldırılacak sütunların indeksleri

    Returns:
        DataFrame: Güncellenmiş veri kümesi
        dict: Sütun indeksi ve silinen kayıt sayısı içeren sözlük
    """
    original_length = len(data)
    removed_counts = {}  # Sütun indeksi ve silinen kayıt sayısı
    for column_index in column_indices:
        column_name = data.columns[column_index]
        # Verilen sütundaki değerlerin timestamp formatına uygun olup olmadığını kontrol eder
        data.loc[:, column_name] = pd.to_datetime(data[column_name], errors='coerce',
                                                  format='%Y-%m-%d %H:%M:%S')  # Örnek format belirtilmiştir, gerektiğinde değiştirilebilir
        # Geçerli datetime değerlerini içeren satırları filtreler
        valid_rows = data[column_name].notna()
        # Geçersiz timestamp değerlerini içeren satırları kaldırır
        data = data[valid_rows]
        # Silinen kayıt sayısını hesaplar ve sözlüğe ekler
        removed_counts[column_index] = original_length - len(data)
    return data, removed_counts


df.columns
df, removed_counts = remove_invalid_timestamps_in_columns(df, [2, 3, 4])  # İlgili sütunların indeksleri
print("Silinen kayıt sayıları:", removed_counts)
df.shape
df.describe().T


def min_max_time(df):
    prod_time_min = df["PROD_TIME"].min()
    prod_time_max = df["PROD_TIME"].max()
    call_time_min = df["CALL_TIME"].min()
    call_time_max = df["CALL_TIME"].max()
    finish_time_min = df["FINISH_TIME"].min()
    finish_time_max = df["FINISH_TIME"].max()
    return prod_time_min, prod_time_max, call_time_min, call_time_max, finish_time_min, finish_time_max


prod_time_min, prod_time_max, min_call_time, max_call_time, min_finish_time, max_finish_time = min_max_time(df)
print("PROD_TIME Min:", prod_time_min)
print("PROD_TIME Max:", prod_time_max)
print("CALL_TIME Min:", min_call_time)
print("CALL_TIME Max:", max_call_time)
print("FINISH_TIME Min:", min_finish_time)
print("FINISH_TIME Max:", max_finish_time)

################
# PROD_TIME sütununu datetime formatına dönüştürme.
df["PROD_TIME"] = pd.to_datetime(df["PROD_TIME"])
df["CALL_TIME"] = pd.to_datetime(df["CALL_TIME"])
df["FINISH_TIME"] = pd.to_datetime(df["FINISH_TIME"])

# mesai saati içndeki biletler alındı
df = df[((df['PROD_TIME'].dt.time >= pd.to_datetime('09:00').time()) &
         (df['PROD_TIME'].dt.time <= pd.to_datetime('12:30').time())) |
        ((df['PROD_TIME'].dt.time >= pd.to_datetime('13:30').time()) &
         (df['PROD_TIME'].dt.time <= pd.to_datetime('17:00').time()))]

print(df[df["FINISH_TIME"] < df[
    "CALL_TIME"]].shape)  # finish time call time dan her zaman büyük olmalı, aksi bir kayıt yok
print(
    df[df["CALL_TIME"] < df["PROD_TIME"]].shape)  # call time finish time dan her zaman büyük olmalı, aksi bir kayıt yok


def extract_date_info(dataframe, datetime_column):
    # DataFrame'in bir kopyasını oluşturma
    df_copy = dataframe.copy()

    # 'datetime_column' adında bir sütunun olduğunu varsayalım ve datetime türüne dönüştürme
    df_copy[datetime_column] = pd.to_datetime(df_copy[datetime_column])

    # Yıl, Ay, Gün, Saat ve Haftanın Günü bilgilerini çıkaralım
    df_copy['Year'] = df_copy[datetime_column].dt.year
    df_copy['Month'] = df_copy[datetime_column].dt.month
    df_copy['Day'] = df_copy[datetime_column].dt.day
    df_copy['Hour'] = df_copy[datetime_column].dt.hour
    df_copy['Day_of_week'] = df_copy[datetime_column].dt.day_name()
    df_copy['Season'] = df_copy[datetime_column].dt.month.apply(lambda x: 'Winter' if x in [12, 1, 2] else (
        'Spring' if x in [3, 4, 5] else ('Summer' if x in [6, 7, 8] else 'Fall')))

    return df_copy


# Fonksiyonu çağırarak DataFrame'i güncelleme
df = extract_date_info(df, 'PROD_TIME')
df.head()

# İNCELEMELER
years = df.groupby("Year").count()
years.head()
df.groupby("Hour")["TICKET_ID"].count()
df.groupby(["BRNCH_CODE", "Hour"])["TICKET_ID"].count()

#
df.loc[:, "WAITING_PERIOD(second)"] = (df["CALL_TIME"] - df["PROD_TIME"]).dt.total_seconds()
df.loc[:, "PROCESS_PERIOD(second)"] = (df["FINISH_TIME"] - df["CALL_TIME"]).dt.total_seconds()

df.describe().T
df2 = df.copy()
df2.head()


# OUTLİER ANALİZİ
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    print(low_limit, up_limit)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def outlier_thresholds(dataframe, col_name, q1=0.15, q3=0.85):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def remove_outliers(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_cleaned = dataframe[(dataframe[col_name] >= low_limit) & (dataframe[col_name] <= up_limit)].copy()
    return df_cleaned


print(df2.shape)
check_outlier(df2, "WAITING_PERIOD(second)")
outlier_thresholds(df2, "WAITING_PERIOD(second)")
df3 = remove_outliers(df2, "WAITING_PERIOD(second)")
print(df3.shape)

plt.figure(figsize=(8, 6))
plt.boxplot(df3[df3['BRNCH_CODE'] == "EEEE"]['WAITING_PERIOD(second)'], vert=False,
            flierprops=dict(marker='o', markerfacecolor='r', markersize=8, linestyle='none'))
plt.xlabel('WAITING_PERIOD(second)')
plt.title('WAITING_PERIOD(second) Boxplot')
plt.show()

df3 = df3[df3['WAITING_PERIOD(second)'] <= 1000]
df3.describe()

print(df3.shape)
check_outlier(df3, "PROCESS_PERIOD(second)")
outlier_thresholds(df3, "PROCESS_PERIOD(second)")
df4 = remove_outliers(df3, "PROCESS_PERIOD(second)")
print(df4.shape)
df4.describe()

plt.figure(figsize=(8, 6))
plt.boxplot(df4[df4['BRNCH_CODE'] == "EEEE"]['PROCESS_PERIOD(second)'], vert=False,
            flierprops=dict(marker='o', markerfacecolor='r', markersize=8, linestyle='none'))
plt.xlabel('PROCESS_PERIOD(second)')
plt.title('PROCESS_PERIOD(second) Boxplot')
plt.show()

print(df4.shape)
df4.describe().T
df4.head()

df4 = df4[df4["WAITING_PERIOD(second)"] > 0]  # bekleme süresi 0 sn olamaz
df4 = df4[df4["PROCESS_PERIOD(second)"] > 20]  # bekleme süresi 20sn'den az olamaz
df4.describe().T
print(df4.shape)


def label_morning_afternoon(dataframe, datetime_column):
    # 'datetime_column' adında bir sütunun olduğunu varsayalım
    # Önce bu sütunu datetime türüne dönüştürelim
    dataframe[datetime_column] = pd.to_datetime(dataframe[datetime_column])

    # Öğleden önce olan saatleri etiketleyelim
    dataframe['Time_of_day_PROD'] = 'Morning'
    dataframe.loc[(dataframe[datetime_column].dt.hour >= 13), 'Time_of_day_PROD'] = 'Afternoon'

    return dataframe


# Fonksiyonu çağırarak DataFrame'i güncelleme
df4 = label_morning_afternoon(df4, 'PROD_TIME')
df4.head()
df4.groupby("Time_of_day_PROD").count()

df4.to_csv("siramatik_model_input.csv", index=False)
