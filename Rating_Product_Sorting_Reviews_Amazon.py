import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)  # max genişlik 500 karakter
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("datasets/(ODEV)amazon_review.csv")
df.head()
df.shape
df.info()

# Adım 1: Ürünün ortalama puanını hesaplayınız.
df["overall"].mean()  # 4.587589013224822

# Adım 2: Tarihe göre ağırlıklı puan ortalamasını hesaplayınız
df["reviewTime"] = pd.to_datetime(df["reviewTime"])  # reviewTime değişkenini tarih değişkeni olarak tanıtmanız
df["reviewTime"].max()  # ->> 2014-12-07
current_date = pd.to_datetime('2014-12-07 00:00:00')  # reviewTime'ın max değerini current_date olarak kabul etmeniz
df["days"] = (current_date - df["reviewTime"]).dt.days

# burada önce LABEL yazmadan sadece q=4 diyerek uyguladık. sonrasında label'ları ekledik ki isimlendirme daha net olsun
df["days_quantile"] = pd.qcut(df["days"], q=4, labels=["1-[0_280]", "2-[281_430]", "3-[431_600]",
                                                       "4-[431_600]"])  # gün cinsinden ifade edilen değişkeni quantile fonksiyonu ile 4'e bölüp (3 çeyrek verilirse 4 parça çıkar) çeyrekliklerden gelen değerlere göre ağırlıklandırma yapmanız gerekir

df.loc[df["days"] <= 280, "overall"].mean() * 42 / 100 + \
df.loc[(df["days"] > 280) & (df["days"] <= 430), "overall"].mean() * 27 / 100 + \
df.loc[(df["days"] > 430) & (df["days"] <= 600), "overall"].mean() * 16 / 100 + \
df.loc[(df["days"] > 600), "overall"].mean() * 15 / 100
# 4.622394890703036  genel ortalama

# Adım 3: Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.
# (4.6957928802588995, 4.636140637775961, 4.571661237785016, 4.4462540716612375) her çeyreğin ortalaması ---> güncel zamana yaklaştıkça ortalama artmış

##################################################################################################
# Görev 2: Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyiniz.
##################################################################################################

# Adım 1: helpful_no değişkenini üretiniz.
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head()


# df.loc[df["helpful_no"]>=5, "helpful_no"].sort_values(ascending=False) --> örnek olsun diye en faydasız bulunanları filtreledim.

# Adım 2: score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayıp veriye ekleyiniz.
# score_pos_neg_diff

def score_up_down_diff(up, down):
    return up - down


def score_average_rating(up, down):  # girilen değer toplamı 0 ise
    if up + down == 0:
        return 0
    return up / (up + down)


def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


# score_pos_neg_diff
df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)
# score_average_rating
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)
# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df.sort_values("wilson_lower_bound", ascending=False).head(50)
