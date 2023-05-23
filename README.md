# Projede Yaklaşık 10.000 deri kanseri görüntüsü üzerinde çalıştım. Derin öğrenme modeli olarak CNN algoritmasını kullandım.
#Projede 7 farklı deri kanseri türü üzerinde çalıştım. Bu deri kanser türleri ve projedeki kısaltılmış isimleri ;
‘v': 'Melanocytic nevi', 
'mel': 'Melanoma', 
'bkl': 'Benign keratosis-like lesions ', 
'bcc': 'Basal cell carcinoma', 
'akiec': 'Actinic keratoses', 
'vasc': 'Vascular lesions‘  
'df': 'Dermatofibroma' .

#PROJE KISIMLARI 
1. Ortam ve kütüphane kurulumları, data frame oluşturup verilere genel bakış
2.Data Preprocessing : deri kanseri görüntülerinin data frame’e sütun olarak eklenmesi
3.Read Picle File : Programın ilk çalışmadan sonra yukarda ki adımlarının debug edilip data frame nesnesine döşütürülmesi.
4.Standardization : İmage dosyası üzerinden modelleme yapılarak target class üzerinden kanser türü sonuçları oluşturulacak. Burada One Hot Encoding yöntemi uygulanarak kanser türleri numaralandırılacak.
5.Derin Öğrenme Algoritması : CNN modellemesi 
6.Layerler Oluşturma : Convolutional,Pooling,Flattenin, Dense ve Output katmanlarından oluşuyor.
7.Compiling the Model : Modelin derlenmesi aşamasıdır. Optimize algoritması olarak Adam kullanılır.
8.Training the Model: Modelin eğitilmesi aşamasıdır. Epoch, batch size gibi değerler test edilerek en doğru sonuca ulaşılmaya çalışılır.
9.Using Our Models to Make Predictions : Yeni veriler ile tahmin yapılarak model geliştirilir.
Son aşama olarak da GUI Skin Cancer Classification : Cilt kanseri sınıflandırılması yapılır ve kullanıcı arayüzü oluşturulur.

Deri kanseri veri seti özellikleri ve görseller üzerinden kullanıcının seçtiği bir görüntünün yüzde kaç ihtimal ile hangi kanser türüne ait olduğu bilgisi Result kısmında kullanıcıya verilir.
En yüksek oranı taşıyan kanser türü işaretlenir. Görsel ile ilgili değerler Features kısmında listelenir.

![image](https://github.com/busraCin/SkinCancerClassification/assets/69642923/1815913c-2088-4b31-a3ef-e96a260cc51b)

# Kütüphaneler ve Teknolojiler
Python
NumPy & Pandas
MatPlotLib  & Seaborn 
Tkinter
Keras
