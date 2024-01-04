import cv2
import numpy as np
import pickle


dosya = "rfc_model.rfc"
rfc = pickle.load(open(dosya,"rb"))     #   şifrelediğimiz dosyayı okuyoruz

            #   bu yapıda sınıfları tanımladık. karakter seti içindeki klasörleri bi nevi kısayol haline getirdik.
sinifs = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10,
          'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20,
          'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30,
          'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35, 'arkaplan': 36}

index = list(sinifs.values())           #   sinifs in valulerini aldık
siniflar = list(sinifs.keys())          #   anahtarları aldık

def islem(img):
    
    yeni_boy = img.reshape((1600,5,5))          #   5e5 1600 tane parça oluşturduk
    orts = []                                   #   her bir parçanın ortalamasını almak için değişken oluşturduk
    for parca in yeni_boy:                      #   her bir parçayı tek tek inceliyoruz
        ort = np.mean(parca)                    #   parçaların ortalamasını aldık
        orts.append(ort)                        #   aldığımız değeri orts a atadık
    orts = np.array(orts)       
    orts = orts.reshape(1600,)
    return orts


def plakaAyristir(mevcutPlaka):
    mevcutPlaka = sorted(mevcutPlaka,key=lambda x:x[1])
    mevcutPlaka = np.array(mevcutPlaka)
    mevcutPlaka = mevcutPlaka[:,0]
    mevcutPlaka = mevcutPlaka.tolist()

    # plaka bir sayı ile başlamalı
    # plakanin basında en fazla 2 rakam bulunabilir
    karakterAdim = 0
    i = 0

    while i < len(mevcutPlaka):
        try:
            int(mevcutPlaka[i])
            karakterAdim += 1
        except ValueError:
            if karakterAdim > 0:
                if i - 2 >= 0:
                    mevcutPlaka = mevcutPlaka[i - 2:]
                else:
                    mevcutPlaka = mevcutPlaka[:i]
                break
            mevcutPlaka.pop(i)
        else:
            i += 1

        

    # plaka bir sayi ile bitmeli
    # plakanın sonunda en fazla 4 rakam olabilir
    karakterAdim=0
    for i in range(len(mevcutPlaka)):
        kontrolIndex = -1 + (-1*karakterAdim)
        try:
            int(mevcutPlaka[kontrolIndex])
            karakterAdim+=1
        except:
            if karakterAdim>0:
                karIndex = len(mevcutPlaka)-karakterAdim
                print("karkter:",mevcutPlaka[karIndex])
                mevcutPlaka = mevcutPlaka[:karIndex+4]
                break
            mevcutPlaka.pop(kontrolIndex)
    
    return mevcutPlaka


def plakaTani(img,plaka):
    if plaka is not None:
        global index,siniflar
        
        plaka += (0,) * (4 - len(plaka))
        x,y,w,h = plaka
        
        print("Plaka değikeninden gelen konum",plaka)
        
        if x is None: x=1
        elif y is None: y=1
        elif w is None: w=1
        elif h is None: h=1
        else: pass
        
        if(w>h):                                                        #   sadece plakayı alacak şekilde resmi kesiyoruzki daha iyi çalışalım
            plaka_bgr = img[y:y+h, x:x+w].copy()
        elif (w == h):
            return img, plaka
        else:
            plaka_bgr = img[y:y+w, x:x+h].copy()
            
            #   plaka değişkeninden gelen veriyi 


        H,W = plaka_bgr.shape[:2]

        plaka_resim = cv2.cvtColor(plaka_bgr, cv2.COLOR_BGR2GRAY)       #   daha iyi çalışmak için resmi siyah beyaz yaptık

        plaka_resim = cv2.adaptiveThreshold(plaka_resim, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        #   eşeikleme yapıyoruz(resim,  eşiğin üstünde kalan değerlerin alacağı değer, eşikleme algoritması,    ayrıştırma tipi, filtrenin değeri, komşu sayısı)

        kernel = np.ones((3,3), np.uint8)                                               #   kernel oluşturuyoruz
        th_img = cv2.morphologyEx(plaka_resim, cv2.MORPH_OPEN, kernel, iterations=1)    #   gürültü yok ediyoruz.



        cnt = cv2.findContours(th_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)      #   counturları buldurduk konumunu istedik
        cnt = cnt[0]            #   sadece x,y,w,h değerlerini istedik
        cnt = sorted(cnt, key=cv2.contourArea, reverse=True)[:15]  

        yaz = plaka_bgr.copy()     
        mevcutPlaka = []
        for i,c in enumerate(cnt):                      
            rect = cv2.minAreaRect(c)
            (x,y),(w,h),r =rect
            
            kon1 = max([w,h]) < W/4         #   yukarda aldığımız w ve h değerlerinden hangisinin büyük olduğunu aldık ve bu değer ana resmimizin 4 te 1 inden küçük mü diye sorduk
            kon2 = w*h > 200                #   bulduğumuz alan 200 den büyük mü
            
            if kon1 and kon2:
                
                box=cv2.boxPoints(rect)     #   cv2.boxPoints() sol üst, sağ üst gibi değerleri r değişkeniyle hesaplayıp bize verir
                box=np.int64(box)           #   ondalık ifadeleri int64 e dönüştürdük
                
                minx = np.min(box[:,0])     #   bütün x lerden en küçük değeri aldık
                miny = np.min(box[:,1])     #   bütün y lerden en küçük değer
                maxx = np.max(box[:,0])     #   en büyük x değeri
                maxy = np.max(box[:,1])     #   en büyük y değeri
                
                odak = 2                    #   0 a 0 almaması için odak adında değer tanımladık

                minx = max(0, minx - odak)  #   eksi değer almamsı için 0 a eşitledik 
                miny = max(0, miny - odak)
                maxx = min(W, maxx + odak)  #   fazla değer almaması için w ve h değerine eşitliyoruz
                maxy = min(H, maxy + odak)

                kesim = plaka_bgr[miny:maxy, minx:maxx].copy()      #   resmi miny den maxy ye kadar ve minx ten maxx e kadar kestik ve kopyaladık

                tani = cv2.cvtColor(kesim,cv2.COLOR_BGR2GRAY)
                tani = cv2.resize(tani,(200,200))
                tani = tani/255
                oznitelikler = islem(tani)
                karakter = rfc.predict([oznitelikler])[0]
                ind = index.index(karakter)
                sinif = siniflar[ind]
                if sinif=="arkaplan":
                    continue
                
                #cv2.putText(yaz, sinif, (minx-2,miny-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
                mevcutPlaka.append([sinif,minx])                         
                cv2.drawContours(img, [box], 0, (0,255,0), 2)       #   boxın üzerine bulduğun counteri çiz
        if len(mevcutPlaka)>0:
            mevcutPlaka = plakaAyristir(mevcutPlaka)                #   okunan değerin plaka oluğ olmadığını algılıyoruz
        
        return img,mevcutPlaka
    
    
    else:
        return img,plaka