import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

        

def plaka_konum(img):

    img_bgr = img
    img_gri = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)     #   resmi gri formatına dönüştürdük
    ar_img = cv2.medianBlur(img_gri, 3) 
    ar_img = cv2.medianBlur(ar_img, 3)                      #   seçili resmi 2 kere 3 birim bulanıklaşştırdık

    ortalama = np.median(ar_img)                            #   fotorafın yoğunluğunu numpy kütüpanesi ile buluyoruz
    low = 0.67*ortalama*1                                   #   yoğunluk alt ve üst değerlari
    high = 1.33*ortalama*1
    
    kenarlik = cv2.Canny(ar_img, low, high)                 #   canny algoritmasi ile kenarlık tespit ettik canny bir kenarlık bulma algoritmasıdır
                                                            #   sobel ve prewitt algoritmasıda işimizi görür ancak fotoğraflarda yakınlaştırma olduğu için en iyi canny
    
    kenarlik = cv2.dilate(kenarlik, np.ones((2,2), np.uint8), iterations=1)         #   cv2.dilate ile genişletme yaptık. numpy.ones ilede kernel oluşturduk

    cnt = cv2.findContours(kenarlik, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        #   diktörtgen bulmak için findContours kullandık(img, yapı, hangi metod)
    cnt = cnt[0]                                                                    #   sadece x,y,w,h değerlerini aldık
    cnt = sorted(cnt, key=cv2.contourArea, reverse=True)                            #   mevcut counter alanına göre sıralama yaptık

    h,w = 500,500
    plaka = None


    for c in cnt:
        rect = cv2.minAreaRect(c)                             #   dikdörtgeni olabildiğince minimum boyutlarda almasını istedik                           
        (x,y) , (w,h) , r = rect
        if(w > h and w > h * 2) or (h > w and h > w*2):         #   oranı en az 2 olan dikdörtgenler
            box = cv2.boxPoints(rect)                           #   dikdörtgen oluşturduk
            box = np.int64(box)
            
            minx = np.min(box[:,0])                             #   dikdörtgenin koşelerinin kordinatlarını alıyoruz
            miny = np.min(box[:,1])
            maxx = np.max(box[:,0])
            maxy = np.max(box[:,1])
            
            muh_plaka = img_gri[miny:maxy, minx:maxx].copy()    #aldığımız kısmı kopyaladık
            muh_medyan = np.median(muh_plaka)                   #muh_plakanın medyanını aldık
            
            kon1 = muh_medyan > 100 and muh_medyan < 200        # 1. kontrol medyan değeri 100 den büyük 200 den küçük olucak
            kon2 = abs(h / w - 4.2) < 2                          # 2. ve 3. kontrol boyut sınırlandırma ile ilgili
            kon3 = abs(w / h - 4.2) < 2
            kon4 = True #-20 <= np.degrees(r) <= -200 or 20 <= np.degrees(r) <= 200 
            
            kon = False
            
            if(kon1 and ((kon2 or kon3) and kon4) ):
                cv2.drawContours(img_bgr, [box], 0, (0,255,0), 2)       # kontrolleri sağlıyorsa plakayı yeşille çevrele
                print("resimdeki konumu ", minx, miny, maxx, maxy)
                
                plaka = [minx,miny,w,h]
                plaka = [int(i) for i in [minx,miny,w,h]]
                
                kon = True
            else:
                pass
            
            if(kon):
                return plaka               #   plakanın konumunu döndürüyoruz yani plaka değişkeni
            else:
                return None
        
    return[]

