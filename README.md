1. Analiza datesetu - rozmiary 512x384 wszystkie
Jasnosc od 78 do 224 przez jasne tlo na niektorych
Ostrosc ma duzy rozrzut od 2 do 4127
dodalem hash MD5 do wykrywania duplikatow zdjec

Wyszlo 0 zepsutych plikow i 3 duplikaty, klasy sa niezbalansowane bo trash ma tylko 137 zdj

2.Czyszczenie
Usuniecie duplikatow, nieostrych, za bialych i za czarnych - lacznie 160 zdjec do wywalenia bylo

3.Te ramki (ai do poprawy) 
albo porownywanie koloru z tlem albo wykrywanie krawedzi (canny) - rozmycie, wykrycie krawedzi, pogrubienie, wybor najwiekszego ksztaltu i prostokat wokol niego

4. Podzial danych 
Train 70 Val 15 test 15 ze stratyfikacja

5. Preprocessing i zapis dla yolo
Skopiowalem te zdjecia bo yolo wymaga struktury yolo/images/train,val,test i yolo/labbels/...
Dla kazdego zdjecia zapisalem txt z ramka w formacie yolo
Stworzylem .yamla zeby yolo wiedzialo gdzie sa dane i jak sie nazywaja klasy

6. Skalowanie do 512x512 z proporcjami i normalizacja robi podobniez samo YOLO przy wczytywaniu 



# OSKAR 06.10.2026

Dodałem drugi dataset oraz wykonałem porównanie, poprawiłem tez kilka rzeczy od Mikołaja 

Kolejny etap to próba wytrenowania modelu i sprawdzenia jak sobie radzi baseline + parametry i greenAI
