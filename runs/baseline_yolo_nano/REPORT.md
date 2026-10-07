# Raport baseline YOLO Nano

## Model baseline

- Model: YOLO11n detection z pretrained `yolo11n.pt`.
- Dataset: `/Users/oskar/studia/sem7/projekt_Gielczyk/garbage-yolo/Trash Detection Dataset/CUSTOM_DATASET`; 6 klas: BIODEGRADABLE, CARDBOARD, GLASS, METAL, PAPER, PLASTIC.
- Train: 3565 obrazów, 23512 obiektów.
- Validation: 1008 obrazów, 7284 obiektów.
- Test: 505 obrazów, 3286 obiektów.

## Walidacja danych i splitu

| Split | Obrazy | Pliki adnotacji | Obiekty | Udział obrazów | Śr. obiektów/obraz | Min–max | Obrazy bez obiektów |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 3565 | 3566 | 23512 | 70.2% | 6.60 | 0–293 | 32 |
| validation | 1008 | 1008 | 7284 | 19.9% | 7.23 | 1–311 | 0 |
| test | 505 | 505 | 3286 | 9.9% | 6.51 | 1–271 | 0 |

Łącznie jest **5078 obrazów** i **34082 obiektów**. Rozdzielczość: 1 unikalna (640 × 640 px). Adnotacje: 32924 wierszy ramek i 1158 wierszy poligonów. Ramki poligonów zostały wyliczone z oryginalnych współrzędnych przez Ultralytics. Nie wygenerowano sztucznych adnotacji.

Problemy wykryte w całym zbiorze:
- **image_without_label: 29**
- **label_without_image: 30**
- **empty_label: 3**
- **identical_image_duplicate: 29**

Szczegóły i przykładowe ścieżki: `audit_issues.json`. 29 par identycznych obrazów jest **wewnątrz train**. Nie ma identycznych obrazów przechodzących między train, validation i test. Heurystyczna kontrola identyfikatorów źródła w nazwie pliku przed `.rf.` również nie znalazła wspólnych identyfikatorów między splitami (0); nie zastępuje to wizualnej kontroli near-duplicates. Większość problemów parowania wynika z dopisków `(1)` do nazw kopii plików: kopie obrazów są bez etykiet, a część kopii etykiet nie ma obrazu. Przed kolejnymi eksperymentami należy porównać kopie z oryginałami i usunąć niespójne kopie po ręcznym przeglądzie; w tym baseline dane pozostawiono bez zmian. Puste etykiety także wymagają przeglądu. Audyt nie znalazł nieczytelnych obrazów, nieprawidłowych identyfikatorów klas, wartości NaN/Inf ani ramek poza obrazem.

## EDA detekcji

| Klasa | Obiekty | Udział | Obrazy z klasą |
|---|---:|---:|---:|
| BIODEGRADABLE | 19019 | 55.8% | 898 |
| CARDBOARD | 2718 | 8.0% | 871 |
| GLASS | 2407 | 7.1% | 861 |
| METAL | 3820 | 11.2% | 1167 |
| PAPER | 2168 | 6.4% | 844 |
| PLASTIC | 3950 | 11.6% | 933 |

Imbalance ratio: **8.77**. Najmniejsza klasa to PAPER. Największa: BIODEGRADABLE. Największa różnica udziału klasy między splitami to 14.1 punktu procentowego dla BIODEGRADABLE; proporcje nie są identyczne.

Jasność obrazu: mediana train 178.4, validation 177.6, test 175.2 w skali 0–255. Kontrast (odchylenie skali szarości): 51.6, 51.6, 47.3. Ostrość (wariancja Laplacianu): **166.7** w train, **105.7** w validation, **119.1** w test. To sygnał możliwego przesunięcia jakości obrazu. Metryki jakości obrazu są wskaźnikami do ręcznej inspekcji, nie dowodem na przyczynę.

**40.1% obiektów** ma ramkę mniejszą niż 1% powierzchni obrazu; **6.2%** ma ramkę większą niż 50%. **42.3%** środków ramek leży w środkowej połowie szerokości i wysokości obrazu (przy równomiernym rozkładzie byłoby 25%). Widać umiarkowaną centralizację. Mediany powierzchni ramek per klasa i rozkłady znajdują się w `object_statistics.csv` i `figures/`.

Dataset składa się z kwadratowych obrazów 640 × 640 px. W galerii `most_objects.png` część kadrów klasy BIODEGRADABLE przedstawia półki z owocami i warzywami. To widoczny sygnał ryzyka, że model nauczy się rozpoznawać świeżą żywność jako „odpad”. Bez metadanych o pochodzeniu oraz próbnych kadrów z docelowej kamery telefonu **nie można potwierdzić reprezentatywności** scen mobilnych; jednolita rozdzielczość i różnice ostrości są dodatkowymi powodami do testu w realnej domenie.

## Trening

- Epoki: 3 / 10; zatrzymano po 3. epoce na życzenie użytkownika; najlepsza epoka: 3.
- Wejście: 320 px; batch: 32.
- Optymalizator: AdamW; bazowy learning rate z logu optymalizatora: 0.001000; weight decay: 0.0005.
- Early stopping patience: 5; seed: 42; urządzenie: cpu.
- Czas okna treningu: 27.7 min.
- Szczegóły epok: `training_metrics.csv`; pełny przebieg i ostrzeżenia: `training.log`.

## Jakość na teście

Precision i recall poniżej to średnie po klasach zwracane przez Ultralytics; F1 jest średnią harmoniczną tych dwóch wartości. mAP również jest średnią po klasach.

- Precision: **0.279**
- Recall: **0.293**
- F1-score: **0.285**
- mAP@0.50: **0.193**
- mAP@0.50:0.95: **0.106**

| Klasa | Precision | Recall | F1 | AP@0.50 | AP@0.50:0.95 |
|---|---:|---:|---:|---:|---:|
| BIODEGRADABLE | 0.220 | 0.400 | 0.284 | 0.223 | 0.085 |
| CARDBOARD | 0.393 | 0.209 | 0.273 | 0.214 | 0.130 |
| GLASS | 0.254 | 0.036 | 0.063 | 0.061 | 0.030 |
| METAL | 0.255 | 0.452 | 0.326 | 0.317 | 0.197 |
| PAPER | 0.404 | 0.176 | 0.246 | 0.181 | 0.122 |
| PLASTIC | 0.145 | 0.483 | 0.223 | 0.164 | 0.069 |

Najlepiej wykrywana klasa według AP@0.50:0.95: **METAL**; najsłabiej: **GLASS**. Macierze pomyłek, PR i krzywe confidence są w `figures/`; przykłady trafień, false positives, false negatives, pomyłek klas i niskiej pewności w `predictions/`. Kategorie przykładów służą inspekcji; statystyki globalne pochodzą z ewaluacji Ultralytics.

## Green AI

- Energia wg CodeCarbon: **0.010574 kWh**.
- Emisja wg CodeCarbon: **0.006999 kg CO₂eq**.
- Czas: **27.7 min**.
- CPU / GPU / RAM energy: 0.007806 / 0.000000 / 0.002768 kWh.

CodeCarbon na tym macOS nie miał dostępu do bezpośredniego licznika energii CPU i oszacował pobór na podstawie obciążenia oraz TDP. Są to **oszacowania z rzeczywistego przebiegu**, nie pomiar watomierzem. Nie należy interpretować ich jako dokładnych odczytów sprzętowych. Okno obejmuje trzy pełne epoki i ich walidację, bez EDA i końcowej ewaluacji. Po zapisie trzeciej epoki proces rozpoczął fragment kolejnego batcha przed przerwaniem; ten krótki fragment również mógł wejść do oszacowania.

## Plik modelu

- `best.pt`: `/Users/oskar/studia/sem7/projekt_Gielczyk/garbage-yolo/runs/baseline_yolo_nano/weights/best.pt` (21.21 MB).
- `last.pt`: `/Users/oskar/studia/sem7/projekt_Gielczyk/garbage-yolo/runs/baseline_yolo_nano/weights/last.pt`.

## Wnioski i następne eksperymenty

1. Dataset jest wyraźnie niezbalansowany: największa klasa ma 8.77 razy więcej obiektów niż najmniejsza. Najmniejsze klasy to PAPER, GLASS.
2. Małe obiekty są istotnym wyzwaniem: 40.1% ramek zajmuje <1% obrazu. Bez osobnej metryki AP względem wielkości nie można ilościowo stwierdzić, jak mocno obniżają jakość modelu.
3. Środki ramek są częściej w centrum niż przy rozkładzie równomiernym (42.3% wobec 25%). Możliwy bias obejmuje też różnice ostrości między splitami i nieznaną zgodność z obrazami z telefonu.
4. Precision i recall na teście różnią się o 0.014. Ocena, czy proporcja jest wystarczająca w aplikacji mobilnej, wymaga ustalenia kosztu false positives i false negatives.
5. Brak wystarczających danych epok do oceny trendu overfittingu. Wynik validation mAP50:0.95 0.154 wobec test 0.106 pokazuje zmianę jakości między splitami, ale sam nie dowodzi przyczyny.
6. Ten model jest udokumentowanym, krótkim punktem odniesienia. Pierwotny harmonogram miał 10 epok, lecz został przerwany po trzeciej; nowy trening z `--epochs 3` może dać inny harmonogram learning rate i inne wagi. Kolejne kroki: przegląd niespójnych kopii i pustych etykiet, kontrola małych obiektów i skuteczności per klasa, test na rzeczywistych kadrach telefonu oraz ewentualne eksperymenty poprawiające balans. Test pozostaje zestawem końcowej oceny, a nie strojenia.
