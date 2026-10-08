# 🗑️ Garbage YOLO — rozpoznawanie odpadów

Celem projektu jest mobilny system AI rozpoznający i lokalizujący odpady na obrazie z kamery telefonu w czasie rzeczywistym. Obecny etap obejmuje analizę datasetu, przygotowanie danych oraz kod treningu modelu bazowego **YOLO11 Nano** z pomiarem wpływu środowiskowego. Eksport modelu i integracja z aplikacją mobilną będą kolejnymi etapami.

Notebook `EDA.ipynb` robi **eksploracyjną analizę danych (EDA)** zbioru zdjęć śmieci do detekcji obiektów (format YOLO). Nie trenuje żadnego modelu – sprawdza tylko, **co jest w danych i czy są z nimi problemy**, zanim ktokolwiek zacznie trening.

## Co jest w zbiorze

- Ścieżka: `Trash Detection Dataset/CUSTOM_DATASET/`
- Podział na `train`, `valid`, `test` (w każdym foldery `images/` i `labels/`)
- Plik `data.yaml` z nazwami klas
- 6 klas: `BIODEGRADABLE`, `CARDBOARD`, `GLASS`, `METAL`, `PAPER`, `PLASTIC`
- Etykiety YOLO: bounding boxy `class_id x_center y_center width height` (wartości 0–1); część oznaczeń treningowych zawiera wielokąty segmentacji, konwertowane podczas przygotowania danych.

## Jak uruchomić EDA

1. Wrzuć folder `Trash Detection Dataset` obok notebooka (albo zmień `DATASET_ROOT` w drugiej komórce).
2. Zainstaluj biblioteki:
   ```bash
   pip install numpy pandas matplotlib seaborn opencv-python pyyaml jupyter
   ```
3. Odpal `EDA.ipynb` i wykonaj komórki po kolei.

## Co robi kod – krok po kroku

| Krok | Co się dzieje |
|------|---------------|
| **Importy i konfiguracja** | Ładuje biblioteki, ustawia ścieżki do splitów, rozszerzenia obrazów oraz progi: mały box < 1% powierzchni zdjęcia, duży box > 50%. Ustawia seed `42`. |
| **Wczytanie klas** | Czyta `data.yaml` i robi słownik `id → nazwa klasy`. |
| **`analyze_split()`** | Główna funkcja. Dla każdego splitu przechodzi po wszystkich zdjęciach i plikach `.txt`. Opis poniżej. |
| **Podsumowanie splitów** | Tabela: liczba zdjęć, labeli, boxów, średnia/min/max obiektów na zdjęcie oraz liczniki problemów. |
| **Rozkład klas** | Wykres słupkowy liczby boxów na klasę (osobno dla train/valid/test), procenty i **imbalance ratio** (największa klasa / najmniejsza). |
| **Obiekty na zdjęcie** | Statystyki i histogramy (oś X ucięta do 99. percentyla, żeby kilka zdjęć z setkami obiektów nie psuło wykresu). |
| **Rozdzielczości** | Sprawdza, czy wszystkie zdjęcia mają ten sam rozmiar. |
| **Rozmiary boxów** | `describe()` dla szerokości, wysokości, pola i proporcji boxów. |
| **Położenie boxów** | Scatter plot środków boxów (gdzie na zdjęciu najczęściej leżą obiekty). |
| **Raport błędów** | Podsumowanie problemów per split i podgląd pierwszych 30 błędów adnotacji. |

### Co sprawdza `analyze_split()`

- czy każde zdjęcie ma plik z labelem (`missing_labels`) i odwrotnie (`orphan_labels`)
- czy zdjęcia da się otworzyć (`broken_images`) i czy labele nie są puste (`empty_labels`)
- czy każda linia ma dokładnie 5 wartości i czy są to poprawne liczby (bez `NaN`/`inf`)
- czy `class_id` istnieje w `data.yaml`
- czy box mieści się w zdjęciu (środek w 0–1, szerokość/wysokość > 0 i ≤ 1, krawędzie nie wychodzą poza obraz)
- dla poprawnych boxów liczy rozmiar w px, pole, proporcje oraz flagi `is_small` / `is_large`

Zła linia **nie przerywa** analizy – trafia do tabeli błędów i jest pomijana w statystykach.

## Najważniejsze wyniki EDA

Poniższe statystyki pochodzą z pierwotnej analizy, przed konwersją wielokątów na bounding boxy. Nie są podsumowaniem oczyszczonego zbioru.

**Rozmiary splitów**

| Split | Zdjęcia | Boxy | Śr. obiektów / zdjęcie |
|-------|--------:|-----:|-----------------------:|
| train | 3565 | 22 354 | 6.27 |
| valid | 1008 | 7 284 | 7.23 |
| test  | 505  | 3 286 | 6.51 |

**Rozkład klas (razem)**

| Klasa | Boxy | % |
|-------|-----:|--:|
| BIODEGRADABLE | 18 123 | 55.0% |
| PLASTIC | 3 846 | 11.7% |
| METAL | 3 761 | 11.4% |
| CARDBOARD | 2 681 | 8.1% |
| GLASS | 2 372 | 7.2% |
| PAPER | 2 141 | 6.5% |

## ⚠️ Wnioski i na co uważać

- **Silny niezbalansowany zbiór** – `BIODEGRADABLE` to ponad połowa wszystkich boxów (imbalance ok. **8.46 : 1** względem `PAPER`). Przy treningu warto pomyśleć o wagach klas lub augmentacji.
- **Wszystkie zdjęcia mają 640×640** – nie trzeba nic ujednolicać.
- **Mediana to 1–2 obiekty na zdjęcie, ale zdarzają się zdjęcia z ponad 270 obiektami** – to outliery, które mocno ciągną średnią w górę.
- **Split `train` ma problemy z danymi:**
  - 29 zdjęć bez pliku z labelem
  - 30 labeli bez zdjęcia
  - 3 puste labele
  - **1158 oznaczeń wielokątami**, które EDA zgłasza jako błędy, ponieważ oczekuje 5 wartości w każdym wierszu; `prep_dataset.ipynb` zamienia je na bounding boxy
- Splity `valid` i `test` są czyste – 0 błędów.

Przed treningiem uruchom `prep_dataset.ipynb`, aby uporządkować dane i ujednolicić format oznaczeń.

## Przygotowanie danych i trening baseline

Wymagany jest Python 3.10+ oraz Jupyter. Uruchamiaj notebooki z katalogu głównego projektu `garbage-yolo`, aby ścieżki do danych były poprawne. Pierwsze komórki nowych notebooków instalują potrzebne biblioteki; instalacja oraz pierwsze pobranie wag modelu wymagają internetu.

1. `EDA.ipynb` — analiza danych, wykonana na wcześniejszym etapie.
2. `prep_dataset.ipynb` — przygotowanie oczyszczonego zbioru.
3. `train_baseline.ipynb` — trening, pomiar Green AI i końcowa ocena na zbiorze testowym.

### `prep_dataset.ipynb`

Notebook tworzy osobny folder `dataset_clean/`, pozostawiając źródłowy dataset bez zmian. Zachowuje wszystkie 6 klas i istniejący podział danych. Sprawdza poprawność zdjęć i etykiet, konwertuje wielokąty na bounding boxy oraz pomija całe pary z brakującymi, pustymi lub niepoprawnymi oznaczeniami. Puste etykiety są odrzucane, ponieważ nie potwierdzono, że przedstawiają wyłącznie tło.

Do nowego folderu trafiają tylko potrzebne obrazy, ich etykiety, poprawiony `data.yaml` i raport `prep_report.json`. Osierocone etykiety i stare pliki cache są pomijane. Identyczne pliki obrazów są usuwane także między splitami, z pierwszeństwem dla `test`, następnie `valid` i `train`; nie jest to wykrywanie podobnych zdjęć. Powtórzone boxy w jednej etykiecie również są usuwane.

Kontrola reguł na obecnym zbiorze, bez modyfikowania źródła, wskazała:

| Split | Zdjęcia po oczyszczeniu |
|-------|-----------------------:|
| train | 3533 |
| valid | 1008 |
| test | 505 |

W `train` pomijane są 32 zdjęcia (29 bez etykiety i 3 z pustą etykietą), a 1158 wielokątów jest konwertowanych do boxów. Raport wygenerowany po uruchomieniu notebooka zawiera aktualne liczniki i przyczyny odrzucenia plików.

Notebook nie nadpisuje istniejącego `dataset_clean/`. Przy ponownym przygotowaniu ustaw nową ścieżkę `OUT`; wtedy dostosuj również `DATA` w notebooku treningowym. Po przeniesieniu gotowego zbioru na inną maszynę zaktualizuj pole `path` w jego `data.yaml`, ponieważ zawiera ścieżkę absolutną.

### `train_baseline.ipynb`

Baseline wykorzystuje wstępnie wytrenowany model `yolo11n.pt`. Wywołanie `model.train()` realizuje pętlę epok i batchy, oblicza funkcję straty, aktualizuje wagi oraz prowadzi walidację i zapis checkpointów.

| Parametr | Ustawienie |
|----------|------------|
| Model | YOLO11 Nano, detekcja 6 klas |
| Liczba epok | maksymalnie 50 |
| Batch / rozmiar obrazu | 8 / 640 × 640 |
| Optymalizator / początkowy learning rate | AdamW / 0.001 |
| Early stopping | `patience=10` |
| Seed | 42 |
| Urządzenie | automatycznie CUDA → MPS → CPU |

Najlepszy model jest wybierany na zbiorze `valid`. Zbiór `test` służy końcowej ocenie: mAP50, mAP50–95, precision i recall. W razie braku pamięci zmniejsz `BATCH` w komórce konfiguracyjnej.

### Green AI — energia i emisje

`OfflineEmissionsTracker` z CodeCarbon śledzi całe wywołanie treningowe, w tym walidację wykonywaną przez trener. Końcowy test jest poza tym pomiarem. Tracker zapisuje raport także przy błędzie lub przerwaniu treningu dzięki blokowi `try/finally`.

- `emissions` — szacowana emisja CO₂eq w kg.
- `energy_consumed` — szacowane zużycie energii w kWh.
- `duration` — czas pomiaru w sekundach.

`COUNTRY = "POL"` należy dostosować do fizycznej lokalizacji komputera lub serwera. Pomiar dotyczy całej maszyny, więc inne procesy wpływają na wynik. Na Apple Silicon pomiar energii GPU może być niepełny lub wymagać dodatkowych uprawnień. Dane CodeCarbon są estymacją; brak raportu nie oznacza zerowej emisji. Raport jest zapisywany lokalnie, bez wysyłania do API CodeCarbon.

### Pliki wynikowe

Każde uruchomienie treningu tworzy własny folder `runs/baseline_<data>_<czas>_<id>/`:

| Plik | Zawartość |
|------|-----------|
| `weights/best.pt` | najlepsze wagi według walidacji |
| `weights/last.pt` | ostatni checkpoint |
| `results.csv` | straty i metryki kolejnych epok |
| `emissions.csv` | emisje CO₂eq, energia i czas |
| `environment.json` | wersje bibliotek, urządzenie, kraj i seed |
| `test_metrics.json` | metryki końcowego testu |
| `test/` | wyniki i wykresy końcowej walidacji na zbiorze testowym |

Przy przygotowaniu notebooków sprawdzono składnię i reguły czyszczenia danych. **Trening nie został uruchomiony** — repozytorium nie zawiera jeszcze wyników wytrenowanego baseline'u.

## Struktura repo

```text
.
├── EDA.ipynb
├── prep_dataset.ipynb
├── train_baseline.ipynb
├── README.md
├── Trash Detection Dataset/
│   └── CUSTOM_DATASET/
│       ├── data.yaml
│       ├── train/ (images, labels)
│       ├── valid/ (images, labels)
│       └── test/  (images, labels)
├── dataset_clean/               # powstaje po przygotowaniu danych
│   ├── data.yaml
│   ├── prep_report.json
│   ├── train/ (images, labels)
│   ├── valid/ (images, labels)
│   └── test/  (images, labels)
└── runs/                        # powstaje po uruchomieniu treningu
```
