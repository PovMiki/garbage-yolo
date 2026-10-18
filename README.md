# 🗑️ Garbage YOLO — rozpoznawanie odpadów

Celem projektu jest mobilny system AI rozpoznający i lokalizujący odpady na obrazie z kamery telefonu w czasie rzeczywistym. Obecny etap obejmuje analizę datasetu, przygotowanie danych, trening i porównanie modeli **YOLO11 Nano** (baseline, augmentacja) z pomiarem wpływu środowiskowego. Strojenie hiperparametrów, XAI, eksport do ONNX i aplikacja mobilna są jeszcze przed nami.

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

## Status projektu

| # | Krok | Stan |
|---|------|------|
| 1 | Dataset + EDA | ✅ `EDA.ipynb` |
| 3 | Preprocessing, podział train/valid/test | ✅ `prep_dataset.ipynb` → `dataset_clean/` |
| 2 | Baseline YOLO11n | ✅ jeden zamrożony config: `configs/baseline.yaml` |
| 4 | Metryki + Green AI | 🟡 wspólny `scripts/evaluate.py`; do przeliczenia dla pozostałych modeli |
| 5 | Augmentacja | 🟡 offline oversampling gotowy (`nowe/`), brak porównania z baseline'em na tym samym configu |
| 6 | Strojenie hiperparametrów | ❌ kod w `stare/aug_hiper.ipynb`, nigdy nie uruchomiony do końca |
| 7–10 | Wybór modelu, XAI, ONNX, aplikacja mobilna | ❌ |

## Baseline (zamrożony)

`configs/baseline.yaml`: YOLO11n (`yolo11n.pt`), 50 epok, imgsz 640, batch 8, seed 42, AdamW lr0=0.001, cosine LR, patience 10, **domyślne augmentacje Ultralytics** (mosaic, hsv, fliplr, scale…).
Wynik na teście (505 zdjęć), zreprodukowany lokalnie przez `scripts/evaluate.py`: **mAP50 0.476, mAP50-95 0.334**, precision 0.634, recall 0.421. Wagi: `stare/baseline_v0_best.pt`.

Zasada: **jeden eksperyment = kopia configu ze zmianą jednej rzeczy**, ten sam seed (najlepiej kilka seedów), jedna ewaluacja. Porównujemy zawsze z tym baseline'em.

## Uruchamianie (lokalnie lub w Colabie)

```bash
pip install -r requirements.txt
python scripts/train.py configs/baseline.yaml --name baseline_s42                     # trening + energia + ewaluacja
python scripts/train.py configs/baseline.yaml --name no_flip --set fliplr=0.0          # eksperyment: jedna zmiana
python scripts/evaluate.py runs/baseline_s42/weights/best.pt --name baseline_s42 \
    --seed 42 --config configs/baseline.yaml --emissions runs/baseline_s42/emissions.csv
```

`evaluate.py` dopisuje wiersz do `results/results.csv` (mAP50, mAP50-95, P, R, F1, latencja CPU, FPS, rozmiar, GFLOPs, parametry, kWh, commit, host) i do `results/results_per_class.csv`.

### Jak czytać metryki Green AI

- **Porównuj energię w kWh, nie CO₂.** CodeCarbon liczy CO₂ ze współczynnika sieci wykrytego regionu serwera; ten sam trening w innym regionie Colaba daje inne CO₂ (np. 0.35 vs 0.14 kg/kWh).
- **Latencję porównuj tylko w obrębie jednej maszyny i jednego uruchomienia.** Ten sam model dał na różnych sesjach Colaba 59–128 ms. To pomiar PyTorch na CPU (preprocessing + inferencja + NMS), nie ONNX na telefonie.
- **F1** to średnia z F1 per klasa liczonego z P i R Ultralytics (`F1_macro`).
- Jeden przebieg to jedna liczba; różnice rzędu 1 pp przy 505 zdjęciach testowych mogą być szumem.

## Dane: `prep_dataset.ipynb`

Tworzy `dataset_clean/` (źródło zostaje bez zmian): zachowuje 6 klas i split, konwertuje 1158 wielokątów na boxy, pomija całe pary z brakującymi/pustymi/niepoprawnymi etykietami (32 zdjęcia w `train`), usuwa identyczne pliki obrazów między splitami (pierwszeństwo: test, valid, train) i powtórzone boxy. Raport: `dataset_clean/prep_report.json`. Wynik: 3533 / 1008 / 505 zdjęć.
`dataset_clean/data.yaml` ma ścieżkę absolutną z komputera autora. `scripts/*` i notebooki Colaba tworzą własną kopię z poprawnym `path`, więc nic nie trzeba edytować.

## Notebooki historyczne (`stare/`, `nowe/`)

| Plik | Co to jest |
|------|-----------|
| `stare/baseline_colab.ipynb` | trening baseline_v0 na Colabie (T4, ~61 min, 0.076 kWh) |
| `stare/aug_hiper.ipynb` | augmentacje online + Keras Tuner; **nie uruchomiony do końca**, brak wyników |
| `stare/wyniki/` | wyniki baseline_v0 (`results.csv`, `test_metrics.json`, `emissions.csv`) |
| `nowe/Model_Baseline_KOD.ipynb` | baseline **bez żadnej augmentacji** (batch 16, optimizer auto, seed 0). **Nie jest naszym baseline'em** (test mAP50-95 0.248), tylko punktem odniesienia dla augmentacji offline |
| `nowe/augmentacja_colab.ipynb` | oversampling offline klas mniejszościowych (+861 zdjęć), model `aug_balanced` (test mAP50-95 0.259) |
| `nowe/podglad_augmentacji.ipynb` | podgląd augmentacji i rozkładu klas po oversamplingu |
| `nowe/wyniki/` | wyniki `baseline` (bez aug) i `aug_balanced` |

⚠️ Wyniki z `nowe/` **nie są porównywalne** z baseline'em z `configs/baseline.yaml` (inne augmentacje, batch, optimizer, seed). Oversampling offline nie został jeszcze sprawdzony na tym samym configu co baseline.

## Znane problemy do rozwiązania

- GLASS to najsłabsza klasa (~0.20 mAP50-95 w baseline). W macierzy pomyłek `aug_balanced` ok. 36% szkła trafia w tło (niewykryte), a BIODEGRADABLE i METAL są często wykrywane na tle.
- Dotychczasowe wyniki to pojedyncze przebiegi (jeden seed), bez przedziałów ufności.
- Brak pomiaru latencji dla ONNX Runtime (właściwego środowiska docelowego).

## Struktura repo

```text
.
├── EDA.ipynb
├── prep_dataset.ipynb
├── configs/baseline.yaml          # zamrożony baseline
├── scripts/                       # train.py, evaluate.py, common.py
├── results/                       # results.csv, results_per_class.csv
├── requirements.txt
├── stare/                         # baseline_v0 (domyślne aug.) + niedokończony tuning
├── nowe/                          # baseline bez aug., augmentacja offline
├── Trash Detection Dataset/       # dane źródłowe (Roboflow)
├── dataset_clean/                 # dane oczyszczone (train/valid/test)
└── runs/                          # wyniki lokalnych treningów (poza gitem)
```
