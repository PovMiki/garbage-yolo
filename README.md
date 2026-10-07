# 🗑️ EDA – Trash Detection Dataset

Notebook `EDA.ipynb` robi **eksploracyjną analizę danych (EDA)** zbioru zdjęć śmieci do detekcji obiektów (format YOLO). Nie trenuje żadnego modelu – sprawdza tylko, **co jest w danych i czy są z nimi problemy**, zanim ktokolwiek zacznie trening.

## Co jest w zbiorze

- Ścieżka: `Trash Detection Dataset/CUSTOM_DATASET/`
- Podział na `train`, `valid`, `test` (w każdym foldery `images/` i `labels/`)
- Plik `data.yaml` z nazwami klas
- 6 klas: `BIODEGRADABLE`, `CARDBOARD`, `GLASS`, `METAL`, `PAPER`, `PLASTIC`
- Etykiety w formacie YOLO: `class_id x_center y_center width height` (wartości 0–1)

## Jak uruchomić

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

## Najważniejsze wyniki

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
  - **1158 błędów adnotacji** (pierwsze 30 widać w ostatniej komórce z tabelą błędów)
- Splity `valid` i `test` są czyste – 0 błędów.

Zanim ruszy trening, trzeba przejrzeć i naprawić (lub wyrzucić) wadliwe pliki w `train`.

## Struktura repo

```
.
├── EDA.ipynb
├── README.md
└── Trash Detection Dataset/
    └── CUSTOM_DATASET/
        ├── data.yaml
        ├── train/ (images, labels)
        ├── valid/ (images, labels)
        └── test/  (images, labels)
```
