# Wykrywanie odpadów — część ML

## O co chodzi?

Celem projektu jest model, który widzi odpady na zdjęciu, rysuje wokół nich ramki i podaje klasę oraz pewność przewidywania. W przyszłości ma działać w aplikacji mobilnej. **Na tym etapie przygotowaliśmy i sprawdziliśmy tylko część ML.**

Używamy gotowego, małego modelu **YOLO11n** z wagami pretrained. Dostosowujemy go do sześciu klas: `BIODEGRADABLE`, `CARDBOARD`, `GLASS`, `METAL`, `PAPER`, `PLASTIC`. Nie tworzyliśmy własnej sieci ani sztucznych ramek.

## Co zrobiliśmy?

1. Sprawdziliśmy dataset `Trash Detection Dataset/CUSTOM_DATASET`: 5 078 zdjęć i 34 082 oznaczone obiekty. Zachowaliśmy oryginalne `train/valid/test` (3 565 / 1 008 / 505 zdjęć).
2. Zrobiliśmy EDA: liczebność klas, jakość i rozmiary zdjęć, rozmiary i położenie ramek oraz przykłady z prawdziwymi adnotacjami. Wyniki są w `runs/baseline_yolo_nano/figures/` i `ML_baseline.ipynb`.
3. Zweryfikowaliśmy adnotacje. Są 29 zdjęć bez pasującej etykiety, 30 etykiet bez zdjęcia, 3 puste etykiety i 29 par identycznych zdjęć w `train`. Nie znaleźliśmy identycznych zdjęć między splitami. Szczegóły: `runs/baseline_yolo_nano/audit_issues.json`. **Nie zmienialiśmy obrazów ani adnotacji.**
4. Przeprowadziliśmy krótki baseline YOLO11n na CPU: **3 epoki**, wejście 320 px, batch 32, seed 42. To **wstępny wynik akademicki**, a nie w pełni wytrenowany model.
5. Oceniliśmy `best.pt` najpierw na `valid`, potem na `test`. Na teście: **precision 0,279**, **recall 0,293**, **F1 0,285**, **mAP@0.50 0,193**, **mAP@0.50:0.95 0,106**. Najlepsza klasa według AP@0.50:0.95 to `METAL` (0,197), najsłabsza to `GLASS` (0,030). To za słaby wynik do użycia w aplikacji, ale stanowi uczciwy pierwszy punkt odniesienia.
6. CodeCarbon zarejestrował okno treningu **27,7 min** i oszacował **0,0106 kWh** oraz **0,0070 kg CO₂eq**. Na tym Macu nie było bezpośredniego licznika energii CPU; liczby są szacunkami opartymi na obciążeniu i TDP. Okno pomiaru obejmuje też krótki fragment następnego batcha, zanim przerwanie zadziałało.

Dataset jest niezbalansowany: `BIODEGRADABLE` to 55,8% obiektów, a `PAPER` 6,4%. Około 40% ramek zajmuje mniej niż 1% zdjęcia, więc wykrywanie małych obiektów może być trudne. Obrazy `valid` i `test` są przeciętnie mniej ostre niż `train`. Wśród przykładów `BIODEGRADABLE` są półki ze świeżymi owocami i warzywami, więc model może mylić żywność z odpadami. Bez prawdziwych kadrów z telefonu nie wiemy jeszcze, jak model zadziała w aplikacji.

## Jak uruchomić?

Do ponownego uruchomienia potrzebny jest lokalny folder `Trash Detection Dataset/CUSTOM_DATASET` z obrazami, etykietami i `data.yaml`. Ten dataset nie jest częścią gałęzi `oskar`; zapisane wyniki i wagi można przeglądać bez niego.

Z katalogu głównego repozytorium, używając Python 3.13:

```bash
python -m pip install -r requirements-ml.txt
python -m ml_baseline.audit --dataset 'Trash Detection Dataset/CUSTOM_DATASET' --out runs/academic_3_epoch_example
python -m ml_baseline.train --dataset 'Trash Detection Dataset/CUSTOM_DATASET' --out runs/academic_3_epoch_example --epochs 3 --batch 32 --imgsz 320 --patience 5 > runs/academic_3_epoch_example/training.log 2>&1
```

Trening jest wolny na CPU. Komendy zapisują **nowy** eksperyment, więc nie nadpisują pokazanych niżej wyników. Zapisany przebieg wystartował z limitem 10 epok i został zatrzymany po trzeciej na życzenie użytkownika. Komenda `--epochs 3` uruchamia podobny krótki eksperyment od początku, ale harmonogram learning rate może się różnić, więc wagi nie muszą być identyczne. Do obejrzenia obecnych wyników nie trzeba trenować ponownie.

## Gdzie szukać wyników?

- `ML_baseline.ipynb` — przewodnik po EDA i baseline.
- `runs/baseline_yolo_nano/dataset_summary.json` — liczby dotyczące datasetu.
- `runs/baseline_yolo_nano/training_metrics.csv` — metryki kolejnych epok.
- `runs/baseline_yolo_nano/config.yaml` — ustawienia i wersje bibliotek.
- `runs/baseline_yolo_nano/weights/best.pt` — najlepszy zapisany model PyTorch.
- `runs/baseline_yolo_nano/green_ai.json` — czas i oszacowana energia krótkiego treningu.
- `runs/baseline_yolo_nano/final_metrics.json` — końcowa ocena na valid i test.

**Precision** mówi, jaka część przewidywań była trafna. **Recall** mówi, jaką część rzeczywistych obiektów model znalazł. **mAP** podsumowuje jakość wykrywania i dopasowania ramek; im wyższy, tym lepiej.

## Co dalej?

Najpierw warto przejrzeć wadliwe kopie plików i sprawdzić model na zdjęciach z telefonu. Dopiero potem ma sens dłuższy trening i eksperymenty z poprawą małych oraz rzadkich klas. Aplikacja mobilna i eksport ONNX nie są jeszcze częścią tego etapu.
