"""Generate a Polish baseline report using measured artifacts only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import yaml


def fmt(value,digits=3):
    return 'brak danych' if value is None else f'{value:.{digits}f}'


def generate_report(out: Path):
    out=out.resolve()
    s=json.loads((out/'dataset_summary.json').read_text())
    m=json.loads((out/'final_metrics.json').read_text())
    g=json.loads((out/'green_ai.json').read_text())
    c=yaml.safe_load((out/'config.yaml').read_text())
    im=pd.read_csv(out/'image_statistics.csv')
    bo=pd.read_csv(out/'object_statistics.csv')
    names=s['classes'];total=s['total_objects']
    class_rows=[]
    for name in names:
        x=s['objects_per_class'][name]
        class_rows.append(f'| {name} | {x} | {100*x/total:.1f}% | {s["images_per_class"][name]} |')
    split_rows=[]
    for name,v in s['splits'].items():
        split_rows.append(f'| {name} | {v["images"]} | {v["annotation_files"]} | {v["objects"]} | {v["share_pct"]:.1f}% | {v["mean_objects_per_image"]:.2f} | {v["min_objects_per_image"]}–{v["max_objects_per_image"]} | {v["images_without_objects"]} |')
    per=m['per_class']
    ap_values=[(name,val.get('ap50_95')) for name,val in per.items() if val.get('ap50_95') is not None]
    ap_values.sort(key=lambda x:x[1])
    worst=ap_values[0][0] if ap_values else 'brak danych'
    best=ap_values[-1][0] if ap_values else 'brak danych'
    per_rows=[]
    for name in names:
        v=per.get(name,{})
        per_rows.append(f'| {name} | {fmt(v.get("precision"))} | {fmt(v.get("recall"))} | {fmt(v.get("f1_score"))} | {fmt(v.get("ap50"))} | {fmt(v.get("ap50_95"))} |')
    split_distribution=pd.crosstab(bo.class_name,bo.split,normalize='columns')*100
    max_gap=(split_distribution.max(axis=1)-split_distribution.min(axis=1)).sort_values(ascending=False)
    gap_name=max_gap.index[0];gap=max_gap.iloc[0]
    med=im.groupby('split')[['brightness','contrast','sharpness']].median()
    val_sharp=med.loc['validation','sharpness'];train_sharp=med.loc['train','sharpness']
    small=s['small_objects_area_lt_1pct']/total*100
    large=s['large_objects_area_gt_50pct']/total*100
    central=s['central_boxes_center_in_middle_half']/total*100
    issue_lines='\n'.join(f'- **{k}: {v}**' for k,v in s['issue_counts'].items())
    best_path=out/'weights'/'best.pt'
    stop_note='; zatrzymano po 3. epoce na życzenie użytkownika' if c.get('stop_reason')=='user_requested_after_3_epochs' else ''
    history=pd.read_csv(out/'training_metrics.csv')
    history.columns=history.columns.str.strip()
    map_col=next((col for col in history if 'mAP50-95' in col),None)
    train_box_col=next((col for col in history if 'train/box_loss' in col),None)
    overfit='Brak wystarczających danych epok do oceny trendu overfittingu.'
    if len(history)>=4 and map_col and train_box_col:
        best_map=float(history[map_col].max())
        last_map=float(history[map_col].iloc[-1])
        train_loss_declined=float(history[train_box_col].iloc[-1])<float(history[train_box_col].iloc[0])
        if train_loss_declined and best_map-last_map>0.02:
            overfit=f'Widać możliwy overfitting: train box loss spada, a końcowe validation mAP50:0.95 ({last_map:.3f}) jest o {best_map-last_map:.3f} poniżej maksimum ({best_map:.3f}).'
        else:
            overfit=f'Brak wyraźnego sygnału overfittingu według train box loss i validation mAP50:0.95: maksimum {best_map:.3f}, ostatnia epoka {last_map:.3f}. Krótki trening nie wyklucza go przy dłuższym uczeniu.'
    text=f'''# Raport baseline YOLO Nano

## Model baseline

- Model: {c['model']} z pretrained `{c['pretrained_weights']}`.
- Dataset: `{c['dataset_path']}`; {len(names)} klas: {', '.join(names)}.
- Train: {s['splits']['train']['images']} obrazów, {s['splits']['train']['objects']} obiektów.
- Validation: {s['splits']['validation']['images']} obrazów, {s['splits']['validation']['objects']} obiektów.
- Test: {s['splits']['test']['images']} obrazów, {s['splits']['test']['objects']} obiektów.

## Walidacja danych i splitu

| Split | Obrazy | Pliki adnotacji | Obiekty | Udział obrazów | Śr. obiektów/obraz | Min–max | Obrazy bez obiektów |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(split_rows)}

Łącznie jest **{s['total_images']} obrazów** i **{total} obiektów**. Rozdzielczość: {s['unique_resolutions']} unikalna ({int(s['image_statistics']['width']['50%'])} × {int(s['image_statistics']['height']['50%'])} px). Adnotacje: {s['annotation_types'].get('bbox',0)} wierszy ramek i {s['annotation_types'].get('polygon',0)} wierszy poligonów. Ramki poligonów zostały wyliczone z oryginalnych współrzędnych przez Ultralytics. Nie wygenerowano sztucznych adnotacji.

Problemy wykryte w całym zbiorze:
{issue_lines}

Szczegóły i przykładowe ścieżki: `audit_issues.json`. 29 par identycznych obrazów jest **wewnątrz train**. Nie ma identycznych obrazów przechodzących między train, validation i test. Heurystyczna kontrola identyfikatorów źródła w nazwie pliku przed `.rf.` również nie znalazła wspólnych identyfikatorów między splitami ({len(s.get('cross_split_source_ids_by_filename', []))}); nie zastępuje to wizualnej kontroli near-duplicates. Większość problemów parowania wynika z dopisków `(1)` do nazw kopii plików: kopie obrazów są bez etykiet, a część kopii etykiet nie ma obrazu. Przed kolejnymi eksperymentami należy porównać kopie z oryginałami i usunąć niespójne kopie po ręcznym przeglądzie; w tym baseline dane pozostawiono bez zmian. Puste etykiety także wymagają przeglądu. Audyt nie znalazł nieczytelnych obrazów, nieprawidłowych identyfikatorów klas, wartości NaN/Inf ani ramek poza obrazem.

## EDA detekcji

| Klasa | Obiekty | Udział | Obrazy z klasą |
|---|---:|---:|---:|
{chr(10).join(class_rows)}

Imbalance ratio: **{s['imbalance_ratio']:.2f}**. Najmniejsza klasa to {min(s['objects_per_class'],key=s['objects_per_class'].get)}. Największa: {max(s['objects_per_class'],key=s['objects_per_class'].get)}. Największa różnica udziału klasy między splitami to {gap:.1f} punktu procentowego dla {gap_name}; proporcje nie są identyczne.

Jasność obrazu: mediana train {med.loc['train','brightness']:.1f}, validation {med.loc['validation','brightness']:.1f}, test {med.loc['test','brightness']:.1f} w skali 0–255. Kontrast (odchylenie skali szarości): {med.loc['train','contrast']:.1f}, {med.loc['validation','contrast']:.1f}, {med.loc['test','contrast']:.1f}. Ostrość (wariancja Laplacianu): **{train_sharp:.1f}** w train, **{val_sharp:.1f}** w validation, **{med.loc['test','sharpness']:.1f}** w test. To sygnał możliwego przesunięcia jakości obrazu. Metryki jakości obrazu są wskaźnikami do ręcznej inspekcji, nie dowodem na przyczynę.

**{small:.1f}% obiektów** ma ramkę mniejszą niż 1% powierzchni obrazu; **{large:.1f}%** ma ramkę większą niż 50%. **{central:.1f}%** środków ramek leży w środkowej połowie szerokości i wysokości obrazu (przy równomiernym rozkładzie byłoby 25%). Widać umiarkowaną centralizację. Mediany powierzchni ramek per klasa i rozkłady znajdują się w `object_statistics.csv` i `figures/`.

Dataset składa się z kwadratowych obrazów 640 × 640 px. W galerii `most_objects.png` część kadrów klasy BIODEGRADABLE przedstawia półki z owocami i warzywami. To widoczny sygnał ryzyka, że model nauczy się rozpoznawać świeżą żywność jako „odpad”. Bez metadanych o pochodzeniu oraz próbnych kadrów z docelowej kamery telefonu **nie można potwierdzić reprezentatywności** scen mobilnych; jednolita rozdzielczość i różnice ostrości są dodatkowymi powodami do testu w realnej domenie.

## Trening

- Epoki: {c['epochs_executed']} / {c['max_epochs']}{stop_note}; najlepsza epoka: {c.get('best_epoch')}.
- Wejście: {c['image_size']} px; batch: {c['batch_size']}.
- Optymalizator: {c.get('optimizer_actual')}; bazowy learning rate z logu optymalizatora: {fmt(c.get('learning_rate_actual'),6)}; weight decay: {c['weight_decay']}.
- Early stopping patience: {c['patience']}; seed: {c['seed']}; urządzenie: {c['device']}.
- Czas okna treningu: {fmt(g.get('training_time_minutes'),1)} min.
- Szczegóły epok: `training_metrics.csv`; pełny przebieg i ostrzeżenia: `training.log`.

## Jakość na teście

Precision i recall poniżej to średnie po klasach zwracane przez Ultralytics; F1 jest średnią harmoniczną tych dwóch wartości. mAP również jest średnią po klasach.

- Precision: **{fmt(m.get('precision'))}**
- Recall: **{fmt(m.get('recall'))}**
- F1-score: **{fmt(m.get('f1_score'))}**
- mAP@0.50: **{fmt(m.get('map50'))}**
- mAP@0.50:0.95: **{fmt(m.get('map50_95'))}**

| Klasa | Precision | Recall | F1 | AP@0.50 | AP@0.50:0.95 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(per_rows)}

Najlepiej wykrywana klasa według AP@0.50:0.95: **{best}**; najsłabiej: **{worst}**. Macierze pomyłek, PR i krzywe confidence są w `figures/`; przykłady trafień, false positives, false negatives, pomyłek klas i niskiej pewności w `predictions/`. Kategorie przykładów służą inspekcji; statystyki globalne pochodzą z ewaluacji Ultralytics.

## Green AI

- Energia wg CodeCarbon: **{fmt(g.get('energy_consumed_kwh'),6)} kWh**.
- Emisja wg CodeCarbon: **{fmt(g.get('co2_emissions_kg'),6)} kg CO₂eq**.
- Czas: **{fmt(g.get('training_time_minutes'),1)} min**.
- CPU / GPU / RAM energy: {fmt(g.get('cpu_energy_kwh'),6)} / {fmt(g.get('gpu_energy_kwh'),6)} / {fmt(g.get('ram_energy_kwh'),6)} kWh.

CodeCarbon na tym macOS nie miał dostępu do bezpośredniego licznika energii CPU i oszacował pobór na podstawie obciążenia oraz TDP. Są to **oszacowania z rzeczywistego przebiegu**, nie pomiar watomierzem. Nie należy interpretować ich jako dokładnych odczytów sprzętowych. Okno obejmuje trzy pełne epoki i ich walidację, bez EDA i końcowej ewaluacji. Po zapisie trzeciej epoki proces rozpoczął fragment kolejnego batcha przed przerwaniem; ten krótki fragment również mógł wejść do oszacowania.

## Plik modelu

- `best.pt`: `{best_path}` ({m['model_size_mb']:.2f} MB).
- `last.pt`: `{out/'weights'/'last.pt'}`.

## Wnioski i następne eksperymenty

1. Dataset jest wyraźnie niezbalansowany: największa klasa ma {s['imbalance_ratio']:.2f} razy więcej obiektów niż najmniejsza. Najmniejsze klasy to {', '.join(sorted(names,key=lambda n:s['objects_per_class'][n])[:2])}.
2. Małe obiekty są istotnym wyzwaniem: {small:.1f}% ramek zajmuje <1% obrazu. Bez osobnej metryki AP względem wielkości nie można ilościowo stwierdzić, jak mocno obniżają jakość modelu.
3. Środki ramek są częściej w centrum niż przy rozkładzie równomiernym ({central:.1f}% wobec 25%). Możliwy bias obejmuje też różnice ostrości między splitami i nieznaną zgodność z obrazami z telefonu.
4. Precision i recall na teście różnią się o {abs(m['precision']-m['recall']):.3f}. Ocena, czy proporcja jest wystarczająca w aplikacji mobilnej, wymaga ustalenia kosztu false positives i false negatives.
5. {overfit} Wynik validation mAP50:0.95 {fmt(m['validation']['map50_95'])} wobec test {fmt(m['map50_95'])} pokazuje zmianę jakości między splitami, ale sam nie dowodzi przyczyny.
6. Ten model jest udokumentowanym, krótkim punktem odniesienia. Pierwotny harmonogram miał 10 epok, lecz został przerwany po trzeciej; nowy trening z `--epochs 3` może dać inny harmonogram learning rate i inne wagi. Kolejne kroki: przegląd niespójnych kopii i pustych etykiet, kontrola małych obiektów i skuteczności per klasa, test na rzeczywistych kadrach telefonu oraz ewentualne eksperymenty poprawiające balans. Test pozostaje zestawem końcowej oceny, a nie strojenia.
'''
    (out/'REPORT.md').write_text(text)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    generate_report(args.out)

if __name__=='__main__':main()
