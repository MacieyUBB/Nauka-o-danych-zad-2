# Nauka o danych I — Zadanie 2

Autor: **Maciej Grochowski**  
Numer albumu: **66190**

## Temat

`zadanie.py` wykonuje eksploracyjną analizę danych (EDA) dla zbioru
**Housing Prices Dataset** pobieranego automatycznie z Kaggle Hub:

<https://www.kaggle.com/datasets/yasserh/housing-prices-dataset>

Analiza zawiera:

- wczytanie danych, sprawdzenie typów, braków i statystyk opisowych,
- histogram z krzywą KDE oraz interpretację rozkładu cen,
- detekcję wartości odstających metodami Isolation Forest i LOF,
- PCA po standaryzacji danych,
- wizualizacje t-SNE i UMAP,
- interaktywny wykres w Altair,
- macierz korelacji,
- test ANOVA porównujący ceny w grupach liczby sypialni.

Po każdej analizie skrypt wypisuje obserwację lub interpretację wyniku.

## Struktura projektu

```text
Zad_2/
├── zadanie.py
├── README.md
├── README.txt
├── housing_po_analizie.csv       # generowany po uruchomieniu
└── wykresy/                      # generowany po uruchomieniu
```

## Wymagania i instalacja

Wymagany jest Python 3.10 lub nowszy oraz połączenie z internetem przy
pierwszym pobieraniu danych. Instalacja bibliotek:

```powershell
python -m pip install pandas numpy scikit-learn matplotlib seaborn plotly altair statsmodels umap-learn kagglehub
```

Jeżeli Kaggle poprosi o uwierzytelnienie, należy wykonać logowanie zgodnie
z komunikatem biblioteki `kagglehub`.

## Uruchomienie

W katalogu `Zad_2` wykonaj:

```powershell
python .\zadanie.py
```

Program pobierze plik `Housing.csv`, wykona wszystkie analizy, zapisze
oczyszczone dane z dodatkowymi wynikami do `housing_po_analizie.csv` oraz
zapisze wykresy w katalogu `wykresy`. Wykresy Plotly i Altair są zapisane
zarówno jako interaktywne pliki HTML, jak i statyczne kopie PNG. Pozostałe
wykresy są zapisane jako PNG.

## Interpretacja

Interpretacje są drukowane bezpośrednio w konsoli, w sekcjach odpowiadających
punktom polecenia. Wartości odstające są oznaczane, ale nie są automatycznie
usuwane, ponieważ mogą przedstawiać prawdziwe, rzadkie przypadki.

## Repozytorium

Pełny projekt jest dostępny w repozytorium GitHub:

[MacieyUBB/Nauka-o-danych-zad-2](https://github.com/MacieyUBB/Nauka-o-danych-zad-2)
