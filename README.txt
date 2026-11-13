NAUKA O DANYCH I — ZADANIE 2

Autor: Maciej Grochowski
Numer albumu: 66190

TEMAT

Plik zadanie.py wykonuje eksploracyjna analize danych (EDA) dla zbioru
Housing Prices Dataset pobieranego z Kaggle Hub:
https://www.kaggle.com/datasets/yasserh/housing-prices-dataset

Analiza obejmuje:
- wczytanie danych, typy kolumn, braki i statystyki opisowe,
- rozklad cen z histogramem i krzywa KDE,
- Isolation Forest oraz Local Outlier Factor,
- PCA,
- t-SNE,
- UMAP,
- interaktywny wykres Altair,
- macierz korelacji,
- test ANOVA dla ceny i liczby sypialni.

Po kazdej analizie skrypt wypisuje obserwacje i interpretacje wynikow.

STRUKTURA PROJEKTU

Zad_2/
|-- zadanie.py
|-- README.md
|-- README.txt
|-- housing_po_analizie.csv   (generowany po uruchomieniu)
`-- wykresy/                   (generowany po uruchomieniu)

WYMAGANIA I INSTALACJA

Wymagany jest Python 3.10 lub nowszy. Instalacja:
python -m pip install pandas numpy scikit-learn matplotlib seaborn plotly altair statsmodels umap-learn kagglehub

URUCHOMIENIE

W katalogu Zad_2 wykonaj:
python .\zadanie.py

Program pobierze Housing.csv z Kaggle, wykona analizy, zapisze dane
z dodatkowymi kolumnami do housing_po_analizie.csv i zapisze wykresy
w katalogu wykresy jako pliki PNG.

WARTOSCI ODSTAJACE

Isolation Forest i LOF oznaczaja obserwacje odstajace, ale skrypt ich
automatycznie nie usuwa. Nietypowa obserwacja moze byc prawdziwym,
rzadkim przypadkiem, a nie bledem danych.

REPOZYTORIUM

https://github.com/MacieyUBB/Nauka-o-danych-zad-2
