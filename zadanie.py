# Autor: Maciej Grochowski, nr albumu 66190

"""Eksploracyjna analiza danych Housing Prices z Kaggle."""

from pathlib import Path

import altair as alt
import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

import umap


AUTOR = "Maciej Grochowski"
NUMER_ALBUMU = "66190"
ZBIOR_KAGGLE = "yasserh/housing-prices-dataset"
CECHY = ["area", "bedrooms", "bathrooms", "stories", "parking"]
KATALOG_PROJEKTU = Path(__file__).resolve().parent
KATALOG_WYKRESOW = KATALOG_PROJEKTU / "wykresy"


def zapisz_wykres(rysunek: plt.Figure, nazwa: str) -> None:
    """Zapisuje wykres w katalogu wynikowym i zamyka jego figurę."""
    rysunek.tight_layout()
    rysunek.savefig(KATALOG_WYKRESOW / nazwa, dpi=150)
    plt.close(rysunek)


def interpretuj_skosnosc(skosnosc: float) -> str:
    if skosnosc > 0.5:
        return "Rozkład jest prawostronnie skośny: nieliczne drogie domy wydłużają prawy ogon."
    if skosnosc < -0.5:
        return "Rozkład jest lewostronnie skośny."
    return "Rozkład cen jest w przybliżeniu symetryczny."


def main() -> None:
    KATALOG_WYKRESOW.mkdir(exist_ok=True)

    sciezka_zbioru = Path(kagglehub.dataset_download(ZBIOR_KAGGLE))
    plik_csv = sciezka_zbioru / "Housing.csv"
    if not plik_csv.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku Housing.csv w {sciezka_zbioru}")

    df_surowe = pd.read_csv(plik_csv)
    print(f"Autor: {AUTOR}, nr albumu: {NUMER_ALBUMU}")
    print(f"\nDane pobrano z: {plik_csv}")
    print("\n--- 1. Wczytanie i przygotowanie danych ---")
    print(f"Rozmiar przed usunięciem braków: {df_surowe.shape}")
    print("Typy kolumn i braki:")
    print(df_surowe.dtypes)
    print(df_surowe.isna().sum())
    print("\nStatystyki opisowe:")
    print(df_surowe.describe(include="all").transpose())

    df = df_surowe.dropna().reset_index(drop=True).copy()
    print(f"\nUsunięto {len(df_surowe) - len(df)} wierszy z brakami.")
    if df.empty:
        raise ValueError("Po usunięciu brakujących wartości zbiór jest pusty.")

    print("\n--- 2. Rozkład cen ---")
    skosnosc_cen = df["price"].skew()
    print(f"Średnia cena: {df['price'].mean():.2f}")
    print(f"Mediana ceny: {df['price'].median():.2f}")
    print(f"Skośność rozkładu cen: {skosnosc_cen:.3f}")
    print(f"Obserwacja: {interpretuj_skosnosc(skosnosc_cen)}")
    rysunek, osie = plt.subplots(figsize=(8, 5))
    sns.histplot(df["price"], kde=True, ax=osie)
    osie.set(title="Rozkład cen domów", xlabel="Cena", ylabel="Liczba domów")
    zapisz_wykres(rysunek, "01_rozklad_cen.png")

    print("\n--- 3. Wartości odstające ---")
    model_iso = IsolationForest(contamination=0.05, random_state=42)
    df["isolation_forest"] = model_iso.fit_predict(df[["price", "area"]])
    liczba_iso = int((df["isolation_forest"] == -1).sum())
    print(f"Isolation Forest znalazł {liczba_iso} obserwacji odstających.")

    model_lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    df["local_outlier_factor"] = model_lof.fit_predict(df[["price", "area"]])
    liczba_lof = int((df["local_outlier_factor"] == -1).sum())
    print(f"Local Outlier Factor znalazł {liczba_lof} obserwacji odstających.")
    print(
        "Obserwacja: oba algorytmy wskazują niewielką grupę nietypowych "
        "kombinacji powierzchni i ceny; nie usuwam ich automatycznie."
    )

    df["status_odstajacej"] = df["isolation_forest"].map(
        {1: "normalna", -1: "odstająca"}
    )
    wykres_odstajacych = px.scatter(
        df,
        x="area",
        y="price",
        color="status_odstajacej",
        title="Wartości odstające według Isolation Forest",
        labels={"area": "Powierzchnia", "price": "Cena"},
    )
    rysunek, osie = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="area",
        y="price",
        hue="status_odstajacej",
        ax=osie,
    )
    osie.set(title="Wartości odstające według Isolation Forest",
             xlabel="Powierzchnia", ylabel="Cena")
    zapisz_wykres(rysunek, "02_wartosci_odstajace.png")

    print("\n--- 4. PCA ---")
    macierz_cech = StandardScaler().fit_transform(df[CECHY])
    pca = PCA(n_components=2)
    skladowe = pca.fit_transform(macierz_cech)
    df["PC1"], df["PC2"] = skladowe[:, 0], skladowe[:, 1]
    udzial_pca = pca.explained_variance_ratio_
    print(f"Udział wariancji PC1: {udzial_pca[0]:.3f}")
    print(f"Udział wariancji PC2: {udzial_pca[1]:.3f}")
    print(
        f"Obserwacja: dwie składowe wyjaśniają łącznie {udzial_pca.sum():.1%} "
        "zróżnicowania wybranych cech."
    )
    wykres_pca = px.scatter(
        df, x="PC1", y="PC2", color="price", title="PCA: dwie główne składowe"
    )
    rysunek, osie = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="price", palette="viridis", ax=osie)
    osie.set(title="PCA: dwie główne składowe", xlabel="PC1", ylabel="PC2")
    zapisz_wykres(rysunek, "03_pca.png")

    print("\n--- 5. t-SNE ---")
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    wspolrzedne_tsne = tsne.fit_transform(macierz_cech)
    df["tSNE1"], df["tSNE2"] = wspolrzedne_tsne[:, 0], wspolrzedne_tsne[:, 1]
    liczba_klas_prefarea = df["prefarea"].nunique()
    print(f"Wynik t-SNE ma {len(df)} punktów i 2 wymiary.")
    print(
        f"Obserwacja: wizualizacja pozwala ocenić lokalne skupienia; "
        f"kolumna prefarea ma {liczba_klas_prefarea} kategorii."
    )
    rysunek, osie = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="tSNE1", y="tSNE2", hue="prefarea", ax=osie)
    osie.set_title("Wizualizacja t-SNE")
    zapisz_wykres(rysunek, "04_tsne.png")

    print("\n--- 6. UMAP ---")
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, random_state=42)
    wspolrzedne_umap = reducer.fit_transform(macierz_cech)
    df["UMAP1"], df["UMAP2"] = wspolrzedne_umap[:, 0], wspolrzedne_umap[:, 1]
    print("UMAP utworzył dwuwymiarowe odwzorowanie danych.")
    print(
        "Obserwacja: na wykresie można sprawdzić, czy status umeblowania "
        "tworzy widoczne skupienia."
    )
    rysunek, osie = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="UMAP1", y="UMAP2", hue="furnishingstatus", ax=osie)
    osie.set_title("Wizualizacja UMAP")
    zapisz_wykres(rysunek, "05_umap.png")

    print("\n--- 7. Interaktywny wykres Altair ---")
    altair_chart = (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(
            x=alt.X("area:Q", title="Powierzchnia"),
            y=alt.Y("price:Q", title="Cena"),
            color=alt.Color("bedrooms:N", title="Sypialnie"),
            tooltip=["area", "price", "bedrooms", "bathrooms"],
        )
        .interactive()
    )
    rysunek, osie = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="area", y="price", hue="bedrooms", palette="deep", ax=osie)
    osie.set(title="Wykres zależności powierzchni i ceny",
             xlabel="Powierzchnia", ylabel="Cena")
    zapisz_wykres(rysunek, "06_wykres_altair.png")
    print("Wykres Altair przygotowano, a jego statyczną wersję zapisano jako PNG.")

    print("\n--- 8. Macierz korelacji ---")
    oryginalne_cechy_liczbowe = ["price", "area", "bedrooms", "bathrooms", "stories", "parking"]
    macierz_korelacji = df[oryginalne_cechy_liczbowe].corr()
    macierz_bez_przekatnej = macierz_korelacji.where(
        ~pd.DataFrame(
            [[i == j for j in macierz_korelacji.columns]
             for i in macierz_korelacji.columns],
            index=macierz_korelacji.index,
            columns=macierz_korelacji.columns,
        )
    )
    najsilniejsza_para = (
        macierz_bez_przekatnej.abs().stack().sort_values(ascending=False).index[0]
    )
    najsilniejsza_korelacja = macierz_korelacji.loc[najsilniejsza_para]
    print(macierz_korelacji.round(2))
    print(
        f"Obserwacja: najsilniejsza zależność liniowa występuje między "
        f"{najsilniejsza_para[0]} i {najsilniejsza_para[1]} "
        f"(r = {najsilniejsza_korelacja:.3f})."
    )
    rysunek, osie = plt.subplots(figsize=(10, 8))
    sns.heatmap(macierz_korelacji, annot=True, fmt=".2f", cmap="coolwarm", ax=osie)
    osie.set_title("Macierz korelacji")
    zapisz_wykres(rysunek, "07_macierz_korelacji.png")

    print("\n--- 9. Test ANOVA ---")
    model = ols("price ~ C(bedrooms)", data=df).fit()
    wynik_anova = anova_lm(model)
    p_value = float(wynik_anova.loc["C(bedrooms)", "PR(>F)"])
    print(wynik_anova)
    if p_value < 0.05:
        interpretacja_anova = (
            "Wynik jest istotny statystycznie: średnia cena różni się "
            "między co najmniej dwiema grupami liczby sypialni."
        )
    else:
        interpretacja_anova = (
            "Wynik nie jest istotny statystycznie: nie ma podstaw, aby "
            "stwierdzić różnicę średnich cen między grupami."
        )
    print(f"Obserwacja: {interpretacja_anova}")
    print("ANOVA nie wskazuje konkretnej pary grup; do tego potrzebny byłby test post hoc.")

    plik_wynikowy = KATALOG_PROJEKTU / "housing_po_analizie.csv"
    df.to_csv(plik_wynikowy, index=False)
    print(f"\nWyniki analizy zapisano w: {plik_wynikowy}")
    print(f"Wykresy zapisano w: {KATALOG_WYKRESOW}")


if __name__ == "__main__":
    main()
