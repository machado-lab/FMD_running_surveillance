# =========================================
# Spatial smoothing of javali risk
# =========================================

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from libpysal.weights import KNN, Queen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import CADASTRO_PKL, INTERMEDIATE, JAVALI_DIR, JAVALI_PKL


# Make matplotlib cache writable in restricted environments.
MPL_CACHE = INTERMEDIATE / ".mpl_cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA = 0.6
ITERATIONS = 3
KNN_K = 6


def _normalize_ibge(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("Int64")
    as_text = numeric.astype(str).str.replace("<NA>", "", regex=False).str.strip()
    return as_text


def _resolve_excel_path() -> Path:
    candidates = [
        JAVALI_DIR / " javaliesinrs.xlsx",  # current file name in repo
        JAVALI_DIR / "javaliesinrs.xlsx",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find javali Excel file. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def _load_javali_survey() -> pd.DataFrame:
    xlsx_path = _resolve_excel_path()
    print(f"Loading javali survey: {xlsx_path}")
    df = pd.read_excel(xlsx_path)
    df.columns = df.columns.str.strip()

    fixed_presence_col = (
        "NewColumn.Você tem conhecimento da presença de suínos asselvajados no município "
        "(Por observação direta ou relatos de produtores, controladores, etc.)?"
    )
    if fixed_presence_col in df.columns:
        presence_col = fixed_presence_col
    else:
        candidates = [
            c for c in df.columns
            if "presença de suínos asselvajados" in c.lower()
            or "presenca de suinos asselvajados" in c.lower()
        ]
        if not candidates:
            raise KeyError(
                "Could not find javali presence column in survey file. "
                f"Available columns: {df.columns.tolist()}"
            )
        presence_col = candidates[0]

    if "IBGE" not in df.columns:
        raise KeyError(
            "Missing required column 'IBGE' in javali survey file. "
            f"Available columns: {df.columns.tolist()}"
        )

    presence = df[presence_col].astype(str).str.strip().str.lower()
    df["javali_risk"] = np.select(
        [
            presence.isin(["sim", "yes", "y", "s"]),
            presence.isin(["não", "nao", "no", "n"]),
        ],
        [1.0, 0.0],
        default=np.nan,
    )
    df["javali_risk"] = df["javali_risk"].fillna(0.0)

    out = pd.DataFrame(
        {
            "code_muni": _normalize_ibge(df["IBGE"]),
            "javali_risk": df["javali_risk"],
        }
    )
    out = out[out["code_muni"] != ""].copy()
    # Multiple forms can exist for the same municipality: keep "any presence".
    out = out.groupby("code_muni", as_index=False)["javali_risk"].max()
    return out


def _iterative_smooth(risk: np.ndarray, weights, alpha: float, iterations: int) -> np.ndarray:
    risk_iter = risk.astype(float).copy()
    for _ in range(iterations):
        neighbor_mean = np.array(
            [
                np.mean([risk_iter[j] for j in weights.neighbors[i]])
                if weights.neighbors[i]
                else risk_iter[i]
                for i in range(len(risk_iter))
            ],
            dtype=float,
        )
        risk_iter = (1 - alpha) * risk_iter + alpha * neighbor_mean
    return risk_iter


def _queen_from_geodataframe(gdf):
    # libpysal has version differences around the use_index keyword.
    try:
        return Queen.from_dataframe(gdf, use_index=False)
    except TypeError:
        return Queen.from_dataframe(gdf)


def _plot_geodataframe(gdf, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(
        column="javali_risk_smooth_iter",
        cmap="viridis",
        legend=True,
        edgecolor="black",
        linewidth=0.2,
        ax=ax,
    )
    ax.set_title("Javali risk in municipalities")
    ax.set_axis_off()
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_fallback_scatter(df, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(
        df["lon"],
        df["lat"],
        c=df["javali_risk_smooth_iter"],
        cmap="viridis",
        s=20,
        linewidths=0,
    )
    fig.colorbar(scatter, ax=ax, label="Smoothed javali risk")
    ax.set_title("Javali risk in municipalities (fallback KNN smoothing)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _run_with_geobr(survey: pd.DataFrame):
    import geobr

    gdf = geobr.read_municipality(code_muni="RS", year=2020)
    gdf["code_muni"] = _normalize_ibge(gdf["code_muni"])
    gdf = gdf.merge(survey, on="code_muni", how="left")
    gdf["javali_risk"] = gdf["javali_risk"].fillna(0.0)

    w = _queen_from_geodataframe(gdf)
    w.transform = "R"
    risk = gdf["javali_risk"].to_numpy(dtype=float)
    gdf["javali_risk_smooth_iter"] = _iterative_smooth(
        risk=risk,
        weights=w,
        alpha=ALPHA,
        iterations=ITERATIONS,
    )

    _plot_geodataframe(gdf, JAVALI_DIR / "javali_risk_map_smoothed.png")
    return gdf


def _run_fallback_knn(survey: pd.DataFrame) -> pd.DataFrame:
    if not CADASTRO_PKL.exists():
        raise FileNotFoundError(
            f"Fallback requires cadastro data, but not found: {CADASTRO_PKL}"
        )

    cadastro = pd.read_pickle(CADASTRO_PKL)
    needed = {"COD_MUNICIPIO_IBGE", "LATITUDE_DECIMAL", "LONGITUDE_DECIMAL"}
    missing = sorted(needed - set(cadastro.columns))
    if missing:
        raise KeyError(f"Fallback cannot run. Missing cadastro columns: {missing}")

    muni = (
        cadastro[list(needed)]
        .dropna(subset=["COD_MUNICIPIO_IBGE", "LATITUDE_DECIMAL", "LONGITUDE_DECIMAL"])
        .assign(code_muni=lambda x: _normalize_ibge(x["COD_MUNICIPIO_IBGE"]))
        .query("code_muni != ''")
        .groupby("code_muni", as_index=False)
        .agg(
            lat=("LATITUDE_DECIMAL", "mean"),
            lon=("LONGITUDE_DECIMAL", "mean"),
        )
    )

    muni = muni.merge(survey, on="code_muni", how="left")
    muni["javali_risk"] = muni["javali_risk"].fillna(0.0)

    risk = muni["javali_risk"].to_numpy(dtype=float)
    if len(muni) > 1:
        k = min(KNN_K, len(muni) - 1)
        w = KNN.from_array(muni[["lon", "lat"]].to_numpy(), k=k)
        w.transform = "R"
        muni["javali_risk_smooth_iter"] = _iterative_smooth(
            risk=risk,
            weights=w,
            alpha=ALPHA,
            iterations=ITERATIONS,
        )
    else:
        muni["javali_risk_smooth_iter"] = risk

    _plot_fallback_scatter(muni, JAVALI_DIR / "javali_risk_map_smoothed_fallback.png")
    return muni


def main() -> None:
    survey = _load_javali_survey()
    print(f"Survey municipalities with labels: {len(survey)}")

    try:
        out = _run_with_geobr(survey)
        print("Used geobr municipality polygons + Queen contiguity smoothing.")
    except Exception as exc:
        print(f"geobr municipality load failed ({exc}). Switching to offline fallback.")
        out = _run_fallback_knn(survey)
        print("Used cadastro centroids + KNN smoothing fallback.")

    out.to_pickle(str(JAVALI_PKL))
    print(f"Saved: {JAVALI_PKL}")
    print(f"Output columns: {out.columns.tolist()}")
    print(out[["code_muni", "javali_risk_smooth_iter"]].head())


if __name__ == "__main__":
    main()
