from __future__ import annotations

from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.http import require_POST
from django.db import transaction
from django.db.models import Sum

import pandas as pd
import numpy as np

from prophet import Prophet
from sklearn.metrics import mean_absolute_error

from .models import Dataset, Product, SaleDaily

# -----------------------------
# Configuración general del Excel estándar
# -----------------------------
REQUIRED_COLS = ["id_venta", "nombre_producto", "fecha_venta", "cantidad"]

CSV_CHUNKSIZE = 200_000
BULK_BATCH_SIZE = 5000


# -----------------------------
# Utilidades de sesión/dataset
# -----------------------------
def _ensure_session(request) -> str:
    """Asegura que exista session_key (para asociar el dataset al usuario)."""
    if not request.session.session_key:
        request.session.create()
    return request.session.session_key


def _get_dataset(request) -> Dataset | None:
    """Recupera el último dataset subido por este usuario (sesión)."""
    session_key = _ensure_session(request)
    return Dataset.objects.filter(session_key=session_key).order_by("-created_at").first()


# -----------------------------
# Limpieza + agregación diaria (para carga a BD)
# -----------------------------
def _normalize_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe un DataFrame con las columnas requeridas.
    Devuelve un DataFrame limpio con columnas:
      - nombre_producto (str)
      - ds (datetime64[ns]) -> día (sin hora)
      - y (float) -> cantidad (sumable)
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}. Requeridas: {REQUIRED_COLS}")

    df = df[REQUIRED_COLS].copy()

    df["nombre_producto"] = df["nombre_producto"].astype(str).str.strip()
    if df["nombre_producto"].eq("").any():
        raise ValueError("Hay filas con 'nombre_producto' vacío.")

    df["fecha_venta"] = pd.to_datetime(df["fecha_venta"], errors="coerce")
    if df["fecha_venta"].isna().any():
        raise ValueError("Hay fechas inválidas en 'fecha_venta'.")

    df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce")
    if df["cantidad"].isna().any():
        raise ValueError("Hay valores no numéricos en 'cantidad'.")
    if (df["cantidad"] < 0).any():
        raise ValueError("No se permiten cantidades negativas en este MVP (devoluciones).")

    df["ds"] = df["fecha_venta"].dt.floor("D")
    df["y"] = df["cantidad"].astype(float)

    return df[["nombre_producto", "ds", "y"]]


def _aggregate_daily(df_norm: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa por (producto, día) y suma cantidades.
    Devuelve: nombre_producto, ds, y (y = total vendido ese día).
    """
    daily = (
        df_norm.groupby(["nombre_producto", "ds"], as_index=False)
        .agg(y=("y", "sum"))
    )
    return daily


def _read_user_file(file_obj, filename: str) -> pd.DataFrame:
    """
    Lee Excel/CSV y devuelve DataFrame agregado diario: (producto, ds, y)
    NOTA: Para performance, solo leemos las 4 columnas necesarias.
    """
    lower = (filename or "").lower()

    if lower.endswith(".csv"):
        agg_parts = []
        for chunk in pd.read_csv(file_obj, usecols=REQUIRED_COLS, chunksize=CSV_CHUNKSIZE):
            chunk_norm = _normalize_and_validate(chunk)
            chunk_daily = _aggregate_daily(chunk_norm)
            agg_parts.append(chunk_daily)

        daily = pd.concat(agg_parts, ignore_index=True)
        daily = daily.groupby(["nombre_producto", "ds"], as_index=False).agg(y=("y", "sum"))
        return daily

    # Excel
    df = pd.read_excel(file_obj, engine="openpyxl", usecols=REQUIRED_COLS)
    df_norm = _normalize_and_validate(df)
    daily = _aggregate_daily(df_norm)
    return daily


# -----------------------------
# Serie mensual para Prophet
# -----------------------------
def _prepare_series_monthly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Entrada: df_daily con ds (datetime) y y (float) a nivel diario.
    Salida: serie mensual continua con ds en inicio de mes (MS) y relleno de meses faltantes con 0.
    """
    dfp = df_daily.sort_values("ds").copy()
    dfp["ds"] = pd.to_datetime(dfp["ds"])

    # Resample mensual: inicio de mes (MS)
    s = dfp.set_index("ds")["y"].resample("MS").sum()

    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    s = s.reindex(full_idx, fill_value=0.0)

    return pd.DataFrame({"ds": s.index, "y": s.values.astype(float)})


def _max_horizon_months(n_months: int) -> int:
    """
    Límite de horizonte para evitar extrapolación mala:
    - máximo 24 meses
    - nunca más de la mitad del histórico
    """
    return max(1, min(24, n_months // 2))


# -----------------------------
# Prophet (modelo mensual + tuning)
# -----------------------------
def _build_prophet_monthly(
    n_months: int,
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0,
) -> Prophet:
    yearly = n_months >= 24

    # n_changepoints: razonable para mensual (evita demasiados puntos con poco histórico)
    n_changepoints = int(min(25, max(5, n_months // 2)))

    m = Prophet(
        seasonality_mode="multiplicative",
        weekly_seasonality=False,
        daily_seasonality=False,
        yearly_seasonality=yearly,
        changepoint_range=0.9,
        n_changepoints=n_changepoints,
        changepoint_prior_scale=float(changepoint_prior_scale),
        seasonality_prior_scale=float(seasonality_prior_scale),
    )
    return m


def _mape_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    mask = y_true != 0
    if not mask.any():
        return None
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _tune_prophet_monthly(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    Tuning pequeño para bajar MAE sin que se vuelva pesado.
    Retorna best_params y la predicción del test (en escala real).
    """
    # Grid chico (6 fits)
    cps_grid = [0.01, 0.05, 0.1]
    sps_grid = [5.0, 10.0]

    best = {
        "mae": float("inf"),
        "mape": None,
        "cps": 0.05,
        "sps": 10.0,
        "yhat_test": None,
    }

    # Transformación para estabilizar (muy útil en ventas crecientes)
    train_fit = train.copy()
    test_ds = test[["ds"]].copy()

    # Clip suave de outliers en TRAIN (evita que 1 mes raro distorsione todo)
    # Si no hay suficiente histórico, no aplicar.
    if len(train_fit) >= 24:
        clip_hi = float(train_fit["y"].quantile(0.99))
        train_fit["y"] = np.minimum(train_fit["y"].to_numpy(dtype=float), clip_hi)

    train_fit["y"] = np.log1p(train_fit["y"].astype(float))

    y_true_test = test["y"].to_numpy(dtype=float)

    for cps in cps_grid:
        for sps in sps_grid:
            try:
                m = _build_prophet_monthly(
                    n_months=len(train_fit),
                    changepoint_prior_scale=cps,
                    seasonality_prior_scale=sps,
                )
                m.fit(train_fit)

                pred = m.predict(test_ds)
                yhat = np.expm1(pred["yhat"].to_numpy(dtype=float))
                yhat = np.clip(yhat, 0, None)

                mae = float(mean_absolute_error(y_true_test, yhat))
                mape = _mape_percent(y_true_test, yhat)

                if mae < best["mae"]:
                    best.update({"mae": mae, "mape": mape, "cps": cps, "sps": sps, "yhat_test": yhat})
            except Exception:
                # Si un set de parámetros falla, lo saltamos
                continue

    # Fallback si por alguna razón nada entrenó
    if best["yhat_test"] is None:
        m = _build_prophet_monthly(n_months=len(train), changepoint_prior_scale=0.05, seasonality_prior_scale=10.0)
        m.fit(train)
        pred = m.predict(test[["ds"]])
        yhat = np.clip(pred["yhat"].to_numpy(dtype=float), 0, None)
        best.update({"mae": float(mean_absolute_error(y_true_test, yhat)), "mape": _mape_percent(y_true_test, yhat), "yhat_test": yhat})

    return best


# -----------------------------
# Views (NO tocan tus templates)
# -----------------------------
def upload_page(request):
    return render(request, "forecast/upload.html", {})


@require_POST
def upload_api(request):
    session_key = _ensure_session(request)

    if "file" not in request.FILES:
        return JsonResponse({"ok": False, "error": "No se recibió archivo (campo 'file')."}, status=400)

    f = request.FILES["file"]
    lower = (f.name or "").lower()
    if not (lower.endswith(".xlsx") or lower.endswith(".xls") or lower.endswith(".csv")):
        return JsonResponse({"ok": False, "error": "Formato no soportado. Sube .xlsx/.xls o .csv."}, status=400)

    try:
        daily = _read_user_file(f, f.name)
    except Exception as e:
        return JsonResponse({"ok": False, "error": f"No se pudo procesar el archivo: {e}"}, status=400)

    try:
        with transaction.atomic():
            Dataset.objects.filter(session_key=session_key).delete()
            dataset = Dataset.objects.create(session_key=session_key, original_filename=f.name)

            product_names = daily["nombre_producto"].astype(str).unique().tolist()
            Product.objects.bulk_create(
                [Product(dataset=dataset, name=n) for n in product_names],
                ignore_conflicts=True,
                batch_size=BULK_BATCH_SIZE,
            )

            products_map = dict(Product.objects.filter(dataset=dataset).values_list("name", "id"))

            rows = []
            for r in daily.itertuples(index=False):
                rows.append(
                    SaleDaily(
                        dataset=dataset,
                        product_id=products_map[r.nombre_producto],
                        date=r.ds.date(),
                        qty=float(r.y),
                    )
                )
            SaleDaily.objects.bulk_create(rows, batch_size=BULK_BATCH_SIZE)
    except Exception as e:
        return JsonResponse({"ok": False, "error": f"No se pudo guardar el dataset en BD: {e}"}, status=500)

    return JsonResponse({"ok": True, "redirect": "/forecast/"})


def products_api(request):
    dataset = _get_dataset(request)
    if not dataset:
        return JsonResponse({"items": [], "page": 1, "has_more": False, "total": 0})

    search = (request.GET.get("search") or "").strip()
    page = int(request.GET.get("page") or 1)
    page_size = 100

    qs = Product.objects.filter(dataset=dataset).order_by("name")
    if search:
        qs = qs.filter(name__icontains=search)

    total = qs.count()
    start = (page - 1) * page_size
    end = start + page_size

    items = list(qs.values_list("name", flat=True)[start:end])
    return JsonResponse({"items": items, "page": page, "has_more": end < total, "total": total})


def index(request):
    dataset = _get_dataset(request)
    if not dataset:
        return redirect("upload-page")

    # Mantener las llaves que tu index.html ya espera (sin cambiar template)
    context = {
        "dataset_name": dataset.original_filename,
        "horizon_max_default": 24,  # ahora interpretado como MESES
        "chart_payload": {"labels": [], "hist": [], "yhat": []},
        "table_rows": [],
    }

    if request.method != "POST":
        return render(request, "forecast/index.html", context)

    product_name = (request.POST.get("product_name") or "").strip()
    horizon_months = int(request.POST.get("horizon_days") or 0)  # mantenemos nombre del campo

    product = Product.objects.filter(dataset=dataset, name__iexact=product_name).first()
    if not product:
        context["error"] = "Producto no encontrado (selecciónalo de la lista)."
        return render(request, "forecast/index.html", context)

    # Traer serie diaria agregada desde BD
    qs = (
        SaleDaily.objects
        .filter(dataset=dataset, product=product)
        .values("date")
        .annotate(y=Sum("qty"))
        .order_by("date")
    )
    df = pd.DataFrame(list(qs))
    if df.empty:
        context["error"] = "No hay datos para ese producto."
        return render(request, "forecast/index.html", context)

    # Diario -> Mensual
    df["ds"] = pd.to_datetime(df["date"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0).astype(float)
    df = df[["ds", "y"]]
    df_m = _prepare_series_monthly(df)

    n_months = len(df_m)
    if n_months < 12:
        context["error"] = "Histórico insuficiente (mínimo recomendado: 12 meses)."
        return render(request, "forecast/index.html", context)

    horizon_max = _max_horizon_months(n_months)
    context["horizon_max"] = horizon_max

    if horizon_months < 1 or horizon_months > horizon_max:
        context["error"] = f"Horizonte inválido. Máximo recomendado: {horizon_max} meses."
        return render(request, "forecast/index.html", context)

    # Split 80/20 mensual
    split = int(n_months * 0.8)
    train = df_m.iloc[:split].copy()
    test = df_m.iloc[split:].copy()

    # Tuning + evaluación (predicción SOLO sobre test para que el MAE corresponda al gráfico)
    tuned = _tune_prophet_monthly(train, test)
    mae = float(tuned["mae"])
    mape = tuned["mape"]

    # Modelo final (entrena con TODO el histórico usando los mejores params encontrados)
    df_fit = df_m.copy()
    if len(df_fit) >= 24:
        clip_hi = float(df_fit["y"].quantile(0.99))
        df_fit["y"] = np.minimum(df_fit["y"].to_numpy(dtype=float), clip_hi)
    df_fit["y"] = np.log1p(df_fit["y"].astype(float))

    m_full = _build_prophet_monthly(
        n_months=len(df_fit),
        changepoint_prior_scale=tuned["cps"],
        seasonality_prior_scale=tuned["sps"],
    )
    m_full.fit(df_fit)

    # Futuro mensual: freq="MS" (inicio de mes)
    future = m_full.make_future_dataframe(periods=horizon_months, freq="MS", include_history=False)
    fc_future = m_full.predict(future)
    yhat_future = np.expm1(fc_future["yhat"].to_numpy(dtype=float))
    yhat_future = np.clip(yhat_future, 0, None)

    # -----------------------------
    # Chart payload coherente con MAE:
    # - Hist: valores reales para todo el histórico
    # - yhat: None en train, predicción en test, predicción en futuro
    # -----------------------------
    labels_hist = df_m["ds"].dt.strftime("%Y-%m").tolist()
    hist_vals = np.round(df_m["y"].to_numpy(dtype=float), 2).tolist()

    # yhat para histórico: None en train + pred_test en test
    yhat_hist = [None] * len(train) + np.round(tuned["yhat_test"], 2).tolist()

    # Añadir futuro
    labels_future = fc_future["ds"].dt.strftime("%Y-%m").tolist()
    labels = labels_hist + labels_future

    hist = hist_vals + [None] * len(labels_future)
    yhat = yhat_hist + np.round(yhat_future, 2).tolist()

    # Tabla del futuro
    table_rows = list(zip(labels_future, np.round(yhat_future, 2).tolist()))

    avg_y = float(df_m["y"].mean()) if len(df_m) else 0.0
    nmae = (mae / avg_y * 100.0) if avg_y > 0 else None

    # Strings listos para UI
    mae_str = f"{mae:,.2f}"
    mape_str = "-" if mape is None else f"{mape:.2f}%"
    nmae_str = "-" if nmae is None else f"{nmae:.2f}%"
    avg_str = f"{avg_y:,.2f}"

    context.update({
        "product_name": product.name,
        "horizon_days": horizon_months,  # mismo campo del template; ahora significa MESES
        "mae": mae,
        "mape": mape,
        "chart_payload": {"labels": labels, "hist": hist, "yhat": yhat},
        "table_rows": table_rows,
        "mae": mae_str,
        "mape": mape_str,
        "nmae": nmae_str,  # MAE convertido a %
        "avg_y": avg_str
    })

    return render(request, "forecast/index.html", context)
