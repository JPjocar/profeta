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

REQUIRED_COLS = ["id_venta", "nombre_producto", "fecha_venta", "cantidad"]

# Solo en caso de CSV (funciona pero no utilizarlo)
CSV_CHUNKSIZE = 200_000
BULK_BATCH_SIZE = 5000


# Create session or get last session
def _ensure_session(request) -> str:
    if not request.session.session_key:
        request.session.create()
    return request.session.session_key

# Get dataset
def _get_dataset(request) -> Dataset | None:
    session_key = _ensure_session(request)
    return Dataset.objects.filter(session_key=session_key).order_by("-created_at").first()


# Clean data and validate
def _normalize_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}. Requeridas: {REQUIRED_COLS}")

    df = df[REQUIRED_COLS].copy()

    df["nombre_producto"] = df["nombre_producto"].astype(str).str.strip()
    if df["nombre_producto"].eq("").any():
        raise ValueError("Hay filas con 'nombre_producto' vacío.")

    df["fecha_venta"] = pd.to_datetime(df["fecha_venta"], errors="coerce", dayfirst=True)
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


# Agrupamos por nombre y sello de fecha
def _aggregate_daily(df_norm: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df_norm.groupby(["nombre_producto", "ds"], as_index=False)
        .agg(y=("y", "sum"))
    )
    return daily


def _read_user_file(file_obj, filename: str) -> pd.DataFrame:

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


# Prophet personalizado para daily segun la cantidad de datos
def build_prophet(n_days: int) -> Prophet:
    weekly = n_days >= 14
    yearly = n_days >= 365

    # Parámetros conservadores por defecto
    m = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        changepoint_range=0.9,
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=5,
        seasonality_mode="multiplicative",
    )

    if n_days >= 90:
        m.add_seasonality(name="monthly", period=30.5, fourier_order=5)

    return m


def _max_horizon_days(n_days: int) -> int:
    # Max: 90 days, Min: No mas de la mitad
    return max(3, min(90, n_days // 2))


# Rellenar los dias faltantes con cero
def _prepare_series_daily(df_product: pd.DataFrame) -> pd.DataFrame:
    df_product = df_product.sort_values("ds").copy()

    # Obtener el rango completo diario
    full = pd.DataFrame({"ds": pd.date_range(df_product["ds"].min(), df_product["ds"].max(), freq="D")})
    out = full.merge(df_product[["ds", "y"]], on="ds", how="left")

    # Dias sin registro llenar con cero ventas
    out["y"] = pd.to_numeric(out["y"], errors="coerce").fillna(0.0).astype(float)
    return out


def _run_daily_forecast(request, dataset: Dataset, context: dict) -> dict:
    product_name = (request.POST.get("product_name") or "").strip()
    horizon = int(request.POST.get("horizon_days") or 0)

    product = Product.objects.filter(dataset=dataset, name__iexact=product_name).first()
    if not product:
        context["error"] = "Producto no encontrado(selecciona de la lista)"
        return context

    # Agrega la serie desde la DB
    qs = (
        SaleDaily.objects
        .filter(dataset=dataset, product=product)
        .values("date")
        .annotate(y=Sum("qty"))
        .order_by("date")
    )
    df = pd.DataFrame(list(qs))
    if df.empty:
        context["error"] = "No hay registros para ese producto"
        return context

    df["ds"] = pd.to_datetime(df["date"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0).astype(float)
    df = df[["ds", "y"]]
    df = _prepare_series_daily(df)

    n_days = len(df)
    if n_days < 14:
        context["error"] = "Historico insuficiente(minimo recomendado >= 14 días)."
        return context

    horizon_max = _max_horizon_days(n_days)
    context["horizon_max"] = horizon_max
    context["horizon_unit"] = "días"

    if horizon < 1 or horizon > horizon_max:
        context["error"] = f"Horizonte incorrecto. Maximo aceptable: {horizon_max} dias"
        return context

    # Split 80/20 Prophet
    split = int(n_days * 0.8)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    # Modelo para entrenamiento
    m = build_prophet(len(train))
    m.fit(train)

    # Predecir test
    pred_test = m.predict(test[["ds"]])
    mae = float(mean_absolute_error(test["y"].to_numpy(), pred_test["yhat"].to_numpy()))

    # Calcular MAPE
    y_true = test["y"].to_numpy(dtype=float)
    y_pred = pred_test["yhat"].to_numpy(dtype=float)
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0) if mask.any() else None

    # Modelo utilizando el historico completo
    m2 = build_prophet(len(df))
    m2.fit(df)

    future = m2.make_future_dataframe(periods=horizon, freq="D", include_history=True)
    forecast = m2.predict(future)

    labels = forecast["ds"].dt.strftime("%Y-%m-%d").tolist()
    yhat = forecast["yhat"].clip(lower=0).round(2).tolist()

    hist_map = dict(zip(df["ds"].dt.strftime("%Y-%m-%d"), df["y"].round(2)))
    hist = [hist_map.get(d, None) for d in labels]

    future_table = forecast.tail(horizon).copy()
    table_rows = list(zip(
        future_table["ds"].dt.strftime("%Y-%m-%d").tolist(),
        future_table["yhat"].clip(lower=0).round(2).tolist()
    ))

    # Dar formato mensual de fecha
    avg_y = float(df["y"].mean()) if len(df) else 0.0
    nmae = (mae / avg_y * 100.0) if avg_y > 0 else None

    context.update({
        "product_name": product.name,
        "horizon_days": horizon,
        "mae": f"{mae:,.2f}",
        "mape": "-" if mape is None else f"{mape:.2f}%",
        "nmae": "-" if nmae is None else f"{nmae:.2f}%",
        "avg_y": f"{avg_y:,.2f}",
        "chart_payload": {"labels": labels, "hist": hist, "yhat": yhat},
        "table_rows": table_rows,
    })

    return context


# Convert daily to monthly
def _prepare_series_monthly(df_daily: pd.DataFrame) -> pd.DataFrame:

    dfp = df_daily.sort_values("ds").copy()
    dfp["ds"] = pd.to_datetime(dfp["ds"])

    s = dfp.set_index("ds")["y"].resample("MS").sum()

    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    s = s.reindex(full_idx, fill_value=0.0)

    return pd.DataFrame({"ds": s.index, "y": s.values.astype(float)})


def _max_horizon_months(n_months: int) -> int:
    # Max: 24 months Min: No mas de la mitad de los datos
    return max(1, min(24, n_months // 2))


def _build_prophet_monthly(n_months: int, changepoint_prior_scale: float = 0.05, seasonality_prior_scale: float = 10.0,) -> Prophet:
    #Establecer True si hay registros de años
    yearly = n_months >= 24

    #Best cantidad de changepoints
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


# Calculo del mape percent
def _mape_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    mask = y_true != 0
    if not mask.any():
        return None
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _tune_prophet_monthly(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    # Hallar los mejores parametros para el modelo
    cps_grid = [0.01, 0.05, 0.1]
    sps_grid = [5.0, 10.0]

    best = {
        "mae": float("inf"),
        "mape": None,
        "cps": 0.05,
        "sps": 10.0,
        "yhat_test": None,
    }

    train_fit = train.copy()
    test_ds = test[["ds"]].copy()

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
                continue

    if best["yhat_test"] is None:
        m = _build_prophet_monthly(
            n_months=len(train),
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        m.fit(train)
        pred = m.predict(test[["ds"]])
        yhat = np.clip(pred["yhat"].to_numpy(dtype=float), 0, None)
        best.update({
            "mae": float(mean_absolute_error(y_true_test, yhat)),
            "mape": _mape_percent(y_true_test, yhat),
            "yhat_test": yhat
        })

    return best


def _run_monthly_forecast(request, dataset: Dataset, context: dict) -> dict:
    product_name = (request.POST.get("product_name") or "").strip()
    horizon_months = int(request.POST.get("horizon_days") or 0)

    product = Product.objects.filter(dataset=dataset, name__iexact=product_name).first()
    if not product:
        context["error"] = "Producto no encontrado en lista"
        return context

    qs = (
        SaleDaily.objects
        .filter(dataset=dataset, product=product)
        .values("date")
        .annotate(y=Sum("qty"))
        .order_by("date")
    )
    df = pd.DataFrame(list(qs))
    if df.empty:
        context["error"] = "No hay datos para el producto seleccionado"
        return context

    df["ds"] = pd.to_datetime(df["date"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0).astype(float)
    df = df[["ds", "y"]]
    df_m = _prepare_series_monthly(df)

    n_months = len(df_m)
    if n_months < 12:
        context["error"] = "Historico insuficiente(minimo recomendado >= 12 meses)"
        return context

    # Show max horizon months to user
    horizon_max = _max_horizon_months(n_months)
    context["horizon_max"] = horizon_max
    context["horizon_unit"] = "meses"

    if horizon_months < 1 or horizon_months > horizon_max:
        context["error"] = f"Horizonte incorrecto. Maximo es: {horizon_max} meses."
        return context

    split = int(n_months * 0.8)
    train = df_m.iloc[:split].copy()
    test = df_m.iloc[split:].copy()

    tuned = _tune_prophet_monthly(train, test)
    mae = float(tuned["mae"])
    mape = tuned["mape"]

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

    future = m_full.make_future_dataframe(periods=horizon_months, freq="MS", include_history=False)
    fc_future = m_full.predict(future)
    yhat_future = np.expm1(fc_future["yhat"].to_numpy(dtype=float))
    yhat_future = np.clip(yhat_future, 0, None)

    labels_hist = df_m["ds"].dt.strftime("%Y-%m").tolist()
    hist_vals = np.round(df_m["y"].to_numpy(dtype=float), 2).tolist()

    yhat_hist = [None] * len(train) + np.round(tuned["yhat_test"], 2).tolist()

    labels_future = fc_future["ds"].dt.strftime("%Y-%m").tolist()
    labels = labels_hist + labels_future

    hist = hist_vals + [None] * len(labels_future)
    yhat = yhat_hist + np.round(yhat_future, 2).tolist()

    table_rows = list(zip(labels_future, np.round(yhat_future, 2).tolist()))

    avg_y = float(df_m["y"].mean()) if len(df_m) else 0.0
    nmae = (mae / avg_y * 100.0) if avg_y > 0 else None

    context.update({
        "product_name": product.name,
        "horizon_days": horizon_months,
        "mae": f"{mae:,.2f}",
        "mape": "-" if mape is None else f"{mape:.2f}%",
        "nmae": "-" if nmae is None else f"{nmae:.2f}%",
        "avg_y": f"{avg_y:,.2f}",
        "chart_payload": {"labels": labels, "hist": hist, "yhat": yhat},
        "table_rows": table_rows,
    })
    return context


# Upload page
def upload_page(request):
    return render(request, "forecast/upload.html", {})


#Verify fields in colums
@require_POST
def upload_api(request):
    session_key = _ensure_session(request)

    if "file" not in request.FILES:
        return JsonResponse({"ok": False, "error": "No se recibio el archivo"}, status=400)

    f = request.FILES["file"]
    lower = (f.name or "").lower()
    if not (lower.endswith(".xlsx") or lower.endswith(".xls") or lower.endswith(".csv")):
        return JsonResponse({"ok": False, "error": "Formato NO soportado. Intenta con .xlsx/.xls o .csv."}, status=400)

    try:
        daily = _read_user_file(f, f.name)
    except Exception as e:
        return JsonResponse({"ok": False, "error": f"No se puede procesar el archivo: {e}"}, status=400)

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

    # Select el nivel de detalle (daily/monthly)
    granularity = (request.POST.get("granularity") or request.GET.get("granularity") or "daily").strip().lower()
    if granularity not in ("daily", "monthly"):
        granularity = "daily"

    context = {
        "dataset_name": dataset.original_filename,
        "chart_payload": {"labels": [], "hist": [], "yhat": []},
        "table_rows": [],
        "granularity": granularity,
        "horizon_max_default_daily": 90,
        "horizon_max_default_monthly": 24,
        "horizon_unit": "días" if granularity == "daily" else "meses",
        "horizon_max_default": 90 if granularity == "daily" else 24,
    }

    if request.method != "POST":
        return render(request, "forecast/index.html", context)

    if granularity == "monthly":
        context = _run_monthly_forecast(request, dataset, context)
    else:
        context = _run_daily_forecast(request, dataset, context)

    return render(request, "forecast/index.html", context)
