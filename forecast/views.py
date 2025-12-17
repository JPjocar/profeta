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

# Si el usuario sube CSV muy grande, leer en partes (evita cargar todo a memoria).
CSV_CHUNKSIZE = 200_000

# Inserciones masivas a BD: batches para no hacer 1 query gigante.
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
# Limpieza + agregación diaria
# -----------------------------
def _normalize_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe un DataFrame con las columnas requeridas.
    Devuelve un DataFrame limpio con columnas:
      - nombre_producto (str)
      - ds (datetime64[ns])  -> día (sin hora)
      - y  (float)           -> cantidad (sumable)
    """
    # Normalizar nombres de columnas
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}. Requeridas: {REQUIRED_COLS}")

    df = df[REQUIRED_COLS].copy()

    # Producto: string limpio
    df["nombre_producto"] = df["nombre_producto"].astype(str).str.strip()
    if df["nombre_producto"].eq("").any():
        raise ValueError("Hay filas con 'nombre_producto' vacío.")

    # Fecha: convertir a datetime
    df["fecha_venta"] = pd.to_datetime(df["fecha_venta"], errors="coerce")
    if df["fecha_venta"].isna().any():
        raise ValueError("Hay fechas inválidas en 'fecha_venta'.")

    # Cantidad: numérica
    df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce")
    if df["cantidad"].isna().any():
        raise ValueError("Hay valores no numéricos en 'cantidad'.")
    if (df["cantidad"] < 0).any():
        raise ValueError("No se permiten cantidades negativas en este MVP (devoluciones).")

    # Convertir a día (sin hora)
    df["ds"] = df["fecha_venta"].dt.floor("D")

    # Renombrar cantidad a y
    df["y"] = df["cantidad"].astype(float)

    return df[["nombre_producto", "ds", "y"]]


def _aggregate_daily(df_norm: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa por (producto, día) y suma cantidades.
    Devuelve: nombre_producto, ds, y (y = total vendido ese día).
    """
    daily = (
        df_norm
        .groupby(["nombre_producto", "ds"], as_index=False)
        .agg(y=("y", "sum"))
    )
    return daily


def _read_user_file(file_obj, filename: str) -> pd.DataFrame:
    """
    Lee Excel/CSV y devuelve DataFrame.
    NOTA: Para performance, solo leemos las 4 columnas necesarias.
    """
    lower = (filename or "").lower()

    if lower.endswith(".csv"):
        # Leer CSV completo (si no es enorme) o en chunks (si es grande).
        # Para poder hacer chunksize, primero intentamos chunksize siempre; luego agregamos incremental.
        agg_parts = []

        for chunk in pd.read_csv(file_obj, usecols=REQUIRED_COLS, chunksize=CSV_CHUNKSIZE):
            chunk_norm = _normalize_and_validate(chunk)
            chunk_daily = _aggregate_daily(chunk_norm)
            agg_parts.append(chunk_daily)

        # Unir agregados parciales y volver a agregar (porque un mismo producto/día puede aparecer en varios chunks)
        daily = pd.concat(agg_parts, ignore_index=True)
        daily = daily.groupby(["nombre_producto", "ds"], as_index=False).agg(y=("y", "sum"))
        return daily

    # Excel (xlsx/xls)
    # Para Excel no hay chunksize estándar como en read_csv; aquí minimizamos lectura con usecols.
    df = pd.read_excel(file_obj, engine="openpyxl", usecols=REQUIRED_COLS)
    df_norm = _normalize_and_validate(df)
    daily = _aggregate_daily(df_norm)
    return daily


# -----------------------------
# Prophet (modelo)
# -----------------------------
def build_prophet(n_days: int) -> Prophet:
    """
    Crea un Prophet apropiado según el tamaño del histórico (en días).
    - Con poco histórico: modelo conservador (evita sobreajuste).
    - Con más histórico: permite más componentes.
    """
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

    # “Mensual” aproximada solo si hay suficiente histórico (evita inventar ondas con pocos días)
    if n_days >= 90:
        m.add_seasonality(name="monthly", period=30.5, fourier_order=5)

    return m


def _max_horizon_days(n_days: int) -> int:
    """
    Límite de horizonte para evitar extrapolación mala.
    Regla: máximo 90 días, y nunca más de la mitad del histórico.
    """
    return max(3, min(90, n_days // 2))


def _prepare_series_daily(df_product: pd.DataFrame) -> pd.DataFrame:
    """
    Entrada: df_product con ds,y para un producto.
    Salida: serie diaria continua (rellena días faltantes con 0).
    """
    df_product = df_product.sort_values("ds").copy()

    # Rango completo diario
    full = pd.DataFrame({"ds": pd.date_range(df_product["ds"].min(), df_product["ds"].max(), freq="D")})
    out = full.merge(df_product[["ds", "y"]], on="ds", how="left")

    # Día sin registro = 0 ventas (retail)
    out["y"] = pd.to_numeric(out["y"], errors="coerce").fillna(0.0).astype(float)
    return out


# -----------------------------
# Views (NO tocan tus templates)
# -----------------------------
def upload_page(request):
    # Muestra la página principal (NoIntuition + botón subir).
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
        # Lee y devuelve DIRECTAMENTE el agregado diario: (producto, ds, y)
        daily = _read_user_file(f, f.name)
    except Exception as e:
        return JsonResponse({"ok": False, "error": f"No se pudo procesar el archivo: {e}"}, status=400)

    # Guardar en BD solo lo agregado (mucho más pequeño que 500k)
    try:
        with transaction.atomic():
            # Reemplazar dataset anterior de esta sesión
            Dataset.objects.filter(session_key=session_key).delete()
            dataset = Dataset.objects.create(session_key=session_key, original_filename=f.name)

            # Crear productos (únicos)
            product_names = daily["nombre_producto"].astype(str).unique().tolist()
            Product.objects.bulk_create(
                [Product(dataset=dataset, name=n) for n in product_names],
                ignore_conflicts=True,
                batch_size=BULK_BATCH_SIZE
            )
            products_map = dict(Product.objects.filter(dataset=dataset).values_list("name", "id"))

            # Crear ventas diarias agregadas
            rows = []
            for r in daily.itertuples(index=False):
                rows.append(SaleDaily(
                    dataset=dataset,
                    product_id=products_map[r.nombre_producto],
                    date=r.ds.date(),
                    qty=float(r.y),
                ))

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

    # Mantener las llaves que tu index.html ya espera (sin cambiar el template)
    context = {
        "dataset_name": dataset.original_filename,
        "horizon_max_default": 90,
        "chart_payload": {"labels": [], "hist": [], "yhat": []},
        "table_rows": [],
    }

    if request.method == "POST":
        product_name = (request.POST.get("product_name") or "").strip()
        horizon = int(request.POST.get("horizon_days") or 0)

        product = Product.objects.filter(dataset=dataset, name__iexact=product_name).first()
        if not product:
            context["error"] = "Producto no encontrado (selecciónalo de la lista)."
            return render(request, "forecast/index.html", context)

        # Traer la serie agregada diaria desde BD (esto ya es pequeño)
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

        # Convertir a ds/y y completar días faltantes
        df["ds"] = pd.to_datetime(df["date"])
        df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0).astype(float)
        df = df[["ds", "y"]]
        df = _prepare_series_daily(df)

        n_days = len(df)
        if n_days < 14:
            context["error"] = "Histórico insuficiente (mínimo recomendado: 14 días)."
            return render(request, "forecast/index.html", context)

        horizon_max = _max_horizon_days(n_days)
        context["horizon_max"] = horizon_max

        if horizon < 1 or horizon > horizon_max:
            context["error"] = f"Horizonte inválido. Máximo recomendado: {horizon_max} días."
            return render(request, "forecast/index.html", context)

        # Split 80/20 temporal
        split = int(n_days * 0.8)
        train = df.iloc[:split].copy()
        test = df.iloc[split:].copy()

        # Entrenar modelo para evaluación (NUEVA instancia)
        m = build_prophet(len(train))
        m.fit(train)

        pred_test = m.predict(test[["ds"]])
        mae = float(mean_absolute_error(test["y"].to_numpy(), pred_test["yhat"].to_numpy()))

        # MAPE (ignora y_true == 0 para evitar división por cero)
        y_true = test["y"].to_numpy(dtype=float)
        y_pred = pred_test["yhat"].to_numpy(dtype=float)
        mask = y_true != 0
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0) if mask.any() else None

        # Entrenar modelo final con todo el histórico (NUEVA instancia)
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

        context.update({
            "product_name": product.name,
            "horizon_days": horizon,
            "mae": mae,
            "mape": mape,
            "chart_payload": {"labels": labels, "hist": hist, "yhat": yhat},
            "table_rows": table_rows,
        })

    return render(request, "forecast/index.html", context)
