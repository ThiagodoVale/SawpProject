
import numpy as np
import xarray as xr

def detrend_3d(da: xr.DataArray) -> xr.DataArray:
    """
    Remove tendência linear mantendo a climatologia.
    """
    Y = da.values  # shape (time, lat, lon)
    t = np.arange(Y.shape[0])

    # Vetorizado para nanmean e covariância sem warnings
    with np.errstate(invalid="ignore", divide="ignore"):
        Y_mean = np.nanmean(Y, axis=0)
        t_mean = np.nanmean(t)
        cov = np.nanmean((t[:, None, None] - t_mean) * (Y - Y_mean), axis=0)
        var_t = np.nanmean((t - t_mean)**2)

        # slope
        a = cov / var_t

        # intercept: média preservada
        b = Y_mean - a * t_mean

        # aplicar modelo linear
        trend = a[None, :, :] * t[:, None, None] + b[None, :, :]

        # remover tendência
        Y_dt = Y - trend + Y_mean

    return xr.DataArray(
        Y_dt,
        dims=da.dims,
        coords=da.coords,
        attrs=da.attrs
    )


def detrend_4d(da: xr.DataArray) -> xr.DataArray:
    """
    Remove tendência linear mantendo a climatologia (para 4D)
    dims: time, level, lat, lon
    """
    print(">>> DEBUG detrend_4d <<<")
    print("Dims:", da.dims)
    print("Shape:", da.shape)
    Y = da.values
    t = np.arange(Y.shape[0])

    with np.errstate(invalid="ignore", divide="ignore"):
        Y_mean = np.nanmean(Y, axis=0)
        t_mean = np.nanmean(t)
        cov = np.nanmean((t[:, None, None, None] - t_mean) * (Y - Y_mean), axis=0)
        var_t = np.nanmean((t - t_mean)**2)

        a = cov / var_t
        b = Y_mean - a * t_mean

        trend = a[None, ...] * t[:, None, None, None] + b[None, ...]
        Y_dt = Y - trend + Y_mean

    return xr.DataArray(
        Y_dt,
        dims=da.dims,
        coords=da.coords,
        attrs=da.attrs
    )

def detrend_4d_gpu(da: xr.DataArray) -> xr.DataArray:
    """
    Remove tendência linear mantendo a climatologia (versão GPU - CuPy)
    dims: time, level, lat, lon
    """
    import cupy as cp

    print(">>> DEBUG detrend_4d_gpu <<<")
    print("Dims:", da.dims)
    print("Shape:", da.shape)

    # --- Envia dados para GPU ---
    Y = cp.asarray(da.values)
    t = cp.arange(Y.shape[0], dtype=Y.dtype)

    with cp.errstate(invalid="ignore", divide="ignore"):
        # Médias
        Y_mean = cp.nanmean(Y, axis=0)
        t_mean = cp.nanmean(t)

        # Covariância
        cov = cp.nanmean(
            (t[:, None, None, None] - t_mean) * (Y - Y_mean),
            axis=0
        )

        # Variância de t
        var_t = cp.nanmean((t - t_mean)**2)

        # Coeficientes da regressão a*t + b
        a = cov / var_t
        b = Y_mean - a * t_mean

        # Tendência ao longo do tempo
        trend = a[None, ...] * t[:, None, None, None] + b[None, ...]

        # Detrend + manutenção da climatologia
        Y_dt = Y - trend + Y_mean

    # --- Traz de volta para CPU para criar o DataArray ---
    Y_dt_cpu = cp.asnumpy(Y_dt)

    return xr.DataArray(
        Y_dt_cpu,
        dims=da.dims,
        coords=da.coords,
        attrs=da.attrs
    )

# =============================================================================
# import numpy as np
# import xarray as xr
# 
# 
# def _get_time_dim(da: xr.DataArray) -> str:
#     if "time" in da.dims:
#         return "time"
#     if "valid_time" in da.dims:
#         return "valid_time"
#     raise ValueError(f"Nenhuma dimensão temporal encontrada em {da.dims}")
# 
# 
# def _clim_mask(da: xr.DataArray, clim_start=1980, clim_end=2010):
#     time_dim = _get_time_dim(da)
# 
#     years = da[time_dim].dt.year.values
#     mask = (years >= clim_start) & (years <= clim_end)
# 
#     if not np.any(mask):
#         raise ValueError(
#             f"Nenhum dado encontrado no período climatológico "
#             f"{clim_start}-{clim_end}."
#         )
# 
#     return mask
# 
# 
# def detrend_3d(
#     da: xr.DataArray,
#     clim_start: int = 1959,
#     clim_end: int = 2023,
# ) -> xr.DataArray:
#     """
#     Remove tendência linear mantendo como referência a climatologia
#     do período clim_start-clim_end.
# 
#     Esperado:
#     dims: time/valid_time, latitude, longitude
#     """
# 
#     time_dim = _get_time_dim(da)
# 
#     Y = da.values
#     t = np.arange(Y.shape[0], dtype=float)
# 
#     clim = _clim_mask(da, clim_start, clim_end)
# 
#     with np.errstate(invalid="ignore", divide="ignore"):
#         # Média usada para calcular a tendência no período completo
#         Y_mean_full = np.nanmean(Y, axis=0)
# 
#         t_mean = np.nanmean(t)
# 
#         cov = np.nanmean(
#             (t[:, None, None] - t_mean) * (Y - Y_mean_full),
#             axis=0
#         )
# 
#         var_t = np.nanmean((t - t_mean) ** 2)
# 
#         a = cov / var_t
#         t_ref = np.nanmean(t[clim])
# 
#         trend_correction = a[None, :, :] * (
#             t[:, None, None] - t_ref
#         )
#         
#         Y_dt = Y - trend_correction
# 
#     return xr.DataArray(
#         Y_dt,
#         dims=da.dims,
#         coords=da.coords,
#         attrs=da.attrs,
#         name=da.name,
#     )
# 
# 
# def detrend_4d(
#     da: xr.DataArray,
#     clim_start: int = 1959,
#     clim_end: int = 2023,
# ) -> xr.DataArray:
#     """
#     Remove tendência linear mantendo como referência a climatologia
#     do período clim_start-clim_end.
# 
#     Esperado:
#     dims: time/valid_time, level/pressure_level, latitude, longitude
#     """
# 
#     time_dim = _get_time_dim(da)
# 
#     print(">>> DEBUG detrend_4d <<<")
#     print("Dims:", da.dims)
#     print("Shape:", da.shape)
#     print(f"Climatologia de referência: {clim_start}-{clim_end}")
# 
#     Y = da.values
#     t = np.arange(Y.shape[0], dtype=float)
# 
#     clim = _clim_mask(da, clim_start, clim_end)
# 
#     with np.errstate(invalid="ignore", divide="ignore"):
#         # Média do período completo apenas para ajuste da reta
#         Y_mean_full = np.nanmean(Y, axis=0)
# 
#         t_mean = np.nanmean(t)
# 
#         cov = np.nanmean(
#             (t[:, None, None, None] - t_mean) * (Y - Y_mean_full),
#             axis=0
#         )
# 
#         var_t = np.nanmean((t - t_mean) ** 2)
# 
#         a = cov / var_t
#         t_ref = np.nanmean(t[clim])
# 
#         trend_correction = a[None, ...] * (
#             t[:, None, None, None] - t_ref
#         )
#         
#         Y_dt = Y - trend_correction
# 
#     return xr.DataArray(
#         Y_dt,
#         dims=da.dims,
#         coords=da.coords,
#         attrs=da.attrs,
#         name=da.name,
#     )
# 
# 
# def detrend_4d_gpu(
#     da: xr.DataArray,
#     clim_start: int = 1980,
#     clim_end: int = 2010,
# ) -> xr.DataArray:
#     """
#     Remove tendência linear mantendo como referência a climatologia
#     do período clim_start-clim_end.
# 
#     Versão GPU com CuPy.
#     """
# 
#     import cupy as cp
# 
#     time_dim = _get_time_dim(da)
# 
#     print(">>> DEBUG detrend_4d_gpu <<<")
#     print("Dims:", da.dims)
#     print("Shape:", da.shape)
#     print(f"Climatologia de referência: {clim_start}-{clim_end}")
# 
#     clim = _clim_mask(da, clim_start, clim_end)
# 
#     Y = cp.asarray(da.values)
#     t = cp.arange(Y.shape[0], dtype=Y.dtype)
#     clim_cp = cp.asarray(clim)
# 
#     with cp.errstate(invalid="ignore", divide="ignore"):
#         Y_mean_full = cp.nanmean(Y, axis=0)
# 
#         t_mean = cp.nanmean(t)
# 
#         cov = cp.nanmean(
#             (t[:, None, None, None] - t_mean) * (Y - Y_mean_full),
#             axis=0
#         )
# 
#         var_t = cp.nanmean((t - t_mean) ** 2)
# 
#         a = cov / var_t
#         t_ref = cp.nanmean(t[clim_cp])
#         
#         trend_correction = a[None, ...] * (
#             t[:, None, None, None] - t_ref
#         )
#         
#         Y_dt = Y - trend_correction
# 
#     return xr.DataArray(
#         cp.asnumpy(Y_dt),
#         dims=da.dims,
#         coords=da.coords,
#         attrs=da.attrs,
#         name=da.name,
#     )
# =============================================================================
