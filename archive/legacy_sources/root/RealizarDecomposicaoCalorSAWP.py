"""Medias AMJJ sem tendencia por quantis de SST da SAWP (base 1980-2010).

Calcula os termos mensais primeiro e remove a tendencia linear de cada
termo em cada ponto, preservando a media 1959-2023. A classificacao utiliza
o mesmo recorte, SST detrended e funcao do projeto original.
Dependencias cientificas: numpy, pandas, scipy, h5py.
"""
from pathlib import Path
import argparse
import ast
import json
import h5py
import numpy as np
import pandas as pd
from scipy.io import netcdf_file


def detrend_mean_preserved(a):
    assert np.isfinite(a).all()
    t = np.arange(len(a), dtype=float)
    t -= t.mean()
    slope = np.einsum('t,tij->ij', t, a - a.mean(axis=0)) / (t @ t)
    result = a - t[:, None, None] * slope
    assert np.max(np.abs(result.mean(axis=0) - a.mean(axis=0))) < 1e-9
    residual = np.einsum('t,tij->ij', t, result - result.mean(axis=0)) / (t @ t)
    assert np.max(np.abs(residual)) < 1e-10
    return result


def specific_humidity(temp_k, pressure_pa):
    # ECMWF IFS, saturation over water (Buck): T0=273.16 K.
    e = 611.21 * np.exp(17.502 * (temp_k - 273.16) / (temp_k - 32.19))
    epsilon = 287.0597 / 461.5250
    return epsilon * e / (pressure_pa - (1 - epsilon) * e)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir.parent if script_dir.name == '__pycache__' else script_dir
    parser.add_argument('--base-dir', type=Path, default=default_root)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    root = args.base_dir.resolve()
    out = (args.output_dir or root/'outputs/DecomposicaoCalorSAWP').resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Executa somente a funcao de classificacao, sem importar cartopy/metpy.
    path = root/'RealizarComparacaoQuantisTransporteUmidade.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'classifications')
    namespace = {'np': np, 'pd': pd}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    classification, groups, thresholds = namespace['classifications'](root)
    # Reproduz tambem a funcao usada diretamente por RealizarAnalises.py.
    tree2 = ast.parse((root/'sawp/ProcessamentoTsmQuantisv2.py').read_text(encoding='utf-8'))
    original_function = next(n for n in tree2.body if isinstance(n, ast.FunctionDef) and n.name == 'ProcessarTsmv2')
    exec(compile(ast.Module(body=[original_function], type_ignores=[]), '<ProcessarTsmv2>', 'exec'), namespace)
    with np.load(root/'outputs/detrended_sst.npz') as z:
        lat, lon = z['lat'], z['lon']
        dates = pd.DatetimeIndex(z['time'])
        ys = slice(int(np.argmin(abs(lat+4))), int(np.argmin(abs(lat+12.5))))
        xs = slice(int(np.argmin(abs(lon+34))), int(np.argmin(abs(lon+25))))
        sst_detrended = z['data'][:, ys, xs]
        # Pequeno array recortado para a funcao original.
        reference = namespace['ProcessarTsmv2'](sst_detrended, dates, 0, sst_detrended.shape[2], sst_detrended.shape[1], 0)
    months = dates.to_period('M')
    assert months.equals(pd.period_range('1959-01', '2023-12', freq='M'))
    for g in ['75', '90']:
        assert set(dates[reference['percentis_tsm'][g]].year) == set(groups['original'][g])
    classification.to_csv(out/'ClassificacaoAnos.csv', encoding='utf-8-sig')

    with h5py.File(root/'data/Era5_Dataset_decomposit.nc') as f:
        np.testing.assert_array_equal(f['latitude'][:], lat)
        np.testing.assert_array_equal(f['longitude'][:], lon)
        dt = pd.to_datetime(f['valid_time'][:], unit='s').to_period('M')
        assert dt.equals(months)
        surface = {v: f[v][:, ys, xs].astype(float) for v in ['u10', 'v10', 't2m', 'd2m', 'sp']}
    with netcdf_file(root/'data/Era5_Dataset_sst.nc', mmap=True) as f:
        np.testing.assert_array_equal(f.variables['latitude'].data, lat)
        np.testing.assert_array_equal(f.variables['longitude'].data, lon)
        dt = (pd.Timestamp('1900-01-01') + pd.to_timedelta(f.variables['time'].data.copy(), unit='h')).to_period('M')
        assert dt.equals(months)
        v = f.variables['sst']
        a = v.data[:, ys, xs].copy()
        assert not (a == v._FillValue).any()
        sst = a.astype(float) * float(v.scale_factor) + float(v.add_offset)
        del v, a
    with h5py.File(root/'data/Era5_Dataset_lhsh.nc') as f:
        assert 'moda' in str(f.attrs['history'])
        np.testing.assert_array_equal(f['latitude'][:], lat)
        np.testing.assert_array_equal(f['longitude'][:], lon)
        assert pd.to_datetime(f['valid_time'][:], unit='s').to_period('M').equals(months)
        # Monthly means of daily accumulations: divide by 86400, upward positive.
        flux = {label: -f[var][:, ys, xs].astype(float)/86400
                for label, var in [('LHF_W_m2', 'slhf'), ('SHF_W_m2', 'sshf')]}
    np.testing.assert_allclose(detrend_mean_preserved(sst), sst_detrended, atol=1e-8, rtol=0)
    assert all(np.isfinite(a).all() for a in surface.values())
    assert np.all(surface['d2m'] <= surface['t2m'] + .2)
    q2 = specific_humidity(surface['d2m'], surface['sp'])
    qs_fresh = specific_humidity(sst, surface['sp'])
    # Conventional 2% reduction of saturation specific humidity for seawater.
    qs = .98 * qs_fresh
    raw = dict(flux, U10_m_s=np.hypot(surface['u10'], surface['v10']),
               delta_q_g_kg=(qs-q2)*1000, delta_T_K=sst-surface['t2m'],
               qs_g_kg=qs*1000, q2_g_kg=q2*1000,
               delta_q_sem_correcao_salina_g_kg=(qs_fresh-q2)*1000,
               SST_K=sst, T2m_K=surface['t2m'])
    # Calcula termos nao lineares antes do detrend; nao detrenda duas vezes.
    fields = {k: detrend_mean_preserved(v) for k, v in raw.items()}
    np.testing.assert_allclose(fields['delta_q_g_kg'], fields['qs_g_kg']-fields['q2_g_kg'], atol=1e-11)
    np.testing.assert_allclose(fields['delta_T_K'], fields['SST_K']-fields['T2m_K'], atol=1e-10)
    monthly = pd.DataFrame({k: v.mean(axis=(1,2)) for k, v in fields.items()}, index=dates)
    monthly.index.name = 'data'
    monthly.to_csv(out/'MediasMensaisSAWP_sem_tendencia.csv', encoding='utf-8-sig')
    summaries = []
    used = []
    for g, label in [('75', 'Moderately warm'), ('90', 'Extremely warm')]:
        years = groups['original'][g]
        mask = dates.year.isin(years) & dates.month.isin([4,5,6,7])
        assert mask.sum() == len(years)*4
        summary = dict(composite=label, n_anos=len(years), n_meses=int(mask.sum()))
        summary.update({k: float(monthly.loc[mask, k].mean()) for k in fields})
        # Media direta em espaco e tempo confere com media das medias mensais.
        for k in fields:
            np.testing.assert_allclose(summary[k], fields[k][mask].mean(), atol=1e-10)
        summaries.append(summary)
        used.extend(dict(composite=label, data=str(d.date())) for d in dates[mask])
    pd.DataFrame(summaries).to_csv(out/'TabelaCalorSAWP.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(used).to_csv(out/'MesesUtilizados.csv', index=False, encoding='utf-8-sig')
    report = dict(
        periodo='1959-2023', meses=[4,5,6,7], base_quantis='1980-2010',
        quantis=thresholds['original'], anos={g: groups['original'][g].tolist() for g in ['75','90']},
        area='-12.5 < latitude <= -4; -34 <= longitude < -25 (slices do projeto)',
        grade=dict(nlat=len(lat[ys]), nlon=len(lon[xs]), latitude=[float(lat[ys].min()),float(lat[ys].max())],longitude=[float(lon[xs].min()),float(lon[xs].max())]),
        media='Aritmetica espacial, meses e anos com pesos iguais, como no projeto',
        tendencia='Linear por ponto, todos os 780 meses, mantendo media. Termos derivados calculados antes do detrend.',
        umidade='Buck/IFS sobre agua; epsilon=287.0597/461.5250; qs=0.98*qsat(SST,sp); q2=qsat(d2m,sp)',
        fluxos='LHF=-slhf/86400; SHF=-sshf/86400; W/m2 positivos para cima',
        limites='Termos estimados a partir de medias mensais; U10 e modulo do vento medio mensal. Tabela diagnostica, nao fechamento exato da formula bulk.',
        verificacoes='Datas, grades, finitude, medias preservadas, tendencias nulas, quantis iguais ao script original, medias diretas conferidas',
        tabela=summaries)
    (out/'MetodologiaResultados.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
