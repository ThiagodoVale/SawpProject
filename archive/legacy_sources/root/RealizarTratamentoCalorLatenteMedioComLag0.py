#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quatro mapas de anomalias de LHF sem tendencia com lag de 0 meses.

TSM AMJJ classifica os anos usando quantis 1980-2010, aplicados a 1959-2023.
LHF usa ABRIL-JULHO dos mesmos anos, inclusive no composto normal.
ERA5 monthly means of daily means (moda): LHF = -avg_slhtf W/m2,
positivo do oceano para a atmosfera. Nao dividir pelo numero de dias do mes.
Remove tendencia linear de todos os meses de 1959-2023 preservando a media.

Executar na raiz do projeto:
    python RealizarTratamentoCalorLatenteMedioComLag0.py
Dependencias: numpy, pandas, h5py, matplotlib e cartopy.
Fonte da conversao: https://confluence.ecmwf.int/pages/viewpage.action?pageId=185081924
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
try:
    import h5py
except ImportError:
    h5py = None  # Apenas --somente-figura nao precisa do leitor NetCDF.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch

GROUPS = [('10', 'Extremely cold'), ('25', 'Moderately cold'),
          ('75', 'Moderately warm'), ('90', 'Extremely warm')]


def classify_sst(root):
    with np.load(root/'outputs/detrended_sst.npz') as z:
        dates = pd.DatetimeIndex(z['time'])
        lat, lon = z['lat'], z['lon']
        ys = slice(np.argmin(abs(lat+4)), np.argmin(abs(lat+12.5)))
        xs = slice(np.argmin(abs(lon+34)), np.argmin(abs(lon+25)))
        data = z['data'][:, ys, xs]
    assert dates.to_period('M').equals(pd.period_range('1959-01', '2023-12', freq='M'))
    assert np.isfinite(data).all() and data.min() > 200
    series = pd.Series(data.mean(axis=(1,2))-273.16, index=dates)
    series = series[series.index.month.isin([4,5,6,7])]
    climate = series.loc['1980':'2010'].groupby(series.loc['1980':'2010'].index.month).mean()
    annual = (series-series.index.month.map(climate)).groupby(series.index.year).mean()
    q10,q25,q75,q90 = annual.loc[1980:2010].quantile([.1,.25,.75,.9])
    labels = np.select([annual<=q10,annual<=q25,annual<q75,annual<q90],
                       ['10','25','normal','75'],default='90')
    table = pd.DataFrame({'anomalia_TSM_AMJJ_K':annual, 'grupo':labels})
    table.index.name = 'ano'
    groups = {g: annual.index[labels==g].to_numpy() for g in ['10','25','normal','75','90']}
    assert all(len(v) for v in groups.values())
    return table, groups, dict(zip(['P10','P25','P75','P90'],map(float,[q10,q25,q75,q90]))), lat, lon, ys, xs


def shifted_indices(months, years):
    # Alinhamento por calendario, nunca por indices posicionais arbitrarios.
    sst_months = pd.PeriodIndex([f'{y}-{m:02d}' for y in years for m in [4,5,6,7]], freq='M')
    lh_months = sst_months.copy()
    idx = months.get_indexer(lh_months)
    if (idx < 0).any():
        raise ValueError(f'Meses de LHF ausentes: {lh_months[idx < 0].tolist()}')
    assert np.array_equal(lh_months.month, np.tile([4,5,6,7],len(years)))
    assert np.array_equal(lh_months.year,sst_months.year)
    return idx, sst_months, lh_months


def main():
    p = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    p.add_argument('--base-dir',type=Path,default=here.parent if here.name=='__pycache__' else here)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--sem-figura',action='store_true',help='Calcula e salva os campos sem renderizar.')
    p.add_argument('--somente-figura',action='store_true',help='Renderiza os campos ja calculados na pasta de saida.')
    args = p.parse_args()
    root = args.base_dir.resolve()
    out = (args.output_dir or root/'figuras/CalorLatenteMedioComLag0').resolve()
    out.mkdir(parents=True,exist_ok=True)
    if args.somente_figura:
        draw(out)
        return
    if h5py is None: raise ImportError('Instale h5py para ler o NetCDF de LHF.')
    table, groups, thresholds, lat, lon, ys, xs = classify_sst(root)
    with h5py.File(root/'data/Era5_Dataset_meanHF.nc') as f:
        assert 'moda' in str(f.attrs['history']), 'Esperadas medias mensais.'
        assert f['avg_slhtf'].attrs['units'].decode() == 'W m**-2'
        np.testing.assert_array_equal(f['latitude'][:],lat)
        np.testing.assert_array_equal(f['longitude'][:],lon)
        months = pd.to_datetime(f['valid_time'][:],unit='s').to_period('M')
        assert months.equals(pd.period_range('1959-01','2023-12',freq='M'))
        raw = -f['avg_slhtf'][:].astype(float)
    if not np.isfinite(raw).all(): raise ValueError('Dados LHF ausentes; revisar mascara antes de detrendar.')
    t = np.arange(len(raw),dtype=float)
    t -= t.mean()
    mean = raw.mean(axis=0)
    slope = np.einsum('t,tij->ij',t,raw-mean)/(t@t)
    field = raw - t[:,None,None]*slope
    np.testing.assert_allclose(field.mean(axis=0),mean,atol=1e-10)
    assert np.max(abs(np.einsum('t,tij->ij',t,field-mean)/(t@t))) < 1e-10
    del raw
    comps, pairs, rows = {}, [], []
    for g,years in groups.items():
        idx, sst_months, lh_months = shifted_indices(months,years)
        comps[g] = field[idx].mean(axis=0)
        pairs.extend({'grupo':g,'mes_TSM':str(a),'mes_LHF':str(b)} for a,b in zip(sst_months,lh_months))
        rows.append({'grupo':g,'n_anos':len(years),'n_meses':len(idx),
                     'LHF_AMJJ_SAWP_W_m2':float(comps[g][ys,xs].mean())})
    anomalies = {g:comps[g]-comps['normal'] for g,_ in GROUPS}
    for row in rows:
        row['anomalia_LHF_AMJJ_SAWP_W_m2'] = float((comps[row['grupo']]-comps['normal'])[ys,xs].mean())
    # Confere grupos com a classificacao original ja existente, se disponivel.
    reference = root/'figuras/ComparacaoQuantisThetaE/ClassificacaoAnos.csv'
    if reference.exists():
        ref = pd.read_csv(reference,index_col='ano')
        assert table['grupo'].equals(ref['original'].astype(str).rename('grupo'))
    table.to_csv(out/'ClassificacaoTSM_AMJJ.csv',encoding='utf-8-sig')
    pd.DataFrame(pairs).to_csv(out/'CorrespondenciaMeses_Lag0.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(rows).to_csv(out/'ResumoSAWP_LHF_Lag0.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(out/'CompostosLHF_Lag0_Wm2.npz',lat=lat,lon=lon,
                        **{f'media_{g}':v for g,v in comps.items()},
                        **{f'anomalia_{g}':v for g,v in anomalies.items()})
    limit = max(5., np.ceil(max(np.max(abs(v)) for v in anomalies.values())/5)*5)
    levels = np.linspace(-limit,limit,13)
    info = dict(lag_meses=0,meses_TSM=[4,5,6,7],meses_LHF=[4,5,6,7],periodo=[1959,2023],
                base_quantis=[1980,2010],limiares=thresholds,anos={g:v.tolist() for g,v in groups.items()},
                conversao='-avg_slhtf; dado original ja em W/m2; positivo para cima',
                tendencia='Linear em todos os 780 meses, por ponto; media preservada',
                referencia_anomalias='AMJJ dos anos do grupo normal, tambem com lag 0',
                escala_W_m2=levels.tolist(),resumo=rows)
    (out/'Metodologia.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(rows,ensure_ascii=False,indent=2))
    if not args.sem_figura: draw(out)
    print(f'Resultados salvos em: {out}')


def draw(out):
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    cached = Path.home() / '.local/share/cartopy'
    if cached.exists():
        cartopy.config['data_dir'] = str(cached)

    info = json.loads(
        (out / 'Metodologia.json').read_text(encoding='utf-8')
    )

    levels = info['escala_W_m2']

    with np.load(out / 'CompostosLHF_Lag0_Wm2.npz') as z:
        lon, lat = z['lon'], z['lat']
        anomalies = {
            g: z[f'anomalia_{g}']
            for g, _ in GROUPS
        }

    proj = ccrs.PlateCarree()

    # Figura com margens reduzidas.
    fig, axes = plt.subplots(
        2, 2,
        figsize=(11.6, 9.0),
        subplot_kw={'projection': proj}
    )

    fig.subplots_adjust(
        left=0.085,
        right=0.865,
        bottom=0.085,
        top=0.970,
        wspace=0.05,
        hspace=0.11
    )

    states = cfeature.NaturalEarthFeature(
        'cultural',
        'admin_1_states_provinces_lakes',
        '10m',
        facecolor='none'
    )

    # Identificadores exclusivos dos painéis.
    panel_labels = ['a)', 'b)', 'c)', 'd)']

    for i, (g, _) in enumerate(GROUPS):

        ax = axes.flat[i]

        ax.set_extent(
            [-60, -10, -30, 15],
            crs=proj
        )

        im = ax.contourf(
            lon,
            lat,
            anomalies[g],
            levels=levels,
            cmap='bwr',
            extend='both',
            transform=proj,
            alpha=0.5
        )

        ax.add_feature(
            cfeature.LAND,
            facecolor='#f3f3fa',
            edgecolor='none',
            zorder=3
        )

        ax.coastlines(
            resolution='50m',
            linewidth=0.55,
            zorder=4
        )

        ax.add_feature(
            cfeature.BORDERS.with_scale('50m'),
            linewidth=0.4,
            linestyle=':',
            zorder=4
        )

        ax.add_feature(
            states,
            edgecolor='black',
            linewidth=0.3,
            alpha=0.3,
            zorder=4
        )

        # Região SAWP.
        ax.add_patch(
            Rectangle(
                (-34, -12.5),
                9, 8.5,
                facecolor='lightgray',
                edgecolor='black',
                linewidth=1.2,
                hatch='///',
                alpha=0.18,
                transform=proj,
                zorder=5
            )
        )

        ax.legend(
            handles=[
                Patch(
                    facecolor='lightgray',
                    edgecolor='black',
                    hatch='///',
                    alpha=0.35,
                    label='SAWP'
                )
            ],
            loc='upper right',
            fontsize=8,
            framealpha=1,
            edgecolor='black',
            handlelength=1.5
        )

        # Coordenadas apenas nas bordas externas.
        ax.set_xticks(
            np.arange(-60, -5, 10) if i >= 2 else [],
            crs=proj
        )

        ax.set_yticks(
            np.arange(-30, 20, 10) if i % 2 == 0 else [],
            crs=proj
        )

        ax.tick_params(labelsize=13)

        # Apenas a), b), c) e d).
        ax.set_title(
            panel_labels[i],
            loc='left',
            fontsize=16,
            pad=4
        )

    # Rótulos globais dos eixos.
    fig.supxlabel(
        'Longitude',
        fontsize=19,
        y=0.015
    )

    fig.supylabel(
        'Latitude',
        fontsize=19,
        x=0.015
    )

    # Barra de cores independente da geometria dos mapas.
    cax = fig.add_axes([
        0.895,  # posição horizontal
        0.20,   # posição vertical
        0.024,  # largura
        0.60    # altura
    ])

    cb = fig.colorbar(
        im,
        cax=cax,
        orientation='vertical'
    )

    cb.set_label(
        'Latent heat flux anomaly (W m$^{-2}$)',
        fontsize=15
    )

    cb.ax.tick_params(labelsize=12)

    # Exportação com remoção das margens externas excedentes.
    for ext in ['png', 'pdf']:
        fig.savefig(
            out / f'LH_AnomaliasQuantis_Lag0.{ext}',
            dpi=240,
            bbox_inches='tight',
            pad_inches=0.02,
            facecolor='white'
        )

    plt.close(fig)


if __name__ == '__main__':
    main()
