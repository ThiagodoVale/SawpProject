#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quatro mapas de anomalias de LHF sem tendencia com lag de 0 meses.

TSM AMJJ classifica os anos usando quantis 1980-2010, aplicados a 1959-2023.
LHF usa ABRIL-JULHO dos mesmos anos, inclusive no composto normal.
ERA5 monthly means of daily means (moda): LHF = -slhf / 86400 W/m2,
positivo do oceano para a atmosfera. Nao dividir pelo numero de dias do mes.
Remove tendencia linear de todos os meses de 1959-2023 preservando a media.

Executar na raiz do projeto:
    python RealizarTratamentoCalorLatenteComLag0.py
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

GROUPS = [('10', 'Muito frio'), ('25', 'Moderadamente frio'),
          ('75', 'Moderadamente quente'), ('90', 'Extremamente quente')]


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
    out = (args.output_dir or root/'figuras/CalorLatenteComLag0').resolve()
    out.mkdir(parents=True,exist_ok=True)
    if args.somente_figura:
        draw(out)
        return
    if h5py is None: raise ImportError('Instale h5py para ler o NetCDF de LHF.')
    table, groups, thresholds, lat, lon, ys, xs = classify_sst(root)
    with h5py.File(root/'data/Era5_Dataset_lhsh.nc') as f:
        assert 'moda' in str(f.attrs['history']), 'Conversao exige medias mensais de acumulacoes diarias.'
        assert f['slhf'].attrs['units'].decode() == 'J m**-2'
        np.testing.assert_array_equal(f['latitude'][:],lat)
        np.testing.assert_array_equal(f['longitude'][:],lon)
        months = pd.to_datetime(f['valid_time'][:],unit='s').to_period('M')
        assert months.equals(pd.period_range('1959-01','2023-12',freq='M'))
        raw = -f['slhf'][:].astype(float)/86400.
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
                conversao='-slhf/86400; W/m2 positivos para cima',
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
    cached = Path.home()/'.local/share/cartopy'
    if cached.exists(): cartopy.config['data_dir'] = str(cached)
    info = json.loads((out/'Metodologia.json').read_text(encoding='utf-8'))
    groups, levels = info['anos'], info['escala_W_m2']
    with np.load(out/'CompostosLHF_Lag0_Wm2.npz') as z:
        lon,lat=z['lon'],z['lat']
        anomalies={g:z[f'anomalia_{g}'] for g,_ in GROUPS}
    proj = ccrs.PlateCarree()
    fig,axes=plt.subplots(2,2,figsize=(12,10),subplot_kw={'projection':proj},layout='constrained')
    for i,(g,name) in enumerate(GROUPS):
        ax=axes.flat[i]
        ax.set_extent([-60,-10,-30,15],crs=proj)
        im=ax.contourf(lon,lat,anomalies[g],levels=levels,cmap='bwr',extend='both',transform=proj)
        ax.coastlines(resolution='50m',linewidth=.6)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'),linewidth=.4,linestyle=':')
        ax.add_patch(Rectangle((-34,-12.5),9,8.5,fill=False,edgecolor='black',linewidth=1.2,transform=proj))
        ax.text(-24.5,-4,'SAWP',fontsize=8,transform=proj)
        ax.set_xticks(np.arange(-60,-5,10) if i>=2 else [],crs=proj)
        ax.set_yticks(np.arange(-30,20,10) if i%2==0 else [],crs=proj)
        if i>=2: ax.set_xlabel('Longitude (°)')
        if i%2==0: ax.set_ylabel('Latitude (°)')
        ax.set_title(f'{"abcd"[i]}) {name} (n = {len(groups[g])})',loc='left',fontsize=12)
    cb=fig.colorbar(im,ax=axes,shrink=.72,pad=.02)
    cb.set_label('Anomalia do fluxo de calor latente (W m$^{-2}$)\nPositivo: oceano para atmosfera')
    fig.suptitle('Calor latente com lag de 0 meses | grupo − normal\nTSM: abril–julho; LHF: abril–julho | sem tendência | quantis 1980–2010',fontsize=13)
    for ext in ['png','pdf']: fig.savefig(out/f'LH_AnomaliasQuantis_Lag0.{ext}',dpi=240)
    plt.close(fig)


if __name__ == '__main__':
    main()
