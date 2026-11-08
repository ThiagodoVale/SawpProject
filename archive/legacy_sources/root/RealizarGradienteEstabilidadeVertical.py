#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ThetaE sem tendencia: cortes por quantil de TSM e contraste vertical.

Executar na raiz do SAWP: python RealizarGradienteEstabilidadeVertical.py
Dependencias: numpy, pandas, matplotlib. Nao requer GPU nem importar sawp.

Climatologia mensal e quantis AMJJ de 1980-2010, aplicados a 1959-2023. Recorte espacial da figura original.
Cortes e barras: composto menos normal. Somente abril, maio, junho e julho.
Integrais trapezoidais em pressao crescente (hPa):
    I_inferior = integral de 600 a 1000 hPa de ThetaE dp
    I_superior = integral de 100 a 600 hPa de ThetaE dp
    contraste = I_superior - I_inferior, em K hPa.
As barras usam (I_superior/500 - I_inferior/400), relativo ao grupo normal, em K.
A media de cada camada e sua integral dividida pela espessura (500 ou 400 hPa).
A diferenca das medias nao e uma derivada vertical. O CSV preserva tambem
as integrais em K hPa para rastreabilidade.
O arquivo ThetaE existente foi calculado a partir de T e q sem tendencia;
nao se aplica uma segunda remocao de tendencia aqui.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.lib.format import read_magic, read_array_header_1_0, read_array_header_2_0

GROUPS = [('10', 'Muito frio', '#2166ac'),
          ('25', 'Moderadamente frio', '#67a9cf'),
          ('75', 'Moderadamente quente', '#ef8a62'),
          ('90', 'Extremamente quente', '#b2182b')]


def metadata(path):
    with np.load(path, allow_pickle=False) as z:
        result = {k: z[k] for k in z.files if k != 'data'}
    key = 'time' if 'time' in result else 'valid_time'
    if not np.issubdtype(result[key].dtype, np.datetime64):
        raise ValueError(f'{path}: tempo deve ser datetime64, sem assumir epoca.')
    result['dates'] = pd.DatetimeIndex(result[key])
    if result['dates'].to_period('M').duplicated().any():
        raise ValueError(f'{path}: meses duplicados.')
    for key in ('lat', 'lon'):
        if result[key].ndim != 1:
            raise ValueError(f'{path}: {key} deve ser unidimensional.')
    return result


def legacy_slice(coord, start, stop):
    """Reproduz o recorte half-open usado no projeto; exige ordem esperada."""
    i, j = (int(np.argmin(abs(coord - v))) for v in (start, stop))
    if j <= i:
        raise ValueError('Coordenadas em ordem diferente da figura original.')
    return slice(i, j)


def classify(root, sst_path):
    meta = metadata(sst_path)
    dates = meta['dates']
    expected = pd.period_range(dates.min(), dates.max(), freq='M')
    if not dates.to_period('M').equals(expected):
        raise ValueError('TSM deve conter meses consecutivos em ordem crescente.')
    for year in np.unique(dates.year):
        if sum((dates.year == year) & dates.month.isin([4, 5, 6, 7])) != 4:
            raise ValueError(f'TSM: AMJJ incompleto em {year}.')
    ys = legacy_slice(meta['lat'], -4, -12.5)
    xs = legacy_slice(meta['lon'], -34, -25)
    with np.load(sst_path, allow_pickle=False) as z:
        data = z['data'][:, ys, xs].copy()
    data[~np.isfinite(data) | (data < 0)] = np.nan
    values = np.nanmean(np.nanmean(data, axis=1), axis=1) - 273.15
    del data
    seasonal = pd.DataFrame({'TSM': values}, index=dates)
    seasonal = seasonal.loc[seasonal.index.month.isin([4, 5, 6, 7])].copy()
    climate = seasonal.loc[(seasonal.index.year >= 1980) & (seasonal.index.year <= 2010)]
    monthly_climate = climate.groupby(climate.index.month)['TSM'].mean()
    seasonal['Anomalia'] = seasonal['TSM'] - seasonal.index.month.map(monthly_climate)
    table = seasonal.groupby(seasonal.index.year).mean()
    table.index.name = 'Ano'
    if not np.isfinite(table[['TSM', 'Anomalia']].to_numpy()).all():
        raise ValueError('TSM ou climatologia com valores ausentes.')
    table = table.loc[1959:2023].copy()
    anomaly = table['Anomalia']
    q10, q25, q75, q90 = anomaly.loc[1980:2010].quantile([.1, .25, .75, .9]).to_numpy()
    masks = {'10': anomaly <= q10,
             '25': (anomaly > q10) & (anomaly <= q25),
             'normal': (anomaly > q25) & (anomaly < q75),
             '75': (anomaly >= q75) & (anomaly < q90),
             '90': anomaly >= q90}
    groups = {}
    table['grupo'] = ''
    for group, mask in masks.items():
        years = table.index[mask].to_numpy()
        if not len(years):
            raise ValueError(f'Grupo {group} vazio; nao ha composto a calcular.')
        groups[group] = years
        table.loc[years, 'grupo'] = group
    if (table['grupo'] == '').any():
        raise ValueError('Ha anos sem classificacao de TSM.')
    table.attrs['limites_quantis_K'] = dict(zip(['P10', 'P25', 'P75', 'P90'], [q10, q25, q75, q90]))
    return groups, table


def read_sections(path, groups, reference=None):
    """Le um passo temporal por vez do NPZ, evitando carregar 2-4 GB na RAM."""
    meta = metadata(path)
    levels = np.asarray(meta['level'], dtype=float)
    if levels.ndim != 1 or len(np.unique(levels)) != len(levels):
        raise ValueError(f'{path}: niveis invalidos.')
    order = np.argsort(levels)
    levels = levels[order]
    if levels[0] > 100 or levels[-1] < 1000:
        raise ValueError(f'{path}: cobertura insuficiente para 100-1000 hPa.')
    if reference is not None:
        for key in ('lat', 'lon'):
            if not np.array_equal(meta[key], reference[key]):
                raise ValueError(f'{path}: grade {key} diferente de ThetaE.')
        if not meta['dates'].to_period('M').equals(reference['dates'].to_period('M')):
            raise ValueError(f'{path}: meses diferentes de ThetaE.')
    ys = legacy_slice(meta['lat'], -4, -12.5)
    wanted_years = np.unique(np.concatenate(list(groups.values())))
    selected = np.flatnonzero(meta['dates'].year.isin(wanted_years)
                             & meta['dates'].month.isin([4, 5, 6, 7]))
    for year in wanted_years:
        if sum(meta['dates'][selected].year == year) != 4:
            raise ValueError(f'{path}: faltam meses AMJJ para {year}.')
    sections = []
    selected_set = set(selected.tolist())
    with zipfile.ZipFile(path) as archive, archive.open('data.npy') as stream:
        version = read_magic(stream)
        if version not in ((1, 0), (2, 0)):
            raise ValueError(f'Versao NPY nao suportada: {version}')
        shape, fortran, dtype = (read_array_header_1_0 if version == (1, 0)
                                 else read_array_header_2_0)(stream)
        if fortran or dtype.hasobject:
            raise ValueError('NPZ precisa de array numerico em ordem C.')
        expected = (len(meta['dates']), len(levels), len(meta['lat']), len(meta['lon']))
        if shape != expected:
            raise ValueError(f'{path}: shape {shape}, esperado {expected}.')
        nbytes = int(np.prod(shape[1:])) * dtype.itemsize
        for i in range(shape[0]):
            raw = stream.read(nbytes)
            if len(raw) != nbytes:
                raise ValueError(f'{path}: arquivo truncado no tempo {i}.')
            if i not in selected_set:
                continue
            slab = np.frombuffer(raw, dtype=dtype).reshape(shape[1:])
            field = slab[order, ys, :].astype(float)
            field[~np.isfinite(field) | (field == -999)] = np.nan
            sections.append(np.nanmean(field, axis=1))
    meta['level'] = levels
    return np.stack(sections), meta['dates'][selected], meta


def integrate_layer(profile, levels, top, bottom):
    """Integral positiva em dp; interpola apenas limites internos da grade."""
    p = np.asarray(levels, dtype=float)
    order = np.argsort(p)
    p, profile = p[order], np.asarray(profile)[order]
    if len(np.unique(p)) != len(p) or top < p[0] or bottom > p[-1] or top >= bottom:
        raise ValueError('Limites ou niveis invalidos para integrar.')
    grid = np.unique(np.r_[top, p[(p > top) & (p < bottom)], bottom])
    values = np.interp(grid, p, profile)
    if not np.isfinite(values).all():
        raise ValueError('Perfil incompleto na camada: integral nao calculada.')
    return float(np.sum((values[:-1] + values[1:]) * np.diff(grid) / 2))


def composites(sections, dates, groups):
    return {g: np.nanmean(sections[dates.year.isin(years)], axis=0)
            for g, years in groups.items()}


def plot_bars(ax, summary, column, ylabel):
    values = summary.loc[[g for g, _, _ in GROUPS], column].to_numpy()
    bars = ax.bar(range(4), values, color=[c for _, _, c in GROUPS], width=.62)
    ax.axhline(0, color='#555555', linewidth=.8)
    ax.set_xticks(range(4), [f'{label}\n(n={int(summary.loc[g, "n_anos"])})'
                            for g, label, _ in GROUPS], fontsize=10)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f'{v:,.1f}' for v in values], padding=4, fontsize=10)
    ax.margins(y=.25)


def draw_quantiles(out,comp,meta,summary):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12})
    fig=plt.figure(figsize=(12,12))
    gs=fig.add_gridspec(2,2,left=.10,right=.83,bottom=.34,top=.96,hspace=.12,wspace=.15)
    axes=[fig.add_subplot(gs[i//2,i%2]) for i in range(4)]
    lon,p=meta['lon'],meta['level'];grid=np.linspace(100,1000,60)
    for i,(g,_,_) in enumerate(GROUPS):
        ax=axes[i]
        fields={}
        for v in ['te','u','w']:
            a=comp[v][g]-comp[v]['normal']
            fields[v]=np.column_stack([np.interp(grid,p,a[:,j]) for j in range(len(lon))])
        im=ax.contourf(lon,grid,fields['te'],levels=np.arange(-3,3.01,.5),
                       cmap='RdYlBu_r',extend='both',alpha=.4)
        scales={v:float(np.std(fields[v])) or 1. for v in ['u','w']}
        q = ax.quiver(
            lon[::8],
            grid[::2],
            2 * fields['u'][::2, ::8] / scales['u'],
            -fields['w'][::2, ::8] / scales['w'],
            color='black',
            pivot='mid',
            scale=50,
            width=0.002
        )
        
        if i == 0:
            q_ref = q
        ax.fill_betweenx([1000,200],-40,-30,color='gray',alpha=.2)
        ax.set(xlim=(-60,-10),ylim=(1000,200))
        ax.set_title(f'{"abcd"[i]})',loc='left',fontsize=18)
        ax.set_yticks(np.arange(1000,200,-200) if i%2==0 else [])
        ax.set_xticks(np.linspace(lon.min(),lon.max(),5) if i>=2 else [])
        ax.tick_params(labelsize=14)
    fig.text(.465,.295,'Longitude',ha='center',fontsize=18)
    fig.text(.025,.63,'Pressure Levels (hPa)',va='center',rotation='vertical',fontsize=18)
    axes[0].quiverkey(
        q_ref,
        X=0.15,
        Y=0.255,
        U=2,
        label='1 (Standardized u–w anomalies)',
        coordinates='figure',
        labelpos='E',
        labelsep=0.05,
        fontproperties={'size': 10}
    )
    cax=fig.add_axes([.86,.465,.018,.33])
    cb=fig.colorbar(im,cax=cax)
    cb.set_label('Equivalent Potential Temperature (K)',size=14)
    cb.ax.tick_params(labelsize=12)
    ax=fig.add_axes([.10,.065,.73,.13])
    labels=['Very cold','Moderately cold','Moderately warm','Extremely warm']
    values=[float(summary.loc[g,'anomalia_diferenca_medias_K']) for g,_,_ in GROUPS]
    ax.bar(np.arange(4),values,color=[color for _,_,color in GROUPS],width=.55,zorder=3)
    ax.axhline(0,color='#555555',linewidth=.8)
    ax.set_xticks(np.arange(4),labels,fontsize=11)
    ax.set_ylim(-2, 2)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_ylabel('K',fontsize=12)
    ax.set_title(r'$\theta_e$ gradient',fontsize=15,pad=8)
    ax.set_title('e)', loc='right', fontsize=18, pad=8)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#eeeeee',linewidth=.6,zorder=0)
    for i,value in enumerate(values):
        ax.annotate(f'{value:.2f}',(i,value),xytext=(0,5 if value>=0 else -5),
                    textcoords='offset points',ha='center',va='bottom' if value>=0 else 'top',fontsize=10)
    ax.text(.5,-.36,'Layer means: 600–100 hPa − 1000–600 hPa | Anomalies relative to the normal category',
            ha='center',va='top',transform=ax.transAxes,fontsize=10)
    for ext in ['png','pdf']:fig.savefig(out/f'ThetaE_Quantis_original.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)



def make_figures(out, comp, meta, summary, wind=None):
    if wind is None:
        zeros = {g: np.zeros_like(v) for g, v in comp.items()}
        fields = {'te': comp, 'u': zeros, 'w': zeros}
    else:
        ug, wg, up, wp = wind
        np.testing.assert_array_equal(up, meta['level'])
        np.testing.assert_array_equal(wp, meta['level'])
        fields = {'te': comp, 'u': ug, 'w': wg}
    draw_quantiles(out, fields, meta, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--lon-bars', type=float, nargs=2, default=(-34, -25), metavar=('OESTE', 'LESTE'))
    parser.add_argument('--sem-vetores', action='store_true')
    args = parser.parse_args()
    root = args.base_dir.resolve()
    out = args.output_dir or root / 'figuras' / 'GradienteEstabilidadeVertical'
    out.mkdir(parents=True, exist_ok=True)
    groups, table = classify(root, root / 'outputs' / 'detrended_sst.npz')
    print('Lendo ThetaE por passos temporais...', flush=True)
    sections, dates, meta = read_sections(root / 'outputs' / 'detrended_uvq_te.npz', groups)
    comp = composites(sections, dates, groups)
    mask = (meta['lon'] >= args.lon_bars[0]) & (meta['lon'] <= args.lon_bars[1])
    if not mask.any() or args.lon_bars[0] >= args.lon_bars[1]:
        raise ValueError('Recorte longitudinal das barras vazio ou invertido.')
    rows = []
    # Cada ano recebe o mesmo peso; meses AMJJ completos foram validados.
    for year in np.unique(dates.year):
        profile = np.nanmean(np.nanmean(sections[dates.year == year], axis=0)[:, mask], axis=1)
        lower = integrate_layer(profile, meta['level'], 600, 1000)
        upper = integrate_layer(profile, meta['level'], 100, 600)
        rows.append(dict(ano=int(year), grupo=table.loc[year, 'grupo'],
                         inferior_K_hPa=lower, superior_K_hPa=upper,
                         contraste_K_hPa=upper-lower,
                         media_inferior_K=lower/400, media_superior_K=upper/500,
                         diferenca_medias_K=upper/500-lower/400))
    annual = pd.DataFrame(rows).set_index('ano')
    summary = annual.groupby('grupo').mean(numeric_only=True)
    summary['n_anos'] = annual.groupby('grupo').size()
    summary['anomalia_contraste_K_hPa'] = summary['contraste_K_hPa'] - summary.loc['normal', 'contraste_K_hPa']
    summary['anomalia_inferior_K_hPa'] = summary['inferior_K_hPa'] - summary.loc['normal', 'inferior_K_hPa']
    summary['anomalia_superior_K_hPa'] = summary['superior_K_hPa'] - summary.loc['normal', 'superior_K_hPa']
    summary['anomalia_diferenca_medias_K'] = summary['diferenca_medias_K'] - summary.loc['normal', 'diferenca_medias_K']
    # Confere as barras diretamente contra os mesmos campos dos paineis a-d.
    for g in groups:
        profile = np.nanmean((comp[g] - comp['normal'])[:, mask], axis=1)
        direct = integrate_layer(profile, meta['level'], 100, 600)/500 - integrate_layer(profile, meta['level'], 600, 1000)/400
        if not np.isclose(direct, summary.loc[g, 'anomalia_diferenca_medias_K'], atol=1e-6):
            raise ValueError(f'Barra e corte vertical inconsistentes no grupo {g}.')
    used = pd.DataFrame({'data': dates, 'ano': dates.year, 'mes': dates.month})
    used['grupo'] = used['ano'].map(table['grupo'])
    if set(used['mes']) != {4, 5, 6, 7} or not (used.groupby('ano').size() == 4).all():
        raise ValueError('A selecao precisa conter exatamente quatro meses AMJJ por ano.')
    used.to_csv(out / 'MesesEfetivamenteUtilizados.csv', index=False)
    wind = None
    if not args.sem_vetores:
        comps, ps = [], []
        for var in ('u', 'w'):
            print(f'Lendo {var} por passos temporais...', flush=True)
            values, dt, info = read_sections(root / 'outputs' / f'detrended_uvq_{var}.npz', groups, meta)
            comps.append(composites(values, dt, groups))
            ps.append(info['level'])
            del values
        wind = (*comps, *ps)
    table.to_csv(out / 'Classificacao_TSM_AMJJ.csv', encoding='utf-8-sig')
    annual.to_csv(out / 'ThetaE_Gradiente_PorAno.csv', encoding='utf-8-sig')
    summary.to_csv(out / 'ThetaE_Gradiente_PorQuantil.csv', encoding='utf-8-sig')
    np.savez_compressed(out / 'ThetaE_CortesPorQuantil.npz', level=meta['level'], lon=meta['lon'],
                        **{f'grupo_{g}': v for g, v in comp.items()})
    (out / 'metodologia.json').write_text(json.dumps(dict(
        base_dir=str(root), climatologia=[1980, 2010], meses=[4, 5, 6, 7],
        periodo_quantis=[1980, 2010], periodo_composites=[1959, 2023],
        limites_quantis_K=table.attrs['limites_quantis_K'],
        populacao_quantis='medias anuais de abril a julho; quantis em 1980-2010',
        entrada_tsm='outputs/detrended_sst.npz', entrada_thetae='outputs/detrended_uvq_te.npz',
        latitude='-12.5 < lat <= -4; recorte da figura original',
        longitude_barras=list(args.lon_bars), media_espacial='aritmetica, como no projeto',
        grupos={g: years.tolist() for g, years in groups.items()},
        integral='trapezios em pressao crescente, K hPa; superior - inferior',
        corte='ThetaE do grupo menos ThetaE normal; eixo 1000 a 200 hPa',
        barras='diferenca de medias de camada: I_superior/500 - I_inferior/400, relativa ao grupo normal, em K',
        vetores='anomalias de u e w normalizadas pelo desvio espacial; escala ilustrativa'
    ), ensure_ascii=False, indent=2), encoding='utf-8')
    make_figures(out, comp, meta, summary, wind)
    print(summary.to_string())
    print(f'Figuras e tabelas salvas em: {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
