#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Composites de agua precipitavel em 850-400 hPa, sem tendencia, AMJJ.

Compara quantis de TSM 1980-2010 e 1959-2023, ambos aplicados a 1959-2023.
Reutiliza a classificacao e integracao de RealizarComparacaoQuantisTransporteUmidade.
Cada painel mostra o grupo menos o normal da respectiva classificacao.
Agua precipitavel = integral(q dp)/g; kg/m2 equivale numericamente a mm.
Executar: python RealizarComparacaoQuantisVaporMediosNiveis.py
"""
from pathlib import Path
from contextlib import ExitStack
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from RealizarComparacaoQuantisTransporteUmidade import classifications, open_field, integrate, GROUPS


def annual_water(root):
    with np.load(root/'outputs/detrended_sst.npz') as z:
        sst_months=pd.DatetimeIndex(z['time']).to_period('M')
    with ExitStack() as stack:
        stream,shape,dtype,meta=open_field(stack,root/'outputs/detrended_uvq_q.npz')
        dates=meta['time']
        if dates.to_period('M').duplicated().any() or not dates.to_period('M').equals(sst_months):
            raise ValueError('Meses de q e TSM precisam coincidir, sem duplicatas.')
        data=np.zeros((65,len(meta['lat']),len(meta['lon'])),dtype=float)
        counts=np.zeros(65,dtype=int)
        used=[]
        size=int(np.prod(shape[1:]))*dtype.itemsize
        for dt in dates:
            raw=stream.read(size)
            if len(raw)!=size:
                raise ValueError('Arquivo de q truncado.')
            if not (1959<=dt.year<=2023 and dt.month in [4,5,6,7]):
                continue
            q=np.frombuffer(raw,dtype).reshape(shape[1:])[meta['order']]
            if not np.isfinite(q).all():
                raise ValueError(f'Valores ausentes em {dt}.')
            data[dt.year-1959]+=integrate(q,meta['level'],850,400)
            counts[dt.year-1959]+=1
            used.append(str(dt.date()))
        if not (counts==4).all():
            raise ValueError('Cada ano deve conter exatamente abril, maio, junho e julho.')
    return data/4,meta,used


def draw(out,method,fields,groups,meta):
    base='1980–2010' if method=='original' else '1959–2023'
    proj=ccrs.PlateCarree()
    fig,axes=plt.subplots(2,2,figsize=(13,12),subplot_kw={'projection':proj},layout='constrained')
    levels=np.arange(-3,3.01,.5)
    for i,(g,name) in enumerate(GROUPS):
        ax=axes.flat[i]
        ax.set_extent([-60,-10,-30,15],crs=proj)
        ax.add_feature(cfeature.LAND.with_scale('50m'),facecolor='lightgrey',alpha=.3,zorder=0)
        image=ax.contourf(meta['lon'],meta['lat'],fields[g],levels=levels,
                          cmap='BrBG',extend='both',alpha=.5,transform=proj)
        ax.coastlines(resolution='50m',linewidth=.6)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'),linewidth=.4,linestyle=':')
        states=cfeature.NaturalEarthFeature('cultural','admin_1_states_provinces_lakes','50m',facecolor='none')
        ax.add_feature(states,edgecolor='black',linewidth=.25,alpha=.35)
        ax.add_patch(Rectangle((-34,-12.5),9,8.5,linewidth=1.2,edgecolor='black',
                               facecolor='lightgray',hatch='///',alpha=.18,transform=proj,zorder=5))
        ax.set_xticks(np.arange(-60,-5,10) if i>=2 else [],crs=proj)
        ax.set_yticks(np.arange(-30,20,10) if i%2==0 else [],crs=proj)
        ax.set_title(f'{"abcd"[i]}) {name} | n={len(groups[g])}',loc='left',fontsize=13)
        if i>=2: ax.set_xlabel('Longitude (°)')
        if i%2==0: ax.set_ylabel('Latitude (°)')
    cb=fig.colorbar(image,ax=axes,shrink=.75,pad=.03)
    cb.set_label('Água precipitável em 850–400 hPa: grupo − normal (mm)')
    fig.suptitle(f'Vapor de água em médios níveis | 850–400 hPa | sem tendência | AMJJ\n'
                 f'Quantis {base} aplicados a 1959–2023',fontsize=15)
    for ext in ['png','pdf']:
        fig.savefig(out/f'VaporMediosNiveis_{method}.{ext}',dpi=240)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    root=args.base_dir.resolve()
    out=args.output_dir or root/'figuras/ComparacaoQuantisVaporMediosNiveis'
    out.mkdir(parents=True,exist_ok=True)
    cached=Path.home()/'.local/share/cartopy'
    if cached.is_dir(): cartopy.config['data_dir']=cached
    table,groups,thresholds=classifications(root)
    print('Integrando q sem tendencia: 850-400 hPa, somente AMJJ...',flush=True)
    annual,meta,used=annual_water(root)
    outputs={}
    for method,selection in groups.items():
        comp={g:annual[years-1959].mean(axis=0) for g,years in selection.items()}
        fields={g:comp[g]-comp['normal'] for g,_ in GROUPS}
        for g,field in comp.items(): outputs[f'{method}_absoluto_{g}']=field
        for g,field in fields.items(): outputs[f'{method}_anomalia_{g}']=field
        draw(out,method,fields,selection,meta)
    table.to_csv(out/'ClassificacaoAnos.csv')
    pd.DataFrame({'data':used}).to_csv(out/'MesesUtilizados.csv',index=False)
    np.savez_compressed(out/'Composites.npz',lat=meta['lat'],lon=meta['lon'],**outputs)
    info=dict(entrada='outputs/detrended_uvq_q.npz',camada_hPa=[850,400],
              integracao='integral de q dp / 9.81; trapezios em Pa; resultado em kg/m2 = mm',
              periodo=[1959,2023],meses=[4,5,6,7],climatologia_TSM=[1980,2010],
              limites_quantis=thresholds,escala_mm=levels_list(),
              anomalia='grupo menos normal da respectiva classificacao',
              anos={m:{g:y.tolist() for g,y in sel.items()} for m,sel in groups.items()})
    (out/'Metodologia.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(table.groupby('original').size())
    print(table.groupby('completo').size())
    print(out.resolve())


def levels_list():
    return np.arange(-3,3.01,.5).tolist()


if __name__=='__main__':
    main()
