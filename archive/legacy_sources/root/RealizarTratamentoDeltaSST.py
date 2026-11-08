#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeltaSST = anomalia de julho menos anomalia de abril, por ano.

SST sem tendencia; climatologias mensais 1980-2010, por ponto da grade.
Mesmos grupos da SST AMJJ usados em RealizarTratamentoCalorLatenteComLag0.
DeltaSST.png mostra a media do delta em cada grupo, sem subtrair o normal.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from RealizarTratamentoCalorLatenteComLag0 import classify_sst, GROUPS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    here=Path(__file__).resolve().parent
    p.add_argument('--base-dir',type=Path,default=here.parent if here.name=='__pycache__' else here)
    p.add_argument('--output-dir',type=Path)
    args=p.parse_args(); root=args.base_dir.resolve()
    out=(args.output_dir or root/'figuras/DeltaSST').resolve();out.mkdir(parents=True,exist_ok=True)
    table,groups,thresholds,lat,lon,ys,xs=classify_sst(root)
    with np.load(root/'outputs/detrended_sst.npz') as z:
        sst=z['data'].astype(float)
        dates=pd.DatetimeIndex(z['time'])
    # Mascara fixa de oceano: nao permite sentinelas terrestres nos calculos.
    ocean=np.all(np.isfinite(sst)&(sst>200)&(sst<330),axis=0)
    years=np.arange(1959,2024)
    index=dates.to_period('M')
    ia=index.get_indexer(pd.PeriodIndex([f'{y}-04' for y in years],freq='M'))
    ij=index.get_indexer(pd.PeriodIndex([f'{y}-07' for y in years],freq='M'))
    assert (ia>=0).all() and (ij>=0).all() and ((ij-ia)==3).all()
    april=sst[ia]; july=sst[ij]; del sst
    base=(years>=1980)&(years<=2010)
    ca=april[base].mean(axis=0); cj=july[base].mean(axis=0)
    aa=np.where(ocean,april-ca,np.nan); aj=np.where(ocean,july-cj,np.nan)
    delta=aj-aa
    np.testing.assert_allclose(delta[:,ocean],((july-april)-(cj-ca))[:,ocean],atol=1e-12)
    np.testing.assert_allclose(delta[base][:,ocean].mean(axis=0),0,atol=1e-10)
    assert ocean[ys,xs].all()
    comp={g:delta[np.isin(years,selection)].mean(axis=0) for g,selection in groups.items()}
    table['anomalia_abril_SAWP_K']=aa[:,ys,xs].mean(axis=(1,2))
    table['anomalia_julho_SAWP_K']=aj[:,ys,xs].mean(axis=(1,2))
    table['DeltaSST_SAWP_K']=delta[:,ys,xs].mean(axis=(1,2))
    table.to_csv(out/'DeltaSST_PorAno.csv',encoding='utf-8-sig')
    rows=[dict(grupo=g,n_anos=len(selection),DeltaSST_SAWP_K=float(comp[g][ys,xs].mean())) for g,selection in groups.items()]
    pd.DataFrame(rows).to_csv(out/'DeltaSST_ResumoSAWP.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(out/'DeltaSST_Campos.npz',lat=lat,lon=lon,anos=years,delta_por_ano=delta,
                        **{f'media_{g}':v for g,v in comp.items()})
    info=dict(formula='(SST_julho-climatologia_julho)-(SST_abril-climatologia_abril)',
              tendencia='SST ja sem tendencia; nenhuma segunda remocao',periodo=[1959,2023],base=[1980,2010],
              classificacao='Mesmos quantis da SST AMJJ; nao sao quantis de DeltaSST',limiares=thresholds,
              anos={g:v.tolist() for g,v in groups.items()},unidade='K, numericamente igual a graus Celsius para diferencas',
              principal='DeltaSST: medias do delta; sem subtracao do grupo normal',
              resumo=rows)
    (out/'Metodologia.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    draw(out,lat,lon,comp,groups)
    print(json.dumps(rows,ensure_ascii=False,indent=2))


def draw(out,lat,lon,fields,groups):
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    cached=Path.home()/'.local/share/cartopy'
    if cached.exists():cartopy.config['data_dir']=str(cached)
    limit=max(.1,np.ceil(max(np.nanmax(abs(fields[g])) for g,_ in GROUPS)*10)/10)
    levels=np.linspace(-limit,limit,13)
    proj=ccrs.PlateCarree()
    fig,axes=plt.subplots(2,2,figsize=(12,10),subplot_kw={'projection':proj},layout='constrained')
    for i,(g,label) in enumerate(GROUPS):
        ax=axes.flat[i];ax.set_extent([-60,-10,-30,15],crs=proj)
        im=ax.contourf(lon,lat,fields[g],levels=levels,cmap='bwr',extend='both',transform=proj)
        ax.add_feature(cfeature.LAND.with_scale('50m'),facecolor='#eeeeee')
        ax.coastlines(resolution='50m',linewidth=.6)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'),linewidth=.4,linestyle=':')
        ax.add_patch(Rectangle((-34,-12.5),9,8.5,fill=False,edgecolor='black',linewidth=1.2,transform=proj))
        ax.text(-24.5,-4,'SAWP',fontsize=8,transform=proj)
        ax.set_xticks(np.arange(-60,-5,10) if i>=2 else [],crs=proj)
        ax.set_yticks(np.arange(-30,20,10) if i%2==0 else [],crs=proj)
        if i>=2:ax.set_xlabel('Longitude (°)')
        if i%2==0:ax.set_ylabel('Latitude (°)')
        ax.set_title(f'{"abcd"[i]}) {label} (n = {len(groups[g])})',loc='left',fontsize=12)
    cb=fig.colorbar(im,ax=axes,shrink=.72,pad=.02)
    cb.set_label('DeltaSST: anomalia de julho − anomalia de abril (°C)')
    fig.suptitle('DeltaSST\nAnomalia de julho − anomalia de abril | sem tendência | quantis AMJJ 1980–2010',fontsize=13)
    for ext in ['png','pdf']:fig.savefig(out/f'DeltaSST.{ext}',dpi=240)
    plt.close(fig)


if __name__=='__main__':main()
