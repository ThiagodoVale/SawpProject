#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Duas classificacoes de TSM AMJJ para composites de umidade sem tendencia.

Figura 1: quantis 1980-2010 aplicados a 1959-2023 (RealizarAnalises.py).
Figura 2: quantis 1959-2023 aplicados a 1959-2023.
Ambas: abril-julho, composto menos grupo normal da respectiva classificacao.
Cores: convergencia (-divergencia) do transporte integrado em 1000-850 hPa.
Setas: transporte integrado em 850-400 hPa, como no script original.
Use --camada-transporte 600 100 para alterar explicitamente a camada.
Dependencias: numpy, pandas, matplotlib, metpy, cartopy.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from metpy.calc import divergence, lat_lon_grid_deltas
from numpy.lib.format import read_magic, read_array_header_1_0, read_array_header_2_0

GROUPS = [('10','Muito frio'), ('25','Moderadamente frio'),
          ('75','Moderadamente quente'), ('90','Extremamente quente')]


def classifications(root):
    path=root/'outputs/detrended_sst.npz'
    with np.load(path) as z:
        time=pd.DatetimeIndex(z['time'])
        lat,lon=z['lat'],z['lon']
        ys=slice(np.argmin(abs(lat+4)),np.argmin(abs(lat+12.5)))
        xs=slice(np.argmin(abs(lon+34)),np.argmin(abs(lon+25)))
        values=z['data'][:,ys,xs]
    assert np.isfinite(values).all(), 'Dados ausentes no recorte SAWP.'
    s=pd.Series(values.mean(axis=1).mean(axis=1)-273.16,index=time)
    s=s[(s.index.year>=1959)&(s.index.year<=2023)&s.index.month.isin([4,5,6,7])]
    assert len(s)==260 and (s.groupby(s.index.year).count()==4).all()
    climate=s[(s.index.year>=1980)&(s.index.year<=2010)]
    climate=climate.groupby(climate.index.month).mean()
    annual=(s-s.index.month.map(climate)).groupby(s.index.year).mean()
    table=pd.DataFrame({'anomalia_TSM_AMJJ_K':annual})
    table.index.name='ano'
    groups,thresholds={},{}
    for method,(start,end) in {'original':(1980,2010),'completo':(1959,2023)}.items():
        q=annual.loc[start:end].quantile([.1,.25,.75,.9]).to_numpy()
        a,b,c,d=q
        labels=np.select([annual<=a,annual<=b,annual<c,annual<d],['10','25','normal','75'],default='90')
        table[method]=labels
        groups[method]={g:annual.index[labels==g].to_numpy() for g in ['10','25','normal','75','90']}
        if any(len(y)==0 for y in groups[method].values()):
            raise ValueError(f'Grupo vazio em {method}.')
        thresholds[method]=dict(zip(['P10','P25','P75','P90'],q.tolist()))
    return table,groups,thresholds


def open_field(stack,path):
    with np.load(path) as z:
        key='time' if 'time' in z.files else 'valid_time'
        meta={k:z[k] for k in ['lat','lon','level']}
        meta['time']=pd.DatetimeIndex(z[key])
    archive=stack.enter_context(zipfile.ZipFile(path))
    f=stack.enter_context(archive.open('data.npy'))
    version=read_magic(f)
    if version not in [(1,0),(2,0)]:
        raise ValueError(f'Versao NPY nao suportada: {version}')
    shape,fortran,dtype=(read_array_header_1_0 if version==(1,0) else read_array_header_2_0)(f)
    assert not fortran and not dtype.hasobject
    assert shape==(len(meta['time']),len(meta['level']),len(meta['lat']),len(meta['lon']))
    meta['order']=np.argsort(meta['level'])
    meta['level']=meta['level'][meta['order']]
    return f,shape,dtype,meta


def integrate(field,p,bottom,top):
    """Integra em pressao crescente; dp em Pa, gravidade 9.81 m/s2."""
    assert top<bottom and top in p and bottom in p
    mask=(p>=top)&(p<=bottom)
    x=field[mask]
    return np.sum(.5*(x[:-1]+x[1:])*np.diff(p[mask])[:,None,None]*100/9.81,axis=0)


def annual_fluxes(root,transport):
    years=np.arange(1959,2024)
    with ExitStack() as stack:
        opened=[open_field(stack,root/'outputs'/f'detrended_uvq_{v}.npz') for v in ['u','v','q']]
        meta=opened[0][3]
        for _,_,_,m in opened:
            for key in ['lat','lon','level']:
                assert np.array_equal(meta[key],m[key]), f'Coordenadas diferentes: {key}'
            assert meta['time'].equals(m['time']), 'Datas diferentes entre variaveis.'
        time=meta['time']
        assert not time.to_period('M').duplicated().any()
        p=meta['level']
        data=np.zeros((len(years),4,len(meta['lat']),len(meta['lon'])),dtype=float)
        counts=np.zeros(len(years),dtype=int)
        used=[]
        for it,dt in enumerate(time):
            frames=[]
            selected=dt.year in years and dt.month in [4,5,6,7]
            for stream,shape,dtype,m in opened:
                size=int(np.prod(shape[1:]))*dtype.itemsize
                raw=stream.read(size)
                assert len(raw)==size, 'Arquivo truncado.'
                if selected:
                    frames.append(np.frombuffer(raw,dtype).reshape(shape[1:])[m['order']])
            if not selected:
                continue
            u,v,q=frames
            assert all(np.isfinite(x).all() for x in frames), 'Campo com valores ausentes.'
            uq,vq=u*q,v*q
            iy=dt.year-1959
            data[iy,0]+=integrate(uq,p,1000,850)
            data[iy,1]+=integrate(vq,p,1000,850)
            data[iy,2]+=integrate(uq,p,*transport)
            data[iy,3]+=integrate(vq,p,*transport)
            counts[iy]+=1
            used.append(str(dt.date()))
            if dt.month==7 and dt.year%10==0:
                print(f'Processado AMJJ de {dt.year}',flush=True)
        assert (counts==4).all(), 'AMJJ incompleto.'
        data/=4
    return data,meta,used


def draw(out,method,fields,groups,meta,transport):
    label='1980–2010' if method=='original' else '1959–2023'
    proj=ccrs.PlateCarree()
    fig,axes=plt.subplots(2,2,figsize=(13,12),subplot_kw={'projection':proj})
    fig.subplots_adjust(left=.07,right=.87,bottom=.13,top=.89,wspace=.08,hspace=.18)
    lon,lat=meta['lon'],meta['lat']
    xx,yy=np.meshgrid(lon,lat)
    for i,(g,name) in enumerate(GROUPS):
        ax=axes.flat[i]
        conv,u,v=fields[g]
        ax.set_extent([-60,-10,-30,15],crs=proj)
        ax.add_feature(cfeature.LAND.with_scale('50m'),facecolor='lightgrey',alpha=.25,zorder=0)
        image=ax.contourf(xx,yy,conv,levels=np.arange(-30,31,10)/1e6,
                          cmap='PuOr',extend='both',alpha=.5,transform=proj)
        ax.coastlines(resolution='50m',linewidth=.6)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'),linewidth=.4,linestyle=':')
        states=cfeature.NaturalEarthFeature('cultural','admin_1_states_provinces_lakes','50m',facecolor='none')
        ax.add_feature(states,edgecolor='black',linewidth=.25,alpha=.35)
        step=10
        arrows=ax.quiver(xx[::step,::step],yy[::step,::step],u[::step,::step],v[::step,::step],
                         transform=proj,scale=200,width=.0025)
        ax.set_xticks(np.arange(-60,-5,10) if i>=2 else [],crs=proj)
        ax.set_yticks(np.arange(-30,20,10) if i%2==0 else [],crs=proj)
        ax.set_title(f'{"abcd"[i]}) {name} | n={len(groups[g])}',loc='left',fontsize=13)
        if i>=2: ax.set_xlabel('Longitude (°)')
        if i%2==0: ax.set_ylabel('Latitude (°)')
    # Referência física comum a todos os painéis, posicionada abaixo dos mapas.
    axes.flat[2].quiverkey(arrows,.43,.055,10,'10 kg m⁻¹ s⁻¹',coordinates='figure',
                           labelpos='E',fontproperties={'size':11})
    cb=fig.colorbar(image,ax=axes,shrink=.72,pad=.03)
    cb.set_label('Convergência de umidade (kg m⁻² s⁻¹)')
    fig.suptitle(f'Quantis {label} aplicados a 1959–2023 | sem tendência | AMJJ\n'
                 f'Cores: 1000–850 hPa | setas: {transport[0]:g}–{transport[1]:g} hPa | grupo − normal',fontsize=15)
    for ext in ['png','pdf']:
        fig.savefig(out/f'ConvergenciaTransporte_{method}.{ext}',dpi=240)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--camada-transporte',type=float,nargs=2,default=[850,400])
    args=parser.parse_args()
    cached_maps=Path.home()/'.local/share/cartopy'
    if cached_maps.is_dir():
        cartopy.config['data_dir']=cached_maps
    root=args.base_dir.resolve()
    out=args.output_dir or root/'figuras/ComparacaoQuantisTransporteUmidade'
    out.mkdir(parents=True,exist_ok=True)
    table,groups,thresholds=classifications(root)
    transport=args.camada_transporte
    data,meta,used=annual_fluxes(root,transport)
    dx,dy=lat_lon_grid_deltas(meta['lon'],meta['lat'])
    outputs={}
    for method,selection in groups.items():
        composites={g:data[years-1959].mean(axis=0) for g,years in selection.items()}
        fields={}
        for g,_ in GROUPS:
            difference=composites[g]-composites['normal']
            conv=-divergence(difference[0],difference[1],dx=dx,dy=dy).magnitude
            fields[g]=(conv,difference[2],difference[3])
            outputs[f'{method}_{g}']=np.stack(fields[g])
        draw(out,method,fields,selection,meta,transport)
    table.to_csv(out/'ClassificacaoAnos.csv')
    pd.DataFrame({'data':used}).to_csv(out/'MesesUtilizados.csv',index=False)
    np.savez_compressed(out/'Composites.npz',lat=meta['lat'],lon=meta['lon'],**outputs)
    info=dict(arquivos=['outputs/detrended_uvq_u.npz','outputs/detrended_uvq_v.npz','outputs/detrended_uvq_q.npz'],
              meses=[4,5,6,7],periodo=[1959,2023],climatologia_TSM=[1980,2010],
              convergencia_hPa=[1000,850],transporte_hPa=transport,limites_quantis=thresholds,
              integracao='trapezios em Pa / 9.81; produtos u*q e v*q calculados antes das medias',
              convergencia='menos divergencia; mesma discretizacao MetPy de RealizarAnalises.py',
              comparacao='cada grupo menos seu normal; o grupo normal muda entre classificacoes',
              campos_NPZ=['convergencia','transporte_u','transporte_v'],
              anos={m:{g:y.tolist() for g,y in sel.items()} for m,sel in groups.items()})
    (out/'Metodologia.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(table.groupby('original').size())
    print(table.groupby('completo').size())
    print(out.resolve())


if __name__=='__main__':
    main()
