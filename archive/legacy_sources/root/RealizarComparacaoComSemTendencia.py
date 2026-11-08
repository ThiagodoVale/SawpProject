#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CT-ST para P90: quantis 1980-2010 e 1959-2023, climatologia 1980-2010.

Somente abril-julho. Cada versao CT/ST define seus proprios anos P90, como
em RealizarAnalises.py. Compara compostos absolutos, sem subtrair normal.
Painel d seleciona datas de cada arquivo e ordena dados pelos niveis reais.
Dependencias: numpy, pandas, matplotlib, metpy, cartopy e o modulo local
RealizarComparacaoQuantisTransporteUmidade.py.
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
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from metpy.calc import divergence, lat_lon_grid_deltas
from RealizarComparacaoQuantisTransporteUmidade import open_field, integrate


def carregar_corte_amjj(caminho, anos):
    """Corte (nivel, longitude), media dos quatro meses por ano selecionado.

Le time/valid_time do proprio arquivo. Retorna corte, niveis crescentes,
longitude e datas usadas. Latitude igual ao recorte original: -12.5 < lat <= -4.
"""
    anos=np.unique(np.asarray(anos,dtype=int))
    if not len(anos): raise ValueError('Grupo de anos vazio.')
    with ExitStack() as stack:
        f,shape,dtype,meta=open_field(stack,Path(caminho))
        dates=meta['time']
        if dates.to_period('M').duplicated().any(): raise ValueError('Meses duplicados.')
        lat=meta['lat']
        mask_lat=(lat>-12.5)&(lat<=-4)
        if not mask_lat.any(): raise ValueError('Recorte latitudinal vazio.')
        total=np.zeros((len(meta['level']),len(meta['lon'])))
        used=[]
        size=int(np.prod(shape[1:]))*dtype.itemsize
        for dt in dates:
            raw=f.read(size)
            if len(raw)!=size: raise ValueError('Arquivo truncado.')
            if dt.year not in anos or dt.month not in [4,5,6,7]: continue
            field=np.frombuffer(raw,dtype).reshape(shape[1:])[meta['order']][:,mask_lat,:]
            if not np.isfinite(field).all(): raise ValueError(f'Valores ausentes em {caminho}: {dt}')
            total+=field.astype(float).mean(axis=1)
            used.append(dt)
        selected=pd.DatetimeIndex(used)
        for year in anos:
            if sorted(selected[selected.year==year].month.tolist()) != [4,5,6,7]:
                raise ValueError(f'AMJJ incompleto em {caminho}, ano {year}.')
    return total/len(used),meta['level'],meta['lon'],selected


def sst_data(path):
    with np.load(path) as z:
        time=pd.DatetimeIndex(z['time']); lat=z['lat'];lon=z['lon'];values=z['data']
    if time.to_period('M').duplicated().any(): raise ValueError('Meses duplicados em TSM.')
    selected=(time.year>=1959)&(time.year<=2023)&time.month.isin([4,5,6,7])
    dt=time[selected]
    if len(dt)!=260: raise ValueError('TSM precisa de 260 meses AMJJ.')
    maps=[]
    for y in range(1959,2024):
        mask=(time.year==y)&time.month.isin([4,5,6,7])
        if sorted(time[mask].month.tolist())!=[4,5,6,7]: raise ValueError('AMJJ incompleto.')
        maps.append(values[mask].mean(axis=0))
    maps=np.stack(maps)
    ys=slice(np.argmin(abs(lat+4)),np.argmin(abs(lat+12.5)))
    xs=slice(np.argmin(abs(lon+34)),np.argmin(abs(lon+25)))
    s=pd.Series(maps[:,ys,xs].mean(axis=1).mean(axis=1),index=np.arange(1959,2024))
    anomaly=s-s.loc[1980:2010].mean()
    groups={}; thresholds={}
    for method,ref in [('original',anomaly.loc[1980:2010]),('completo',anomaly)]:
        q=ref.quantile(.9); groups[method]=anomaly.index[anomaly>=q].to_numpy();thresholds[method]=float(q)
    return maps,groups,thresholds,lat,lon,anomaly


def moisture(root,mode):
    folder,prefix=('outputs','detrended') if mode=='ST' else ('outputs_trended','trended')
    with ExitStack() as stack:
        files=[open_field(stack,root/folder/f'{prefix}_uvq_{v}.npz') for v in ['u','v','q']]
        meta=files[0][3]
        for _,_,_,m in files:
            for k in ['level','lat','lon']: np.testing.assert_array_equal(meta[k],m[k])
            if not meta['time'].equals(m['time']): raise ValueError('Datas de u/v/q diferentes.')
        if meta['time'].to_period('M').duplicated().any(): raise ValueError('Meses duplicados.')
        result=np.zeros((65,5,len(meta['lat']),len(meta['lon'])))
        counts=np.zeros(65,dtype=int)
        for dt in meta['time']:
            frames=[]
            selected=1959<=dt.year<=2023 and dt.month in [4,5,6,7]
            for f,shape,dtype,m in files:
                size=int(np.prod(shape[1:]))*dtype.itemsize; raw=f.read(size)
                if len(raw)!=size: raise ValueError('Arquivo truncado.')
                if selected: frames.append(np.frombuffer(raw,dtype).reshape(shape[1:])[m['order']].astype(float))
            if not selected: continue
            u,v,q=frames;p=meta['level'];y=dt.year-1959
            if not all(np.isfinite(a).all() for a in frames): raise ValueError('Campos invalidos.')
            result[y,0]+=integrate(q,p,850,400)
            result[y,1]+=integrate(u*q,p,850,400)
            result[y,2]+=integrate(v*q,p,850,400)
            result[y,3]+=integrate(u*q,p,1000,850)
            result[y,4]+=integrate(v*q,p,1000,850)
            counts[y]+=1
        if not (counts==4).all(): raise ValueError('AMJJ de umidade incompleto.')
    return result/4,meta


def draw(out,method,diff,lat,lon,p,counts,wind_scale):
    proj=ccrs.PlateCarree();fig=plt.figure(figsize=(13,11),layout='constrained');gs=fig.add_gridspec(2,2)
    axes=[fig.add_subplot(gs[0,0],projection=proj),fig.add_subplot(gs[0,1],projection=proj),
          fig.add_subplot(gs[1,0],projection=proj),fig.add_subplot(gs[1,1])]
    configs=[('sst','a) TSM','bwr',np.arange(-.5,.501,.1),'°C'),
             ('ap','b) Água precipitável | 850–400 hPa','BrBG',np.arange(-1,1.01,.2),'mm'),
             ('conv','c) Convergência 1000–850 | transporte 850–400 hPa','PuOr',np.arange(-30,31,10)/1e6,'kg m⁻² s⁻¹')]
    xx,yy=np.meshgrid(lon,lat)
    for ax,(key,title,cmap,levels,unit) in zip(axes,configs):
        ax.set_extent([-60,-10,-30,15],crs=proj)
        im=ax.contourf(lon,lat,diff[key],levels=levels,cmap=cmap,extend='both',alpha=.6,transform=proj)
        ax.coastlines(resolution='50m',linewidth=.6)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'),linewidth=.4,linestyle=':')
        ax.set_xticks(np.arange(-60,-5,10),crs=proj);ax.set_yticks(np.arange(-30,20,10),crs=proj)
        ax.set(xlabel='Longitude (°)',ylabel='Latitude (°)');ax.set_title(title,fontsize=11,loc='left')
        fig.colorbar(im,ax=ax,shrink=.75,pad=.02,label=unit)
        if key=='conv':
            ar=ax.quiver(xx[::10,::10],yy[::10,::10],diff['fu'][::10,::10],diff['fv'][::10,::10],
                         transform=proj,scale=200,width=.0025)
            ax.quiverkey(ar,.22,.025,10,'10 kg m⁻¹ s⁻¹',coordinates='figure',labelpos='E',fontproperties={'size':9})
    ax=axes[3];grid=np.linspace(100,1000,60)
    interpolated={v:np.column_stack([np.interp(grid,p,diff[v][:,j]) for j in range(len(lon))]) for v in ['te','u','w']}
    im=ax.contourf(lon,grid,interpolated['te'],levels=np.arange(-1,1.01,.2),cmap='RdYlBu_r',extend='both',alpha=.6)
    cross_ar=ax.quiver(lon[::8],grid[::2],2*interpolated['u'][::2,::8]/wind_scale[0],
                       -interpolated['w'][::2,::8]/wind_scale[1],pivot='mid',scale=50,width=.002)
    ax.quiverkey(cross_ar,.67,.025,1,'1 normalized unit (u and ω scaled independently)',
                 coordinates='figure',labelpos='E',fontproperties={'size':9})
    ax.axvspan(-40,-30,color='gray',alpha=.15)
    ax.set(xlim=(-60,-10),ylim=(1000,200),xlabel='Longitude (°)',ylabel='Pressão (hPa)',yticks=[1000,850,700,500,300,200])
    ax.set_title('d) ThetaE e circulação | exclusivamente AMJJ',fontsize=11,loc='left')
    fig.colorbar(im,ax=ax,shrink=.75,pad=.02,label='K')
    base='1980–2010' if method=='original' else '1959–2023'
    fig.suptitle(f'Com tendência − sem tendência | extremamente quente (≥P90)\n'
                 f'Quantis {base} | AMJJ de 1959–2023 | n CT={counts[0]}, n ST={counts[1]}',fontsize=15)
    for ext in ['png','pdf']:fig.savefig(out/f'Comparacao_CT_ST_{method}.{ext}',dpi=240)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();root=args.base_dir.resolve();out=args.output_dir or root/'figuras/ComparacaoComSemTendencia'
    out.mkdir(parents=True,exist_ok=True)
    cached=Path.home()/'.local/share/cartopy'
    if cached.is_dir():cartopy.config['data_dir']=cached
    all_fields={};group_info={};thresholds={};used=[];class_table=pd.DataFrame(index=np.arange(1959,2024))
    coordinates=None
    for mode in ['CT','ST']:
        folder,prefix=('outputs','detrended') if mode=='ST' else ('outputs_trended','trended')
        sstpath=root/folder/('detrended_sst.npz' if mode=='ST' else 'trended_uvq_sst.npz')
        maps,groups,qs,lat,lon,anom=sst_data(sstpath)
        class_table[f'anomalia_{mode}']=anom
        group_info[mode]={m:y.tolist() for m,y in groups.items()};thresholds[mode]=qs
        print(f'Processando umidade {mode}...',flush=True)
        moist,meta=moisture(root,mode)
        np.testing.assert_array_equal(meta['lat'],lat);np.testing.assert_array_equal(meta['lon'],lon)
        if coordinates is not None:
            np.testing.assert_array_equal(coordinates[0],lat);np.testing.assert_array_equal(coordinates[1],lon)
        coordinates=(lat,lon)
        dx,dy=lat_lon_grid_deltas(lon,lat)
        for method,years in groups.items():
            class_table[f'P90_{mode}_{method}']=class_table.index.isin(years)
            fields={'sst':maps[years-1959].mean(axis=0)}
            a,fu,fv,fl_u,fl_v=moist[years-1959].mean(axis=0)
            fields.update(ap=a,fu=fu,fv=fv,conv=-divergence(fl_u,fl_v,dx=dx,dy=dy).magnitude)
            print(f'Cortes AMJJ {mode}, {method}: {years.tolist()}',flush=True)
            for var in ['te','u','w']:
                cut,p,cutlon,dates=carregar_corte_amjj(root/folder/f'{prefix}_uvq_{var}.npz',years)
                np.testing.assert_array_equal(p,meta['level']);np.testing.assert_array_equal(cutlon,lon)
                fields[var]=cut
                used.extend(dict(versao=mode,base=method,variavel=var,data=str(dt.date())) for dt in dates)
            all_fields[(mode,method)]=fields
        del maps,moist
    differences={m:{k:all_fields[('CT',m)][k]-all_fields[('ST',m)][k] for k in all_fields[('CT',m)]} for m in ['original','completo']}
    grid=np.linspace(100,1000,60)
    scale=[]
    for v in ['u','w']:
        values=[np.column_stack([np.interp(grid,p,d[v][:,j]) for j in range(len(lon))]) for d in differences.values()]
        scale.append(max(float(np.std(a)) for a in values) or 1.)
    for method,diff in differences.items():
        draw(out,method,diff,lat,lon,p,[len(group_info[v][method]) for v in ['CT','ST']],scale)
    np.savez_compressed(out/'Campos_CT_ST.npz',lat=lat,lon=lon,level=p,**{f'{m}_{k}':v for m,d in differences.items() for k,v in d.items()})
    class_table.to_csv(out/'ClassificacaoAnos.csv',index_label='ano')
    pd.DataFrame(used).to_csv(out/'DatasCortesThetaE_Circulacao.csv',index=False)
    (out/'Metodologia.json').write_text(json.dumps(dict(climatologia=[1980,2010],periodo=[1959,2023],meses=[4,5,6,7],
        anos_P90=group_info,limites_P90=thresholds,comparacao='P90 CT menos P90 ST; selecoes independentes; sem subtrair normal',
        thetae='datas e niveis do proprio arquivo; niveis ordenados com os dados',
        vetores_corte='u e w normalizados por escalas comuns as duas figuras',escalas_vetores=scale),indent=2,ensure_ascii=False),encoding='utf-8')
    print(out.resolve())


if __name__=='__main__':main()
