#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compara bases de quantis de TSM em cortes u/w + ThetaE e contraste vertical.

Dados sem tendencia; exclusivamente AMJJ de 1959-2023; climatologia 1980-2010.
Figura original: limites de quantis 1980-2010. Figura completa: 1959-2023.
Cortes e barras na figura usam anomalias grupo menos normal.
Contraste = integral(100..600 hPa) - integral(600..1000 hPa), em K hPa.
Media SAWP: -12.5 < latitude <= -4 e -34 <= longitude <= -25.
CSV inclui integrais absolutas, anomalias e numero de anos.
Requer os modulos locais RealizarGradienteEstabilidadeVertical.py e
RealizarComparacaoQuantisTransporteUmidade.py.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from RealizarGradienteEstabilidadeVertical import read_sections, composites, integrate_layer, GROUPS
from RealizarComparacaoQuantisTransporteUmidade import classifications


def draw(out,method,comp,meta,summary):
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
        ax.quiver(lon[::8],grid[::2],2*fields['u'][::2,::8]/scales['u'],
                  -fields['w'][::2,::8]/scales['w'],color='black',pivot='mid',scale=50,width=.002)
        ax.fill_betweenx([1000,200],-40,-30,color='gray',alpha=.2)
        ax.set(xlim=(-60,-10),ylim=(1000,200))
        ax.set_title(f'{"abcd"[i]})',loc='left',fontsize=18)
        ax.set_yticks(np.arange(1000,200,-200) if i%2==0 else [])
        ax.set_xticks(np.linspace(lon.min(),lon.max(),5) if i>=2 else [])
        ax.tick_params(labelsize=14)
    fig.text(.465,.295,'Longitude',ha='center',fontsize=18)
    fig.text(.025,.63,'Pressure Levels (hPa)',va='center',rotation='vertical',fontsize=18)
    key_ax=fig.add_axes([.10,.245,.73,.025])
    key_ax.set(xlim=(0,1),ylim=(0,1)); key_ax.axis('off')
    key=key_ax.quiver([.40],[.5],[1],[0],angles='xy',scale_units='xy',scale=15,
                       color='black',width=.008)
    key_ax.quiverkey(key,.40,.5,1,'1 SD: normalized u–ω anomaly',
                     coordinates='axes',labelpos='E',fontproperties={'size':10})
    key.set_visible(False)
    cax=fig.add_axes([.86,.465,.018,.33])
    cb=fig.colorbar(im,cax=cax)
    cb.set_label('Equivalent Potential Temperature (K)',size=14)
    cb.ax.tick_params(labelsize=12)
    ax=fig.add_axes([.10,.065,.73,.13])
    labels=['Very cold','Moderately cold','Moderately warm','Extremely warm']
    values=[float(summary.loc[g,'anomalia_contraste_K_hPa']) for g,_,_ in GROUPS]
    ax.bar(np.arange(4),values,color=[color for _,_,color in GROUPS],width=.55,zorder=3)
    ax.axhline(0,color='#555555',linewidth=.8)
    ax.set_xticks(np.arange(4),labels,fontsize=11)
    ax.set_ylim(-400,850)
    ax.set_ylabel('K·hPa',fontsize=12)
    ax.set_title('ThetaE gradient',fontsize=15,pad=8)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#eeeeee',linewidth=.6,zorder=0)
    for i,value in enumerate(values):
        ax.annotate(f'{value:.1f}',(i,value),xytext=(0,5 if value>=0 else -5),
                    textcoords='offset points',ha='center',va='bottom' if value>=0 else 'top',fontsize=10)
    ax.text(.5,-.36,'600–100 hPa − 1000–600 hPa | Anomalies relative to the normal category',
            ha='center',va='top',transform=ax.transAxes,fontsize=10)
    for ext in ['png','pdf']:fig.savefig(out/f'ThetaE_Quantis_{method}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();root=args.base_dir.resolve()
    out=args.output_dir or root/'figuras/ComparacaoQuantisThetaE';out.mkdir(parents=True,exist_ok=True)
    table,groups,thresholds=classifications(root)
    comp={m:{} for m in groups};metadata=None;annual=None
    for var in ['te','u','w']:
        print(f'Lendo {var} sem tendencia, somente AMJJ...',flush=True)
        data,dates,meta=read_sections(root/'outputs'/f'detrended_uvq_{var}.npz',groups['original'],metadata)
        if metadata is None:metadata=meta
        np.testing.assert_array_equal(meta['level'],metadata['level'])
        for method,selection in groups.items():comp[method][var]=composites(data,dates,selection)
        if var=='te':
            mask=(meta['lon']>=-34)&(meta['lon']<=-25);rows=[]
            for year in range(1959,2024):
                selected=dates[dates.year==year]
                if sorted(selected.month.tolist())!=[4,5,6,7]:raise ValueError('AMJJ incompleto.')
                profile=data[dates.year==year].mean(axis=0)[:,mask].mean(axis=1)
                lower=integrate_layer(profile,meta['level'],600,1000)
                upper=integrate_layer(profile,meta['level'],100,600)
                rows.append(dict(ano=year,inferior_K_hPa=lower,superior_K_hPa=upper,contraste_K_hPa=upper-lower))
            annual=pd.DataFrame(rows).set_index('ano')
            pd.DataFrame({'data':dates}).to_csv(out/'MesesUtilizados.csv',index=False)
        del data
    summaries={}
    for method in groups:
        a=annual.copy();a['grupo']=table[method]
        summary=a.groupby('grupo').mean()
        summary['n_anos']=a.groupby('grupo').size()
        for v in ['inferior','superior','contraste']:
            summary[f'anomalia_{v}_K_hPa']=summary[f'{v}_K_hPa']-summary.loc['normal',f'{v}_K_hPa']
        for g in groups[method]:
            profile=(comp[method]['te'][g]-comp[method]['te']['normal'])[:,mask].mean(axis=1)
            direct=integrate_layer(profile,meta['level'],100,600)-integrate_layer(profile,meta['level'],600,1000)
            np.testing.assert_allclose(direct,summary.loc[g,'anomalia_contraste_K_hPa'],atol=1e-6)
        summaries[method]=summary
        summary.to_csv(out/f'TabelaThetaE_{method}.csv')
        a.to_csv(out/f'IntegraisPorAno_{method}.csv')
    for method in groups:draw(out,method,comp[method],metadata,summaries[method])
    pd.concat(summaries,names=['base','grupo']).to_csv(out/'TabelaComparativaThetaE.csv')
    table.to_csv(out/'ClassificacaoAnos.csv')
    np.savez_compressed(out/'CortesThetaE_u_w.npz',level=metadata['level'],lon=metadata['lon'],
                        **{f'{m}_{v}_{g}':a for m,vs in comp.items() for v,gs in vs.items() for g,a in gs.items()})
    (out/'Metodologia.json').write_text(json.dumps(dict(climatologia=[1980,2010],periodo=[1959,2023],meses=[4,5,6,7],
        limites_quantis=thresholds,anos={m:{g:y.tolist() for g,y in sel.items()} for m,sel in groups.items()},
        latitude='-12.5 < lat <= -4',longitude_integracao=[-34,-25],
        integracao='trapezios em hPa; superior(100..600) menos inferior(600..1000); K hPa',
        figuras='cortes e barras: grupo menos normal da propria classificacao',
        CSV='inclui tambem integrais absolutas; variaveis anomalia_* sao relativas ao normal',
        vetores='anomalias de u e w normalizadas pelo desvio espacial de cada componente em cada painel, como no original; fator u=2, sinal w invertido, scale=50, width=0.002'),
        ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.concat(summaries)[['n_anos','anomalia_contraste_K_hPa']].to_string())
    print(out.resolve())


if __name__=='__main__':main()
