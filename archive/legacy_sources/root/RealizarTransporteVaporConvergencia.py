"""Reproduce RealizarAnalises.py moisture panels, adding a 10 kg m-1 s-1 quiver key.
AMJJ 1959-2023; detrended inputs; SST quantiles/climatology 1980-2010.
Preserves the original shaded divergence sign (negative denotes convergence).
Run: python RealizarTransporteVaporConvergencia.py
"""
from pathlib import Path
import argparse
import importlib.util
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.quiver import Quiver
import cartopy
import cartopy.crs as ccrs
from metpy.calc import divergence, lat_lon_grid_deltas
from RealizarComparacaoQuantisTransporteUmidade import classifications, annual_fluxes


def draw(root, out, fields, meta):
    # Load the original plotting function without importing sawp's GPU modules.
    spec=importlib.util.spec_from_file_location('original_graphics',root/'sawp/Graficos.py')
    graphics=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graphics)
    fig,axs=plt.subplots(2,2,figsize=(12,10),subplot_kw={'projection':ccrs.PlateCarree()})
    fig.subplots_adjust(hspace=.01,wspace=.15)
    lon,lat=np.meshgrid(meta['lon'],meta['lat'])
    arrows=[]
    magnitudes=[]
    for ax,g,letter in zip(axs.flat,['10','25','75','90'],'abcd'):
        div,u,v=fields[g]
        im=graphics.plotar_grafico_sombreado_vetorial(lon,lat,u,v,div,
            'Moisture flux convergence (kg m⁻² s⁻¹)','k',None,ax,identificador=letter)
        q=next(c for c in ax.collections if isinstance(c,Quiver))
        arrows.append(q)
        speed=np.hypot(q.U,q.V)
        valid=np.isfinite(speed)&(speed>0)
        if np.ndim(q.Umask):valid &= ~q.Umask
        magnitudes.extend(speed[valid].tolist())
        ax.set_title(f'{letter})',loc='left',fontsize=18)
    if not magnitudes:raise ValueError('No finite nonzero vectors available for the reference.')
    # User-selected reference; use the original Quiver object and its scale.
    reference=10.0
    assert all(q.scale==arrows[0].scale and q.units==arrows[0].units for q in arrows)
    fig.text(.45,.05,'Longitude',ha='center',fontsize=20)
    fig.text(.045,.5,'Latitude',va='center',rotation='vertical',fontsize=20)
    cb=fig.colorbar(im,ax=axs,orientation='vertical',shrink=.5,pad=.02)
    cb.set_label('Moisture flux convergence (kg m⁻² s⁻¹)',size=15)
    cb.ax.tick_params(labelsize=12)
    key=axs.flat[3].quiverkey(arrows[3],.70,.055,reference,
        f'{reference:g} kg m⁻¹ s⁻¹',coordinates='figure',labelpos='E',fontproperties={'size':12})
    fig.canvas.draw()
    for ext in ['png','pdf']:
        fig.savefig(out/f'TRSP_AnomaliasQuantis_Subplot.{ext}',dpi=360,bbox_inches='tight',bbox_extra_artists=(*fig.texts,key.text))
    plt.close(fig)
    return reference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    root=args.base_dir.resolve()
    out=args.output_dir or root/'figuras/TransporteVaporConvergencia'
    out.mkdir(parents=True,exist_ok=True)
    cached=Path.home()/'.local/share/cartopy'
    if cached.is_dir():cartopy.config['data_dir']=cached
    table,groups,thresholds=classifications(root)
    data,meta,used=annual_fluxes(root,[850,400])
    selection=groups['original']
    composites={g:data[years-1959].mean(axis=0) for g,years in selection.items()}
    dx,dy=lat_lon_grid_deltas(meta['lon'],meta['lat'])
    fields={}
    for g in ['10','25','75','90']:
        d=composites[g]-composites['normal']
        fields[g]=(divergence(d[0],d[1],dx=dx,dy=dy).magnitude,d[2],d[3])
    reference=draw(root,out,fields,meta)
    table.to_csv(out/'ClassificacaoAnos.csv')
    np.savez_compressed(out/'Composites.npz',lat=meta['lat'],lon=meta['lon'],**{g:np.stack(v) for g,v in fields.items()})
    (out/'Metodologia.json').write_text(json.dumps(dict(
        quantiles=[1980,2010],period=[1959,2023],months=[4,5,6,7],detrended=True,
        transport_hPa=[850,400],shading_hPa=[1000,850],
        shading='Original divergence convention: negative values denote convergence',
        reference_value=reference,reference_units='kg m-1 s-1',
        reference_method='User-selected 10 kg m-1 s-1; original Quiver object and scale',
        thresholds=thresholds,dates=used),indent=2),encoding='utf-8')
    print(f'Reference: {reference:g} kg m-1 s-1; output: {out}',flush=True)


if __name__=='__main__':main()

