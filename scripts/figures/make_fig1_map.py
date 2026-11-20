"""
make_fig1_map.py -- Fig. 1: location map of the nine marine seismic lines.

Panels: (A) Mediterranean Sea; (B) South America and Antarctica locator;
(C) Ross Sea; (D) Drake Passage. Blue = training pool, orange = held-out.

Inputs (under MAPS_ROOT, default ./data/maps; see data/README.md):
  survey_tracks.geojson  track geometry of the nine lines (in the repository);
               used when navigation/ is absent
  navigation/  original navigation files (not public): GC_01_Nav_UTM32.txt,
               GC_03_Nav_UTM32.txt, CROPM31.NAV, IT89-33.txt, IT90AR65.uko,
               crop.kml; when present the tracks are read from here and the
               navigation tables and the GeoJSON are rewritten
  bathymetry/  med.npz, ross.npz, drake.npz  (NOAA ETOPO 2022, 1 arc-min;
               scripts/data_prep/download_map_data.py)
  basemap/     Natural Earth ne_10m_land, ne_50m_land,
               ne_50m_antarctic_ice_shelves_polys

Navigation assumptions (also written to navigation_summary.csv):
  * CG-1, CG-3: easting/northing divided by 10, WGS84 / UTM 32N (EPSG:32632).
  * CROP-M18, CROP-M32: line geometry from the CROP KML (no shot points).
  * IT89-33: antenna positions, no lever-arm correction.
  * SISOC-MEPA, IT97-245: geodesic connector between the two end points;
    intermediate navigation is not available.
  * The full lines are drawn, not only the 301 shot points used in the study.

Outputs (under RESULTS_ROOT/figures and RESULTS_ROOT/map):
  Fig_1.{pdf,svg,png}, navigation_points.csv, navigation_summary.csv,
  survey_tracks.geojson

Dependencies: numpy, scipy, matplotlib, cartopy, pyproj, shapely, pyshp.
No network access is needed.

Usage:
  python scripts/figures/make_fig1_map.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import re, json, csv, xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from shapely.geometry import box
from pyproj import Transformer, Geod

from config import MAPS_ROOT, RESULTS_ROOT

IN=MAPS_ROOT/'navigation'; BASE=MAPS_ROOT/'basemap'; BATHY_DIR=MAPS_ROOT/'bathymetry'
FIG_DIR=RESULTS_ROOT/'figures'; MAP_DIR=RESULTS_ROOT/'map'
FIG_DIR.mkdir(parents=True,exist_ok=True); MAP_DIR.mkdir(parents=True,exist_ok=True)
GEOD=Geod(ellps='WGS84'); PC=ccrs.PlateCarree()
BLUE='#0072B2'; ORANGE='#D55E00'; OCEAN='#EDF5F8'; LAND='#E9E5DC'
BATHY_CMAP=LinearSegmentedColormap.from_list('pale_bathymetry',['#f5fbfc','#e2eff3','#c6dfe8','#a5c7d8','#83a9bd','#6f91a7'])
BATHY_NORM=Normalize(vmin=0,vmax=5000,clip=True)
BATHY={key:dict(np.load(BATHY_DIR/f'{key}.npz')) for key in ['med','ross','drake']}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
tracks={}
def add(name,lon,lat,sp,source,role='Training pool',note='',approx=False):
    lon=np.asarray(lon);lat=np.asarray(lat);sp=np.asarray(sp)
    tracks[name]=dict(lon=lon,lat=lat,sp=sp,source=source,role=role,note=note,approx=approx)

def load_tracks_geojson(path):
    """Read the published track geometry (survey_tracks.geojson) instead of the
    original navigation files. Undoes the split at the 180 deg meridian made by
    dateline_parts(), so the tracks are identical to the ones read from the
    navigation; shot-point numbers are not stored in the GeoJSON."""
    for feat in json.loads(Path(path).read_text())['features']:
        g=feat['geometry'];pr=feat['properties']
        if g['type']=='LineString':
            pts=g['coordinates']
        else:
            parts=g['coordinates'];pts=list(parts[0][:-1])
            for part in parts[1:-1]:pts+=part[1:-1]
            pts+=parts[-1][1:]
        a=np.array(pts,dtype=float)
        add(pr['name'],a[:,0],a[:,1],np.full(len(a),np.nan),'survey_tracks.geojson',pr['role'],pr['note'],pr['approximate'])


TRACKS_GEOJSON=MAPS_ROOT/'survey_tracks.geojson'
if IN.is_dir():
    # original navigation files: also writes the navigation tables and the GeoJSON
    tf=Transformer.from_crs(32632,4326,always_xy=True)
    for file,name,role in [('GC_01_Nav_UTM32.txt','CG-1','Training pool'),('GC_03_Nav_UTM32.txt','CG-3','Held-out survey')]:
        a=np.loadtxt(IN/file); lon,lat=tf.transform(a[:,1]/10,a[:,2]/10)
        add(name,lon,lat,a[:,0],file,role,'Coordinates divided by 10; WGS84 / UTM zone 32N assumed, datum and scalar to be confirmed. Repeated SP retained in file order. GC file names mapped to CG manuscript names.')
    a=np.loadtxt(IN/'CROPM31.NAV',usecols=(0,1,2))
    add('CROP-M31',a[:,0],a[:,1],a[:,2],'CROPM31.NAV',note='Longitude, latitude, SP; datum not specified, treated as WGS84 for this regional draft.')
    ns={'k':'http://www.opengis.net/kml/2.2'}
    for pm in ET.parse(IN/'crop.kml').findall('.//k:Placemark',ns):
        name=pm.findtext('k:name',namespaces=ns)
        if name in ['M18','M32']:
            a=np.array([[float(v) for v in z.split(',')[:2]] for z in pm.find('.//k:coordinates',ns).text.split()])
            add('CROP-'+name,a[:,0],a[:,1],np.full(len(a),np.nan),'crop.kml:'+name,'Held-out survey' if name=='M18' else 'Training pool','KML geographic line geometry; no shotpoint identifiers supplied. Used as a location trace, not per-shot navigation.')
    a=np.array([[float(v) for v in line.split()] for line in (IN/'IT89-33.txt').read_text().splitlines() if line.strip() and not line.strip().startswith('H')])
    add('IT89-33',a[:,1],a[:,0],a[:,2],'IT89-33.txt',note='Header: WGS84 spheroid; coordinates of antenna, source approximately 60 m from antenna. No lever-arm correction applied. Navigation sampled approximately every 10 SP. Antimeridian crossing handled.')
    rows=[]
    pat=re.compile(r'^\s*\S+\s+(\d+?)(\d{2})(\d{2})(\d{2}\.\d+)S(\d{3})(\d{2})(\d{2}\.\d+)E')
    for line in (IN/'IT90AR65.uko').read_text().splitlines():
        m=pat.match(line)
        if not m: raise ValueError('Unparsed IT90 record: '+line)
        sp,ld,lm,ls,od,om,os=map(float,m.groups());rows.append([od+om/60+os/3600,-(ld+lm/60+ls/3600),sp])
    a=np.array(rows)
    add('IT90-65',a[:,0],a[:,1],a[:,2],'IT90AR65.uko',note='Fixed-width SP and latitude/longitude DMS; S and E signs applied. Datum not specified, treated as WGS84 for this regional draft. Navigation sampled approximately every 10 SP.')
    x1,y1=18+50/60+47/3600,40+17/60+52/3600
    x2,y2=18+41/60+40/3600,40+42/60+51/3600
    points=np.array([(x1,y1)]+GEOD.npts(x1,y1,x2,y2,98)+[(x2,y2)])
    add('SISOC-MEPA',points[:,0],points[:,1],np.r_[100,np.full(98,np.nan),2031],'End points (SP 100, SP 2031)',note='Geodesic endpoint connector only, not measured navigation. Endpoint distance 48.00 km versus 96.55 km from (2031-100)*50 m; intermediate SP deliberately not generated. Datum unspecified.',approx=True)


    x1,y1=-67.566,-56.1004
    x2,y2=-69.281,-57.506
    points=np.array([(x1,y1)]+GEOD.npts(x1,y1,x2,y2,98)+[(x2,y2)])
    add('IT97-245',points[:,0],points[:,1],np.r_[100,np.full(98,np.nan),3870],'End points (SP 100, SP 3870)','Held-out survey',note='Geodesic endpoint connector only, not measured navigation. Endpoints 67.566 W / 56.1004 S and 69.281 W / 57.506 S. Endpoint distance 188.340 km versus 188.500 km from (3870-100)*50 m. Intermediate SP not assigned; datum unspecified.',approx=True)

    summary=[]
    with (MAP_DIR/'navigation_points.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['survey','point_index','shotpoint','longitude_deg','latitude_deg','status'])
        for name,t in tracks.items():
            for i,(sp,x,y) in enumerate(zip(t['sp'],t['lon'],t['lat'])):
                writer.writerow([name,i,'' if np.isnan(sp) else int(sp),f'{x:.9f}',f'{y:.9f}','endpoint_connector' if t['approx'] else ('kml_line_geometry' if t['source'].startswith('crop.kml') else 'supplied_navigation')])
            _,_,ds=GEOD.inv(t['lon'][:-1],t['lat'][:-1],t['lon'][1:],t['lat'][1:])
            summary.append(dict(survey=name,source=t['source'],role=t['role'],points=len(t['lon']),unique_SP=len(np.unique(t['sp'][~np.isnan(t['sp'])])),SP_first='' if np.isnan(t['sp'][0]) else int(t['sp'][0]),SP_last='' if np.isnan(t['sp'][-1]) else int(t['sp'][-1]),mapped_length_km=round(float(sum(ds))/1000,3),note=t['note']))
    with (MAP_DIR/'navigation_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)

    def dateline_parts(xs,ys):
        parts=[[[float(xs[0]),float(ys[0])]]]
        for x0,y0,x,y in zip(xs[:-1],ys[:-1],xs[1:],ys[1:]):
            if abs(x-x0)>180:
                xu=x-360 if x>x0 else x+360
                boundary=-180 if xu<x0 else 180
                yc=y0+(y-y0)*(boundary-x0)/(xu-x0)
                parts[-1].append([boundary,float(yc)]);parts.append([[-boundary,float(yc)]])
            parts[-1].append([float(x),float(y)])
        return parts
    features=[]
    for name,t in tracks.items():
        parts=dateline_parts(t['lon'],t['lat'])
        geo={'type':'LineString','coordinates':parts[0]} if len(parts)==1 else {'type':'MultiLineString','coordinates':parts}
        features.append({'type':'Feature','properties':{'name':name,'role':t['role'],'approximate':t['approx'],'note':t['note']},'geometry':geo})
    (MAP_DIR/'survey_tracks.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features},indent=2))
else:
    print(f'{IN} not found: drawing the tracks from {TRACKS_GEOJSON}')
    load_tracks_geojson(TRACKS_GEOJSON)


land10=list(Reader(BASE/'ne_10m_land/ne_10m_land.shp').geometries())
land50=list(Reader(BASE/'ne_50m_land/ne_50m_land.shp').geometries())
ice=list(Reader(BASE/'ne_50m_antarctic_ice_shelves_polys/ne_50m_antarctic_ice_shelves_polys.shp').geometries())
def bathymetry(ax,extent,polar=False):
    d=BATHY['ross' if polar else ('drake' if extent[0]<0 else 'med')]
    xs=d['longitude'];ys=d['latitude'];z=d['elevation']
    if polar:
        ix=np.arange(0,len(xs),2);iy=np.arange(0,len(ys),2)
        levels=[200,500,1000,2000]
    else:
        ix=np.where((xs>=extent[0]-.06)&(xs<=extent[1]+.06))[0]
        iy=np.where((ys>=extent[2]-.06)&(ys<=extent[3]+.06))[0]
        if extent[1]-extent[0]>5:ix=ix[::2];iy=iy[::2];levels=[200,1000,3000]
        elif extent[2]>44:levels=[10,20,30,40,50,100]
        else:levels=[100,200,500,1000,2000,3000]
    xs=xs[ix];ys=ys[iy];depth=-z[np.ix_(iy,ix)]
    depth=np.ma.masked_where((depth<=0)|(depth>12000),depth)
    xe=np.r_[xs[0]-(xs[1]-xs[0])/2,(xs[:-1]+xs[1:])/2,xs[-1]+(xs[-1]-xs[-2])/2]
    ye=np.r_[ys[0]-(ys[1]-ys[0])/2,(ys[:-1]+ys[1:])/2,ys[-1]+(ys[-1]-ys[-2])/2]
    xx,yy=np.meshgrid(xe,ye);pp=ax.projection.transform_points(PC,xx,yy)
    ax.pcolormesh(pp[:,:,0],pp[:,:,1],depth,cmap=BATHY_CMAP,norm=BATHY_NORM,shading='flat',rasterized=True,zorder=.2)
    xx,yy=np.meshgrid(xs,ys);p=ax.projection.transform_points(PC,xx,yy)
    c=ax.contour(p[:,:,0],p[:,:,1],depth,levels=levels,colors='#7294a4',linewidths=.32,alpha=.65,zorder=.5)
    if not polar and extent[1]-extent[0]<5:
        labels=ax.clabel(c,fmt='%d',fontsize=5.5,inline=True,inline_spacing=2)
        for lab in labels:lab.set_clip_on(True)

def coast(ax,extent,polar=False):
    ax.set_facecolor(OCEAN)
    bathymetry(ax,extent,polar)
    shapes=land50 if polar else land10
    if not polar:
        clip=box(extent[0]-1,extent[2]-1,extent[1]+1,extent[3]+1)
        shapes=[g.intersection(clip) for g in shapes if g.intersects(clip)]
    ax.add_geometries(shapes,PC,facecolor=LAND,edgecolor='#777970',linewidth=.5,zorder=1)
    if polar:ax.add_geometries(ice,PC,facecolor='#DCE7EC',edgecolor='#9DAEB5',linewidth=.45,zorder=2)
    ax.set_extent(extent,crs=PC)
    gl=ax.gridlines(crs=PC,draw_labels=True,linewidth=.4,color='#8DABB7',alpha=.55,linestyle=':',zorder=3)
    gl.top_labels=False;gl.right_labels=False;gl.xlabel_style={'size':7};gl.ylabel_style={'size':7}
    if polar:
        gl.xlocator=FixedLocator([160,170,180,-170]);gl.ylocator=FixedLocator([-80,-78,-76,-74,-72]);gl.rotate_labels=False
    ax.spines['geo'].set_linewidth(.7);ax.spines['geo'].set_edgecolor('#768790')

def plottrack(ax,name,lw=1.8):
    t=tracks[name];color=ORANGE if t['role']=='Held-out survey' else BLUE
    p=ax.projection.transform_points(PC,t['lon'],t['lat'])
    line,=ax.plot(p[:,0],p[:,1],color=color,lw=lw,ls='-',zorder=6)
    line.set_path_effects([pe.Stroke(linewidth=lw+1,foreground='white',alpha=.9),pe.Normal()])
    ax.scatter(p[[0,-1],0],p[[0,-1],1],s=12,facecolor='white',edgecolor=color,lw=.8,zorder=7)
def text(ax,x,y,s,size=9,color='#53636D',**kw):
    return ax.text(x,y,s,transform=PC,fontsize=size,color=color,zorder=8,**kw)
def label(ax,name,xytext,frac=.5,textname=None):
    t=tracks[name];i=int(frac*(len(t['lon'])-1));x,y=t['lon'][i],t['lat'][i]
    color=ORANGE if t['role']=='Held-out survey' else BLUE
    ax.annotate(textname or name,xy=(x,y),xycoords=PC._as_mpl_transform(ax),xytext=xytext,textcoords=PC._as_mpl_transform(ax),fontsize=9,fontweight='bold',color=color,zorder=10,bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':1.6})
def scalebar(ax,x,y,km,label_offset=0,horizontal=False):
    xx,yy,_=GEOD.fwd(x,y,90,km*1000)
    pts=np.array([(x,y)]+GEOD.npts(x,y,xx,yy,25)+[(xx,yy)])
    p=ax.projection.transform_points(PC,pts[:,0],pts[:,1])
    if horizontal:
        from scipy.optimize import brentq
        x0,y0=ax.projection.transform_point(x,y,PC)
        def distance(dx):
            lon,lat=PC.transform_point(x0+dx,y0,ax.projection)
            return GEOD.inv(x,y,lon,lat)[2]-km*1000
        dx=brentq(distance,1,km*10000)
        p=np.column_stack([np.linspace(x0,x0+dx,27),np.full(27,y0)])
    ax.plot(p[:,0],p[:,1],color='#263A45',lw=2,zorder=10)
    ax.plot(p[[0,-1],0],p[[0,-1],1],color='#263A45',marker='|',markersize=5,ls='',zorder=10)
    ax.annotate(f'{km} km',xy=(p[len(p)//2,0],p[len(p)//2,1]),xytext=(0,label_offset),textcoords='offset points',fontsize=7,ha='center',va='bottom',color='#263A45',zorder=11,path_effects=[pe.withStroke(linewidth=3,foreground=OCEAN)])
def title(ax,letter,name):ax.set_title(f'{letter}  {name}',loc='left',fontsize=11,fontweight='bold',pad=11,color='#243744')

fig=plt.figure(figsize=(12,8.5),facecolor='white')
axA=fig.add_axes([.055,.15,.44,.80],projection=ccrs.Mercator(central_longitude=14))
axB=fig.add_axes([.60,.49,.32,.45],projection=ccrs.Orthographic(central_longitude=-95,central_latitude=-55))
axC=fig.add_axes([.545,.155,.215,.27],projection=ccrs.SouthPolarStereo(central_longitude=180))
axD=fig.add_axes([.79,.155,.20,.27],projection=ccrs.Mercator(central_longitude=-68))
coast(axA,[7.4,20.6,35.5,46.2]);title(axA,'A','Mediterranean Sea')
for name in ['CG-1','CG-3','CROP-M31','CROP-M32','CROP-M18','SISOC-MEPA']:plottrack(axA,name,1.9)
label(axA,'CROP-M18',(13.9,45.8))
label(axA,'CROP-M31',(15.2,37.05))
label(axA,'SISOC-MEPA',(18.9,40.5),textname='MEPA')
label(axA,'CG-1',(9.75,39.23));label(axA,'CG-3',(9.85,38.63));label(axA,'CROP-M32',(10.0,37.85))
text(axA,8.0,40.4,'Sardinia',8);text(axA,12.9,37.6,'Sicily',8);text(axA,11.8,43.5,'ITALY',10)
text(axA,11.1,39.9,'Tyrrhenian Sea',9,fontstyle='italic');text(axA,17.0,36.15,'Ionian Sea',9,fontstyle='italic')
scalebar(axA,8,35.9,200)
axB.set_global();axB.set_ylim(-6378137,3500000);axB.set_facecolor(OCEAN)
axB.add_geometries(land50,PC,facecolor=LAND,edgecolor='#777970',linewidth=.45,zorder=1)
axB.add_geometries(ice,PC,facecolor='#DCE7EC',edgecolor='#9DAEB5',linewidth=.35,zorder=2)
axB.gridlines(xlocs=np.arange(-180,181,30),ylocs=np.arange(-90,91,20),linewidth=.4,color='#8DABB7',alpha=.5,linestyle=':')
axB.spines['geo'].set_edgecolor('#768790');axB.spines['geo'].set_linewidth(.7)
title(axB,'B','South America and Antarctica')
text(axB,-64,-35,'South\nAmerica',9,ha='center')
text(axB,-75,-83,'Antarctica',9,ha='center')
for extent,ltr,pos in [([158,193,-80,-71],'C',(174,-68)),([-72.2,-64.5,-58.5,-54.2],'D',(-62,-58))]:
    x0,x1,y0,y1=extent
    xs=np.r_[np.linspace(x0,x1,60),np.full(60,x1),np.linspace(x1,x0,60),np.full(60,x0)]
    ys=np.r_[np.full(60,y0),np.linspace(y0,y1,60),np.full(60,y1),np.linspace(y1,y0,60)]
    axB.plot(xs,ys,transform=PC,color='#465C69',lw=1.1,zorder=5)
    text(axB,*pos,ltr,11,fontweight='bold',color='#243744')
for name in ['IT89-33','IT90-65','IT97-245']:plottrack(axB,name,1.4)
coast(axC,[158,193,-80,-71],True);title(axC,'C','Ross Sea')
for name in ['IT89-33','IT90-65']:plottrack(axC,name,1.9)
label(axC,'IT89-33',(-179.5,-76.9));label(axC,'IT90-65',(160,-74.95))
text(axC,178,-73,'Ross Sea',8,fontstyle='italic',ha='center');text(axC,179,-79,'Ross Ice Shelf',7,ha='center')
text(axC,161,-77.7,'Victoria\nLand',6,ha='center');scalebar(axC,172,-80,200,label_offset=5,horizontal=True)
coast(axD,[-72.2,-64.5,-58.5,-54.2]);title(axD,'D','Drake Passage')
plottrack(axD,'IT97-245',2);label(axD,'IT97-245',(-67.6,-57.25))
text(axD,-71.6,-54.7,'Tierra del Fuego',7);scalebar(axD,-71.6,-58.15,100)
handles=[Line2D([0],[0],color=BLUE,lw=2,label='Training pool'),Line2D([0],[0],color=ORANGE,lw=2,label='Held-out survey')]
cax=fig.add_axes([.14,.065,.46,.014])
cb=fig.colorbar(plt.cm.ScalarMappable(norm=BATHY_NORM,cmap=BATHY_CMAP),cax=cax,orientation='horizontal',ticks=[0,1000,2000,3000,4000,5000])
cb.ax.tick_params(labelsize=8,length=2);cb.set_label('Depth (m)',fontsize=9,labelpad=3);cb.outline.set_linewidth(.5)
fig.legend(handles=handles,loc='center',bbox_to_anchor=(.79,.069),frameon=False,ncol=1,fontsize=9,handlelength=3)
for ext in ['pdf','svg','png']:
    fig.savefig(FIG_DIR/f'Fig_1.{ext}',bbox_inches='tight',dpi=300 if ext!='png' else 210)
print(f'Saved {FIG_DIR}/Fig_1.{{pdf,svg,png}}'+(f' and navigation tables in {MAP_DIR}/' if IN.is_dir() else ''))
