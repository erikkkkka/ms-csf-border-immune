#!/usr/bin/env python3
"""Plot the IFNGR1 PCA–IVW estimates."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker

ROOT = Path(__file__).resolve().parents[1]
with (ROOT / 'results/mr_pca_ivw.csv').open() as f:
    rows = sorted(csv.DictReader(f), key=lambda row: row['cell_type'])
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],'mathtext.fontset':'dejavusans','font.size':7,'axes.labelsize':7,'xtick.labelsize':7,'axes.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white','axes.facecolor':'white','text.color':'#2B2D2F','axes.labelcolor':'#2B2D2F','xtick.color':'#2B2D2F'})
MM=1/25.4;W,H=140,55
fig=plt.figure(figsize=(W*MM,H*MM));texts=[]
def text(x,y,s,**kw):
 t=fig.text(x/W,y/H,s,**kw);texts.append(t);return t
a=fig.add_axes([33/W,14/H,51/W,28/H]);a.set_xscale('log');a.set_xlim(.75,2.4);a.set_ylim(0,3)
a.set_yticks([]);a.xaxis.set_major_locator(ticker.FixedLocator([.8,1,1.5,2]));a.xaxis.set_major_formatter(ticker.FixedFormatter(['0.8','1.0','1.5','2.0']));a.xaxis.set_minor_locator(ticker.NullLocator())
for key in ['left','right','top']:a.spines[key].set_visible(False)
a.spines['bottom'].set_color('#7C858D');a.tick_params(axis='x',length=3,width=.6)
a.axvline(1,color='#8A939B',lw=.8,ls=(0,(3,3)),zorder=0)
a.set_xlabel('Odds ratio (log scale)',labelpad=6)
def fy(y):return 14+y/3*28
for x,label,ha in [(5,'Monocytes','left'),(104,'OR (95% CI)','center'),(131,r'Nominal $P$','center')]:text(x,48,label,fontsize=7.5,fontweight='bold',ha=ha,va='center')
colors={'CD14':'#4E79A7','CD16':'#899299'}
for r,y in zip(rows,[2.3,1.0]):
 color=colors[r['cell_type'].removesuffix('_Mono')];lo,hi=float(r['CI95_lower']),float(r['CI95_upper']);o=float(r['odds_ratio'])
 a.errorbar(o,y,xerr=[[o-lo],[hi-o]],fmt='o',markersize=5.5,color=color,ecolor=color,elinewidth=1.25,capsize=2.5,capthick=.8,markeredgecolor='white',markeredgewidth=.45,zorder=3)
 text(5,fy(y),r['cell_type'].removesuffix('_Mono'),va='center',fontsize=8,fontweight='bold' if r['cell_type'].removesuffix('_Mono')=='CD14' else 'normal',color=color)
 text(104,fy(y),f"{o:.2f} ({lo:.2f}–{hi:.2f})",ha='center',va='center',fontsize=7.5)
 text(131,fy(y),f"{float(r['nominal_p']):.3f}",ha='center',va='center',fontsize=7.5)
fig.savefig(ROOT / 'results/Figure S10.pdf')
plt.close(fig)
