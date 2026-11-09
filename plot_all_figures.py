#!/usr/bin/env python3
"""Regenerate all 13 manuscript figures using the common checked kernel.

Run reproduce_all.py first, or supply its generated data directory here.
The plot data are computed from the spectrum, not digitized from artwork.
Legacy Drive scripts were used to recover plotting ranges and conventions.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogFormatterMathtext
import qp_dkp_reproduce as q
import plot_baseline_figures as first_four

COLORS=('#1565b8','#d57116','#24834a','#9653a6')
FIGURES=[
 ('Fig1_spectrum_vector.pdf','figure1_spectral_data.csv','positive spectrum; n=0..30; r=1'),
 ('Fig2_delta_zero.pdf','figure2_contour.csv;figure2_slice.csv','n=10; r=1; Sigma=0.30'),
 ('Fig3_partition_vector.pdf','baseline_observables.csv','Z; r=1; baseline cutoffs'),
 ('Fig4_free_energy_vector.pdf','baseline_observables.csv','F; r=1; baseline cutoffs'),
 ('Fig5_internal_energy_vector.pdf','baseline_observables.csv','U; r=1; baseline cutoffs'),
 ('Fig_entropy_finite_temperature.pdf','adaptive_thermal_observables.csv','S; r=1; E_N/tau>=62'),
 ('Fig_heat_capacity_finite_temperature.pdf','adaptive_thermal_observables.csv','C; r=1; E_N/tau>=62'),
 ('Fig_fixed_sigma_thermal_peaks.pdf','fixed_sigma_peaks.csv','Sigma=.30,.50; r=1,.1; N=200'),
 ('Fig8_small_r_observables.pdf','small_r_thermodynamics.csv','r=1,.1,.01; N=200; shifted observables'),
 ('Fig9_small_r_heat_capacity.pdf','small_r_thermodynamics.csv','all four pairs; r=1,.1,.01; N=200'),
 ('Fig_NR_DKP_comparison.pdf','nr_dkp_thermodynamics.csv','NR and r=.001,.01,.1; N=200'),
 ('Fig_extended_thermal_checks.pdf','adaptive_thermal_observables.csv','r=1; tau=10..1e12; continuum guides'),
 ('Fig10_high_tau_ratios.pdf','adaptive_thermal_observables.csv','r=1; tau=1e2..1e100; continuum target')]


def read_rows(path):
    with path.open() as f:return [{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]


def save(fig,out,name):
    fig.savefig(out/name,bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)


def subset(rows,pair,low=None,high=None):
    selected={row['tau']:row for row in rows
              if (row['mu'],row['nu'])==pair and
              (low is None or row['tau']>=low*(1-1e-13)) and
              (high is None or row['tau']<=high*(1+1e-13))}
    return [selected[x] for x in sorted(selected)]


def contour(out,data):
    energy=lambda mu,nu:math.exp(q.log_energy(10,mu,nu,1))
    grid=np.linspace(.05,.30,51)
    z=np.array([[energy(float(mu),float(nu)) for mu in grid] for nu in grid])
    rows=[dict(mu=float(mu),nu=float(nu),n=10,r=1,E=float(z[j,i]))
          for j,nu in enumerate(grid) for i,mu in enumerate(grid)]
    q.write_csv(data/'figure2_contour.csv',rows)
    ds=np.linspace(-.1,.1,201)
    ss=[dict(Sigma=.30,Delta=float(d),mu=(.30-d)/2,nu=(.30+d)/2,
             E=energy((.30-d)/2,(.30+d)/2)) for d in ds]
    q.write_csv(data/'figure2_slice.csv',ss)
    ratio=lambda n:math.sinh(n*.30/2)/math.sinh(.30/2)
    slope=(10*ratio(11)+9*ratio(10))/(4*math.sqrt(ratio(11)+ratio(10)))
    h=1e-5;fd=(energy((.30-h)/2,(.30+h)/2)-energy((.30+h)/2,(.30-h)/2))/(2*h)
    assert math.isclose(slope,fd,rel_tol=2e-8)
    fig,(a,b)=plt.subplots(1,2,figsize=(8,3.45),gridspec_kw={'width_ratios':[1.18,1]},layout='constrained')
    filled=a.contourf(grid,grid,z,levels=np.linspace(3,11.5,18),cmap='viridis',extend='both')
    fig.colorbar(filled,ax=a,shrink=.8,pad=.03)
    a.plot([.2,.1],[.1,.2],color='white',lw=2.1,ls='--',zorder=4)
    marks=[next(v for v in ss if abs(v['Delta']-d)<1e-12) for d in [-.05,0,.05]]
    a.scatter([v['mu'] for v in marks],[v['nu'] for v in marks],s=37,facecolor='white',edgecolor='#9a174a',lw=1.5,zorder=5)
    a.set(xlabel=r'$\mu$',ylabel=r'$\nu$',title=r'(a) Parameter plane and $\Sigma=0.30$ slice',xlim=(.05,.30),ylim=(.05,.30))
    a.set_aspect('equal',adjustable='box');a.set_xticks(np.arange(.05,.301,.05));a.set_yticks(np.arange(.05,.301,.05))
    b.plot(ds,[v['E'] for v in ss],color='#205d7a',lw=2.3)
    b.axvline(0,color='#777777',ls='--',lw=1)
    b.scatter([v['Delta'] for v in marks],[v['E'] for v in marks],s=44,facecolor='white',edgecolor='#9a174a',lw=1.7,zorder=5)
    b.set(xlabel=r'$\Delta=\nu-\mu$ at fixed $\Sigma=0.30$',ylabel=r'$\epsilon_{10}^{(+)}/m$',title=r'(b) Neighborhood of $\Delta=0$',xlim=(-.1,.1))
    b.set_xticks([-.1,-.05,0,.05,.1]);b.grid(alpha=.2,lw=.6)
    save(fig,out,'Fig2_delta_zero.pdf')
    return dict(contour_points=len(rows),slice_points=len(ss),analytic_slice_slope=slope,finite_difference_slope=fd)


def adaptive_plots(out,data):
    rows=read_rows(data/'adaptive_thermal_observables.csv')
    for key,ylabel,name in [('S',r'$\mathcal{S}=S/k_{\rm B}$','Fig_entropy_finite_temperature.pdf'),
                             ('C',r'$\mathcal{C}_V=C_V/k_{\rm B}$','Fig_heat_capacity_finite_temperature.pdf')]:
        fig,a=plt.subplots(figsize=(6.8,3.6),layout='constrained')
        for pair,c in zip(q.PAIRS,COLORS):
            v=subset(rows,pair,.1,10);a.plot([t['tau'] for t in v],[t[key] for t in v],color=c,lw=2,label=rf'$({pair[0]:g},{pair[1]:g})$')
        v=subset(rows,(0.,0.),.1,10)
        a.plot([t['tau'] for t in v],[t[key] for t in v],color='#34383e',ls='--',lw=1.8,label='Undeformed')
        a.set_xscale('log');a.set(xlim=(.1,10),xlabel=r'$\tau=k_{\rm B}T/m$',ylabel=ylabel);a.grid(alpha=.2)
        a.legend(title=r'$(\mu,\nu)$',loc='upper left',framealpha=.95,edgecolor='none',fontsize=9)
        save(fig,out,name)
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.7),layout='constrained')
    for pair,c in zip(q.PAIRS,COLORS):
        v=subset(rows,pair,10,1e12);t=np.array([row['tau'] for row in v])
        nu=pair[1];A=math.sqrt((math.exp(nu)+1)/(math.exp(nu)-math.exp(-pair[0])))
        L=np.log(t/A);mask=t>=1e4
        for a,key,guide in [(axes[0],'S',np.log((2/nu)*L)),(axes[1],'C',1/L-1/L**2)]:
            a.plot(t,[w[key] for w in v],color=c,lw=1.8)
            a.plot(t[mask],guide[mask],color=c,ls='--',lw=1.3)
    for a in axes:
        a.set_xscale('log');a.set_xlim(10,1e12);a.set_xticks([10,1e4,1e8,1e12]);a.grid(alpha=.2);a.set_xlabel(r'$\tau=k_{\rm B}T/m$')
    axes[0].set(title='(a) Entropy',ylabel=r'$\mathcal{S}$')
    axes[1].set(title='(b) Heat capacity',ylabel=r'$\mathcal{C}_V$',yscale='log');axes[1].axhline(2,color='#34383e',ls=':',lw=1.2)
    axes[0].legend(handles=[Line2D([],[],color=c,lw=2,label=rf'$({mu:g},{nu:g})$') for (mu,nu),c in zip(q.PAIRS,COLORS)],loc='upper left',title=r'$(\mu,\nu)$',frameon=False,fontsize=8.5)
    axes[1].legend(handles=[Line2D([],[],color='#555555',lw=1.8,label='Direct sum'),Line2D([],[],color='#555555',ls='--',label='Continuum guide'),Line2D([],[],color='#34383e',ls=':',label='Undeformed limit')],loc='upper right',frameon=False,fontsize=8.5)
    save(fig,out,'Fig_extended_thermal_checks.pdf')
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.5))
    for pair,c in zip(q.PAIRS,COLORS):
        v=subset(rows,pair,1e2,1e100);t=np.array([w['tau'] for w in v]);sel=np.log10(t)
        # Use only the 197-point 1e2..1e100 diagnostic grid and exact endpoints.
        mask=np.isclose((sel-2)*2,np.round((sel-2)*2),atol=1e-10)
        v=[w for w,m in zip(v,mask) if m];t=np.array([w['tau'] for w in v])
        axes[0].plot(t,[pair[1]*w['Z']/(2*math.log(w['tau'])) for w in v],color=c,lw=1.9)
        axes[1].plot(t,[w['C']*math.log(w['tau']) for w in v],color=c,lw=1.9)
    for a in axes:
        a.set_xscale('log');a.set(xlim=(1e2,1e100),xlabel=r'$\tau=k_{\rm B}T/m$');a.xaxis.set_major_locator(FixedLocator([1e2,1e25,1e50,1e75,1e100]));a.xaxis.set_major_formatter(LogFormatterMathtext());a.axhline(1,color='#222222',ls='--',lw=1.2);a.grid(color='#d9dee5',lw=.65)
    axes[0].set(title='(a) Partition-function normalization',ylabel=r'$R_Z=\nu Z/(2\ln\tau)$',ylim=(.60,1.015))
    axes[1].set(title='(b) Heat-capacity normalization',ylabel=r'$R_C=\mathcal{C}_V\ln\tau$',ylim=(.965,1.085))
    axes[0].text(.97,.08,r'$R_Z\to1$',ha='right',transform=axes[0].transAxes)
    axes[1].text(.97,.08,'Continuum guide: 1',ha='right',transform=axes[1].transAxes,fontsize=10)
    fig.legend(handles=[Line2D([],[],color=c,lw=2,label=rf'$(\mu,\nu)=({mu:.1f},{nu:.1f})$') for (mu,nu),c in zip(q.PAIRS,COLORS)],loc='upper center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False)
    fig.subplots_adjust(left=.075,right=.98,bottom=.14,top=.79,wspace=.30)
    save(fig,out,'Fig10_high_tau_ratios.pdf')


def fixed_sigma(out,data):
    rows=read_rows(data/'fixed_sigma_peaks.csv')
    fig,axes=plt.subplots(2,2,figsize=(7.6,5.3),layout='constrained',sharex=True)
    for row,sigma in enumerate([.30,.50]):
        dc=-2*math.log(math.cosh(sigma/2))
        for r,c in [(1.,COLORS[2]),(.1,COLORS[1])]:
            v=[v for v in rows if v['Sigma']==sigma and v['r']==r]
            assert len(v)==101
            for col,key in enumerate(['x_peak','C_peak']):
                a=axes[row,col];a.plot([t['Delta'] for t in v],[t[key] for t in v],color=c,lw=2,label=rf'$r={r:g}$')
                a.scatter([t['Delta'] for t in v[::50]],[t[key] for t in v[::50]],s=22,color=c,zorder=4)
        for col in [0,1]:
            a=axes[row,col];a.axvspan(-.105,dc,color='#e7f2ea',zorder=0);a.axvline(0,color='#777777',ls='--',lw=1);a.axvline(dc,color='#658d70',ls=':',lw=1.4)
            a.set(title=rf'({"abcd"[row*2+col]}) $\Sigma={sigma:.2f}$',xlim=(-.105,.105),ylabel=r'$x_{\max}=\tau_{\max}/r$' if col==0 else r'$\mathcal{C}_{V,\max}$')
            a.set_xticks([-.1,-.05,0,.05,.1]);a.grid(alpha=.18)
    for a in axes[1]:a.set_xlabel(r'$\Delta=\nu-\mu$')
    axes[0,0].legend(frameon=False,loc='upper left',fontsize=9)
    save(fig,out,'Fig_fixed_sigma_thermal_peaks.pdf')


def small_r(out,data):
    rows=[]
    for mu,nu in q.PAIRS:
        for r in [1.,.1,.01]:
            k=q.excitation(mu,nu,r,200)
            for x in np.geomspace(.2,10 if r==1 else 30,241):
                v=q.canonical(k,float(x))
                rows.append(dict(mu=mu,nu=nu,r=r,x=float(x),tau=r*x,
                                 Z_shifted=math.exp(v['logW']),U_excitation_over_r=x*v['mean_scaled'],S=v['S'],C=v['C'],nmax=200))
    q.write_csv(data/'small_r_thermodynamics.csv',rows)
    colors={1.:'#7b3294',.1:'#d95f02',.01:'#167a84'}
    fig,axes=plt.subplots(2,2,figsize=(6.6,5.2),layout='constrained',sharex=True)
    measures=[('Z_shifted','(a) Partition sum',r'$\widetilde{Z}=e^{1/\tau}Z$'),('U_excitation_over_r','(b) Excitation energy',r'$(\mathcal{U}-1)/r$'),('S','(c) Entropy',r'$\mathcal{S}=S/k_{\rm B}$'),('C','(d) Heat capacity',r'$\mathcal{C}_V=C_V/k_{\rm B}$')]
    for r in [1.,.1,.01]:
        for pair in [(.1,.1),(.2,.1)]:
            v=[t for t in rows if (t['mu'],t['nu'])==pair and t['r']==r]
            for a,(key,title,ylabel) in zip(axes.flat,measures):
                a.plot([t['x'] for t in v],[t[key] for t in v],color=colors[r],ls='-' if pair==(.1,.1) else '--',lw=1.8,label=rf'$r={r:g}$')
    for a,(_,title,ylabel) in zip(axes.flat,measures):a.set(title=title,xscale='log',xlim=(.2,30),ylabel=ylabel);a.grid(alpha=.2)
    for a in axes[1]:a.set_xlabel(r'$\tau/r$')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h[::2],l[::2],loc='outside lower center',ncol=3,fontsize=9,frameon=False)
    save(fig,out,'Fig8_small_r_observables.pdf')
    fig,axes=plt.subplots(2,2,figsize=(6.6,5),layout='constrained',sharex=True,sharey=True)
    for a,pair in zip(axes.flat,q.PAIRS):
        for r in [1.,.1,.01]:
            v=[t for t in rows if (t['mu'],t['nu'])==pair and t['r']==r]
            a.plot([t['x'] for t in v],[t['C'] for t in v],color=colors[r],lw=2,label=rf'$r={r:g}$')
        a.set(title=rf'$(\mu,\nu)=({pair[0]:g},{pair[1]:g})$',xscale='log',xlim=(.2,30),ylim=(0,1.7));a.grid(alpha=.2)
    for a in axes[1]:a.set_xlabel(r'$\tau/r$')
    for a in axes[:,0]:a.set_ylabel(r'$\mathcal{C}_V=C_V/k_{\rm B}$')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=3,fontsize=9,frameon=False)
    save(fig,out,'Fig9_small_r_heat_capacity.pdf')
    return len(rows)


def nr_dkp(out,data):
    rows=[];colors={0:'#333333',.001:'#9653a6',.01:'#1565b8',.1:'#d57116'}
    fig,axes=plt.subplots(2,2,figsize=(7.3,5.1),layout='constrained',sharex=True)
    for a,(mu,nu) in zip(axes.flat,q.PAIRS):
        for r in [0,.001,.01,.1]:
            k=q.excitation(mu,nu,r,200);ys=[]
            for x in np.geomspace(.2,3,241):
                v=q.canonical(k,float(x));ys.append(v['C'])
                rows.append(dict(mu=mu,nu=nu,r=r,x=float(x),Z_shifted=math.exp(v['logW']),U_excitation_over_hbaromega=x*v['mean_scaled'],S=v['S'],C=v['C'],nmax=200))
            a.plot(np.geomspace(.2,3,241),ys,color=colors[r],lw=1.9,ls='--' if r==0 else '-',label='Nonrelativistic' if r==0 else rf'$r={r:g}$')
        a.set(title=rf'$(\mu,\nu)=({mu:g},{nu:g})$',xscale='log',xlim=(.2,3),ylabel=r'$\mathcal{C}_V$');a.grid(alpha=.2)
    for a in axes[1]:a.set_xlabel(r'$x=\tau/r=k_{\rm B}T/(\hbar\omega)$')
    axes[0,1].legend(loc='lower right',frameon=False,fontsize=8.5)
    save(fig,out,'Fig_NR_DKP_comparison.pdf');q.write_csv(data/'nr_dkp_thermodynamics.csv',rows)
    return len(rows)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,default=Path('reproduced/data'))
    ap.add_argument('--output-dir',type=Path,default=Path('reproduced/figures'))
    args=ap.parse_args(argv);out,data=args.output_dir,args.data_dir;out.mkdir(parents=True,exist_ok=True);data.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,'axes.titlesize':10,'pdf.fonttype':42,'ps.fonttype':42})
    first_four.main(['--output-dir',str(out),'--data-dir',str(data)])
    plt.rcParams.update({'font.size':10,'axes.labelsize':11,'legend.fontsize':9,'xtick.labelsize':9,'ytick.labelsize':9})
    checks=contour(out,data);adaptive_plots(out,data);fixed_sigma(out,data)
    checks['small_r_rows']=small_r(out,data);checks['nr_dkp_rows']=nr_dkp(out,data)
    mapping=[dict(figure=i,filename=n,data_files=d,convention=c) for i,(n,d,c) in enumerate(FIGURES,1)]
    q.write_csv(out.parent/'figure_map.csv',mapping)
    assert len(list(out.glob('*.pdf')))==13,'Use a dedicated figure output folder'
    checks['figure_count']=13;checks['all_filenames']=[n for n,d,c in FIGURES]
    (out.parent/'figure_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    print('All 13 figures and their numerical data regenerated.',flush=True)
    return checks


if __name__=='__main__':main()
