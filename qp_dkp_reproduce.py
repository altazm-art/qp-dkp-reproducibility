#!/usr/bin/env python3
"""Reproduce Tables I--V and canonical-sum checks for the qp-DKP oscillator manuscript.

Run: python qp_dkp_reproduce.py --output-dir results
Add --full for figure data, fixed-Sigma scans and high-precision checks.
No network, external datasets or plotting software is required.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import platform
import sys
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import brentq
from scipy.special import exp1, gammaincc, gamma
import mpmath as mp

PAIRS = ((.1,.1),(.1,.2),(.2,.1),(.2,.3))


def log_a(n, mu, nu):
    """log([n+1]+[n]-1), including exact undeformed and ground limits."""
    n = np.asarray(n, dtype=float)
    if mu == 0 and nu == 0:
        with np.errstate(divide='ignore'):
            return np.log(2*n)
    sigma = mu+nu
    if mu < 0 or nu <= 0:
        raise ValueError('Use mu>=0, nu>0, or the undeformed origin.')
    def log_number(k):
        with np.errstate(divide='ignore'):
            return (k-1)*nu+np.log(-np.expm1(-k*sigma))-math.log(-math.expm1(-sigma))
    ls = np.logaddexp(log_number(n+1),log_number(n))
    with np.errstate(divide='ignore',invalid='ignore'):
        answer = ls+np.log(-np.expm1(-ls))
    return np.where(n==0,-np.inf,answer)


def excitation(mu,nu,r,nmax):
    """K=(E-1)/r; r=0 returns the matched NR level A_n/2."""
    la = log_a(np.arange(nmax+1),mu,nu)
    if r == 0:
        return np.exp(la-math.log(2))
    le = .5*np.logaddexp(0,math.log(r)+la)
    # Rationalized square root, evaluated in logs to avoid large A_n.
    return np.exp(la-np.logaddexp(le,0))


def log_energy(n,mu,nu,r):
    return float(.5*np.logaddexp(0,math.log(r)+log_a(n,mu,nu)))


def adaptive_cutoff(mu,nu,r,tau,target=62):
    """Smallest inclusive N with E_N/tau >= target (not a tail bound)."""
    if tau<=0 or r<=0:
        raise ValueError('tau and r must be positive')
    if mu==0 and nu==0:
        return max(0,math.ceil(((target*tau)**2-1)/(2*r)))
    goal=math.log(target)+math.log(tau)
    lo,hi=-1,1
    while log_energy(hi,mu,nu,r)<goal:
        hi*=2
    while hi-lo>1:
        mid=(lo+hi)//2
        if log_energy(mid,mu,nu,r)>=goal:hi=mid
        else:lo=mid
    return hi


def canonical(levels,x):
    """Moments of levels/x; fsum and centered variance improve conditioning."""
    y=np.asarray(levels)/x
    w=np.exp(-y)
    z=math.fsum(w)
    p=w/z
    mean=math.fsum(p*y)
    variance=math.fsum(p*(y-mean)**2)
    third=math.fsum(p*(y-mean)**3)
    return dict(logW=math.log(z),mean_scaled=mean,C=variance,
                S=math.log(z)+mean,stationarity=third-2*variance)


def thermal(mu,nu,r,tau,nmax=None):
    if nmax is None:nmax=adaptive_cutoff(mu,nu,r,tau)
    k=excitation(mu,nu,r,nmax)
    v=canonical(k,tau/r)
    lnz=-1/tau+v['logW']
    return dict(Z=math.exp(lnz),F=1-tau*v['logW'],
                U=1+tau*v['mean_scaled'],S=v['S'],C=v['C'],
                logZ=lnz,nmax=nmax)


def peak(mu,nu,r,nmax,low=.1,high=100):
    levels=excitation(mu,nu,r,nmax)
    xs=np.geomspace(low,high,451)
    vals=[canonical(levels,float(x)) for x in xs]
    roots=[]
    for i in range(len(xs)-1):
        if vals[i]['stationarity']>0>=vals[i+1]['stationarity']:
            x=brentq(lambda t:canonical(levels,t)['stationarity'],xs[i],xs[i+1],xtol=2e-13)
            roots.append((x,canonical(levels,x)['C']))
    if not roots:raise ArithmeticError('No interior heat-capacity maximum')
    best=max(roots,key=lambda z:z[1])
    if best[1]<=max(vals[0]['C'],vals[-1]['C']):
        raise ArithmeticError('Search endpoint exceeds selected peak')
    return best


def first_order(mu,nu,x,nmax=200):
    k=excitation(mu,nu,0,nmax)
    w=np.exp(-k/x);w/=math.fsum(w)
    m1,m2,m3,m4=[math.fsum(w*k**j) for j in range(1,5)]
    return (m4-2*m1*m3+2*m1*m1*m2-m2*m2)/(2*x**3)-(m3-m1*m2)/x**2


def printed_approximation(eta,tau,target=62):
    """Differentiate the literal printed Z_app consistently for all observables.

    This reproduces the comparison expression, not a corrected expansion.
    It is not a globally valid probability distribution or Hamiltonian.
    """
    nmax=math.ceil(((1+target*tau)**2-1)/2)
    y=np.sqrt(2*np.arange(nmax+1,dtype=float)+1)
    b=y/tau
    c=eta**2*y**5/(48*tau)
    w=np.exp(-b)
    z=math.fsum(w*(1-c))
    if z<=0:raise ArithmeticError('Printed approximation has nonpositive Z')
    m1=math.fsum(w*(b+c-b*c))/z
    m2=math.fsum(w*(b*b+2*b*c-b*b*c))/z
    lz=math.log(z)
    return dict(Z=z,F=-tau*lz,U=tau*m1,S=lz+m1,C=m2-m1*m1,nmax=nmax)


def tail_bounds(mu,nu,r,tau,nmax,v):
    """Bounds on omitted raw moments of E/tau, then normalized Z and C.

    Geometric bound is established here only for r=1 deformed levels.
    Undeformed levels use their monotone integral with dn=tau^2*x/r dx.
    """
    x=math.exp(log_energy(nmax+1,mu,nu,r)-math.log(tau))
    if x<2:raise ValueError('Moment-tail bound requires x_* >= 2')
    b=[]
    for j in range(3):
        if mu==0 and nu==0:
            integral=tau*tau/r*gamma(j+2)*gammaincc(j+2,x)
        elif r==1:
            integral=2/nu*(exp1(x) if j==0 else gamma(j)*gammaincc(j,x))
        else:raise ValueError('No deformed geometric bound claimed for r != 1')
        b.append(x**j*math.exp(-x)+integral)
    z=v['Z'];m1=v['U']/tau;m2=v['C']+m1*m1
    e1=(b[1]+abs(m1)*b[0])/z
    e2=(b[2]+abs(m2)*b[0])/z
    return dict(relative_Z_bound=b[0]/z,absolute_C_bound=e2+2*abs(m1)*e1+e1*e1,
                x_first_omitted=x,raw_scaled_moment_bounds=list(map(float,b)))


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def tables(out):
    checks=[];rows=[]
    expected=[[(.477,1.260),(.744,1.370),(1.129,1.468)],
              [(.436,1.006),(.703,1.091),(1.092,1.167)],
              [(.412,1.387),(.655,1.505),(1.007,1.612)],
              [(.397,.907),(.657,.980),(1.037,1.046)]]
    for j,(mu,nu) in enumerate(PAIRS):
        for l,r in enumerate((.5,1,2)):
            x,c=peak(mu,nu,r,150,.1/r,10/r);t=r*x
            assert abs(t-expected[j][l][0])<.000501 and abs(c-expected[j][l][1])<.000501
            rows.append(dict(mu=mu,nu=nu,r=r,tau_peak=t,C_peak=c,nmax=150))
    write_csv(out/'table_I.csv',rows);checks.append('Table I: 12 peaks match printed precision')
    print(checks[-1],flush=True)
    rows=[]
    expected=[(1.351,.13505,1.028),(1.269,.01269,.898),(1.091,.10906,.835),(1.014,.01014,.751),
              (1.138,.11384,1.143),(1.172,.01172,1.010),(.949,.09488,.764),(.909,.00909,.698)]
    for mu,nu in PAIRS:
        for r in (.1,.01):
            x,c=peak(mu,nu,r,200,.2,10);t=r*x;e=expected[len(rows)]
            assert abs(x-e[0])<.000501 and abs(t-e[1])<.00000501 and abs(c-e[2])<.000501
            rows.append(dict(mu=mu,nu=nu,r=r,x_peak=x,tau_peak=t,C_peak=c,nmax=200))
    write_csv(out/'table_II.csv',rows);checks.append('Table II: 8 peaks match printed precision')
    print(checks[-1],flush=True)
    rows=[];cutoff_max=0
    expected=[(1.2265,.8796,1.2687,.8976,.8979),(.9948,.7400,1.0142,.7510,.7511),
              (1.1595,.9910,1.1722,1.0103,1.0108),(.8984,.6896,.9089,.6982,.6983)]
    for j,(mu,nu) in enumerate(PAIRS):
        x0,c0=peak(mu,nu,0,200);x,c=peak(mu,nu,.01,200);d=first_order(mu,nu,x0);pred=c0+.01*d
        for a,e in zip((x0,c0,x,c,pred),expected[j]):assert abs(a-e)<.0000501,(a,e)
        cx=canonical(excitation(mu,nu,.01,300),x)['C'];cutoff_max=max(cutoff_max,abs(cx-c))
        rows.append(dict(mu=mu,nu=nu,x_NR=x0,C_NR=c0,x_DKP=x,C_DKP=c,C_first_order=pred,
                         D=d,exact_increase_percent=100*(c/c0-1),nmax=200))
    assert cutoff_max<2e-12
    write_csv(out/'table_III.csv',rows);checks.append('Table III: NR, DKP and first-order entries match printed precision')
    print(checks[-1],flush=True)
    rows=[]
    for eta in (0.,9e-5):
        for tau in (1.,10.,20.):
            nmax=math.ceil(((1+62*tau)**2-1)/2)
            v=thermal(eta,eta,1,tau,nmax)
            rows.append(dict(eta=eta,tau=tau,method='unexpanded',**{k:v[k] for k in ('Z','F','U','S','C','nmax')}))
            if eta:
                a=printed_approximation(eta,tau)
                rows.append(dict(eta=eta,tau=tau,method='printed_approximation',**a))
    ref=[(.94805,.05335,2.13580,2.08245,1.81068),(99.99180,-46.05088,20.00242,6.60533,1.99953),
         (399.99580,-119.82908,40.00063,7.99149,1.99994),(.94805,.05335,2.13580,2.08245,1.81068),
         (.94805,.05335,2.13580,2.08245,1.81068),(99.87490,-46.03918,19.95718,6.59964,1.97800),
         (99.87030,-46.03872,19.95376,6.59925,1.97518),(394.26524,-119.54048,39.05377,7.92971,1.81542),
         (392.21980,-119.43645,38.41459,7.89255,1.59714)]
    for row,e in zip(rows,ref):
        for key,t in zip(('Z','F','U','S','C'),e):assert abs(row[key]-t)<.0000051,(key,row[key],t)
    write_csv(out/'table_IV.csv',rows);checks.append('Table IV: all 45 thermal values match printed precision')
    print(checks[-1],flush=True)
    cutoff_rows=[]
    for row in rows:
        eta,tau=row['eta'],row['tau']
        n68=math.ceil(((1+68*tau)**2-1)/2)
        v68=(thermal(eta,eta,1,tau,n68) if row['method']=='unexpanded'
             else printed_approximation(eta,tau,target=68))
        cutoff_rows.append(dict(eta=eta,tau=tau,method=row['method'],
                                nmax_62=row['nmax'],nmax_68=n68,
                                C_62=row['C'],C_68=v68['C'],
                                absolute_change=abs(v68['C']-row['C'])))
    cutoff_iv=max(row['absolute_change'] for row in cutoff_rows)
    assert cutoff_iv<4.1e-15,cutoff_iv
    write_csv(out/'table_IV_cutoff_check.csv',cutoff_rows)
    rows=[];bounds=[]
    ref=[(.937451,1.025541,.992494,1.003154),(.945663,1.016990,.993480,1.002163),
         (.943921,1.018792,.993270,1.002373),(.954914,1.007525,.994590,1.001049)]
    for j,(mu,nu) in enumerate(PAIRS):
        for k,tau in enumerate((1e12,1e100)):
            v=thermal(mu,nu,1,tau);rz=nu*v['Z']/(2*math.log(tau));rc=v['C']*math.log(tau)
            assert abs(rz-ref[j][2*k])<.00000051 and abs(rc-ref[j][2*k+1])<.00000051
            rows.append(dict(mu=mu,nu=nu,tau=tau,R_Z=rz,R_C=rc,nmax=v['nmax']))
            bounds.append(dict(mu=mu,nu=nu,tau=tau,**tail_bounds(mu,nu,1,tau,v['nmax'],v)))
    write_csv(out/'table_V.csv',rows);checks.append('Table V: all 16 ratios match printed precision')
    print(checks[-1],flush=True)
    return dict(table_checks=checks,peak_cutoff_200_to_300_abs_change=cutoff_max,
                table_IV_cutoff_62_to_68_max_absolute_C_change=cutoff_iv,
                tail_bounds_table_V=bounds)


def mp_levels(mu,nu,r,nmax):
    mu,nu,r=map(lambda x:mp.mpf(str(x)),(mu,nu,r));q,t=mp.exp(nu),mp.exp(-mu)
    b=[mp.mpf(0)]
    for n in range(nmax+1):b.append(q*b[-1]+t**n)
    a=[b[n+1]+b[n]-1 for n in range(nmax+1)]
    return [v/(mp.sqrt(1+r*v)+1) if r else v/2 for v in a]


def mp_obs(k,x,rest_tau=None):
    y=[a/x for a in k];w=[mp.exp(-a) for a in y];z=mp.fsum(w)
    mean=mp.fsum(a*b for a,b in zip(y,w))/z
    c=mp.fsum(b*(a-mean)**2 for a,b in zip(y,w))/z
    if rest_tau is None:return c
    return dict(Z=mp.exp(-1/rest_tau)*z,U=1+rest_tau*mean,S=mp.log(z)+mean,C=c)


def precision_checks(out):
    result={}
    with mp.workdps(60):
        levels=mp_levels(.2,.1,1,150);worst={k:mp.mpf(0) for k in ('Z','U','S','C')}
        for j in range(199):
            t=mp.mpf('.1')+j*mp.mpf('.05')
            a=mp_obs(levels[:101],t,t);b=mp_obs(levels,t,t)
            for key in worst:worst[key]=max(worst[key],abs((b[key]-a[key])/b[key]))
        result['baseline_60_digits_100_to_150']={k:str(v) for k,v in worst.items()}
    print('60-digit baseline cutoff grid completed',flush=True)
    with mp.workdps(110):
        k=mp_levels(.2,.1,.01,200)
        result['small_r_110_digits_C_150_to_200']=str(abs(mp_obs(k[:151],mp.mpf(1000))-mp_obs(k,mp.mpf(1000))))
    print('110-digit small-r cutoff check completed',flush=True)
    rows=[]
    with mp.workdps(70):
        for mu,nu in PAIRS:
            x0,c0=peak(mu,nu,0,200);x=mp.mpf(str(x0));k=mp_levels(mu,nu,0,300)
            w=[mp.exp(-v/x) for v in k];z=mp.fsum(w)
            m1,m2,m3,m4=[mp.fsum(a*v**j for a,v in zip(w,k))/z for j in range(1,5)]
            d=(m4-2*m1*m3+2*m1*m1*m2-m2*m2)/(2*x**3)-(m3-m1*m2)/x**2
            assert abs(float(d)-first_order(mu,nu,x0))<2e-12
            h=mp.mpf('1e-6');c=mp_obs(k,x)
            ch=mp_obs(mp_levels(mu,nu,h,300),x);ch2=mp_obs(mp_levels(mu,nu,h/2,300),x)
            derivative=2*(ch2-c)/(h/2)-(ch-c)/h
            assert abs(derivative-d)<mp.mpf('1e-9')
            for r in (0,.01):
                xp,cp=peak(mu,nu,r,200);c70=mp_obs(mp_levels(mu,nu,r,300),mp.mpf(str(xp)))
                assert abs(float(c70)-cp)<2e-12
            rows.append(dict(mu=mu,nu=nu,D=str(d),derivative_error=str(abs(derivative-d))))
    result['first_order_70_digits']=rows
    fixed=[]
    with mp.workdps(70):
        def stationarity(k,x):
            y=[v/x for v in k];w=[mp.exp(-v) for v in y];z=mp.fsum(w)
            mean=mp.fsum(a*b for a,b in zip(y,w))/z
            var=mp.fsum(b*(a-mean)**2 for a,b in zip(y,w))/z
            return mp.fsum(b*(a-mean)**3 for a,b in zip(y,w))/z-2*var
        for sigma in (.30,.50):
            for r in (1.,.1):
                for delta in (-.1,0.,.1):
                    mu,nu=(sigma-delta)/2,(sigma+delta)/2
                    xp,cp=peak(mu,nu,r,200);x=mp.mpf(str(xp));k=mp_levels(mu,nu,r,300)
                    c=mp_obs(k,x);change=abs(c-mp_obs(k[:201],x))
                    assert abs(float(c)-cp)<2e-12
                    h=mp.mpf('2e-9')
                    assert stationarity(k,x-h)>0>stationarity(k,x+h)
                    fixed.append(dict(Sigma=sigma,Delta=delta,r=r,x_peak=xp,C_70_digits=str(c),
                                      cutoff_200_to_300_change=str(change),root_bracket_halfwidth='2e-9'))
    result['fixed_sigma_70_digits']=fixed
    print('70-digit NR/DKP and first-order checks completed',flush=True)
    rows=[]
    with mp.workdps(80):
        for mu,nu in PAIRS:
            for st in ('1e12','1e100'):
                t=mp.mpf(st);n=adaptive_cutoff(mu,nu,1,float(t));k=mp_levels(mu,nu,1,n+50)
                a=mp_obs(k[:n+1],t,t);b=mp_obs(k,t,t)
                change=max(abs((a[key]-b[key])/b[key]) for key in ('Z','U','S','C'))
                rows.append(dict(mu=mu,nu=nu,tau=st,nmax=n,max_relative_change=str(change)))
    result['extended_80_digits_n_plus_50']=rows
    (out/'precision_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print('80-digit extended-temperature cutoff checks completed',flush=True)
    return result


def figure_data(out):
    rows=[];bounds=[]
    for mu,nu in (*PAIRS,(0.,0.)):
        for tau in np.linspace(.1,10,199):
            n=100 if nu else math.ceil(max(100,((-tau*math.log(1e-14))**2-1)/2))
            v=thermal(mu,nu,1,float(tau),n)
            rows.append(dict(mu=mu,nu=nu,tau=float(tau),**{k:v[k] for k in ('Z','F','U','S','C','nmax')}))
            if not nu:bounds.append(tail_bounds(mu,nu,1,float(tau),n,v))
    write_csv(out/'baseline_observables.csv',rows)
    rows=[];adaptive_bounds=[]
    for mu,nu in (*PAIRS,(0.,0.)):
        temperatures=(np.geomspace(.1,10,199) if nu==0 else
                      np.r_[np.geomspace(.1,10,199),np.geomspace(10,1e12,241),np.geomspace(1e2,1e100,197)])
        for tau in temperatures:
            v=thermal(mu,nu,1,float(tau));bound=tail_bounds(mu,nu,1,float(tau),v['nmax'],v)
            adaptive_bounds.append(bound)
            rows.append(dict(mu=mu,nu=nu,tau=float(tau),**{k:v[k] for k in ('Z','F','U','S','C','nmax')}))
    write_csv(out/'adaptive_thermal_observables.csv',rows)
    rows=[]
    for sigma in (.30,.50):
        for r in (1.,.1):
            for delta in np.linspace(-.1,.1,101):
                mu,nu=(sigma-delta)/2,(sigma+delta)/2;x,c=peak(mu,nu,r,200)
                c300=canonical(excitation(mu,nu,r,300),x)['C']
                assert abs(c300-c)<2e-12
                rows.append(dict(Sigma=sigma,Delta=float(delta),mu=mu,nu=nu,r=r,x_peak=x,tau_peak=r*x,C_peak=c))
    write_csv(out/'fixed_sigma_peaks.csv',rows)
    print('Baseline, extended-temperature and fixed-Sigma data completed',flush=True)
    return dict(baseline_undeformed_max_relative_Z_bound=max(b['relative_Z_bound'] for b in bounds),
                baseline_undeformed_max_absolute_C_bound=max(b['absolute_C_bound'] for b in bounds),
                adaptive_all_max_relative_Z_bound=max(b['relative_Z_bound'] for b in adaptive_bounds),
                adaptive_all_max_absolute_C_bound=max(b['absolute_C_bound'] for b in adaptive_bounds))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('results'))
    parser.add_argument('--full',action='store_true',help='Also generate figure data and precision checks')
    args=parser.parse_args(argv);out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    summary=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                 mpmath=mp.__version__,float_mantissa_bits=sys.float_info.mant_dig,
                 moment_method='math.fsum, shifted/scaled energies, centered variance',
                 adaptive_terminal_E_over_tau=62,full=args.full)
    summary.update(tables(out))
    if args.full:
        summary['figure_data_bounds']=figure_data(out)
        precision_checks(out)
    (out/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Reproduction completed:',out.resolve(),flush=True)


if __name__=='__main__':
    # %run works normally; direct execution in a notebook ignores kernel -f.
    main([] if Path(sys.argv[0]).name in ('colab_kernel_launcher.py','ipykernel_launcher.py') else None)
