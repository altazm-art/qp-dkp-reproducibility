#!/usr/bin/env python3
"""Discrete high-temperature check: direct sums and discrete high-temperature corrections.

Run with Python 3 and mpmath; no network or external data are used.
The temperature is represented by log(tau/A), avoiding binary64 overflow.
"""
import argparse
import json
from pathlib import Path
import mpmath as mp


def direct(mu, nu, r, L, target=100, extra=0):
    mu, nu, r, L = map(mp.mpf, (mu, nu, r, L))
    den = mp.exp(nu)-mp.exp(-mu)
    B = (mp.exp(nu)+1)/den
    D = (mp.exp(-mu)+1)/den
    A = mp.sqrt(r*B)
    logtau = L+mp.log(A)
    nmax = int(mp.ceil((L+mp.log(target))/(nu/2)))+extra
    terms = [[], [], []]
    for n in range(nmax+1):
        # Exact unexpanded energy, scaled before forming moments.
        energy2 = r*B*mp.exp(n*nu)+(1-r)-r*D*mp.exp(-n*mu)
        x = mp.sqrt(energy2)*mp.exp(-logtau)
        w = mp.exp(-x)
        for k in range(3):
            terms[k].append(x**k*w)
    W = [mp.fsum(t) for t in terms]
    u = W[1]/W[0]
    C = W[2]/W[0]-u*u
    return dict(Z=W[0], U_over_tau=u, C=C, C_logtau=C*logtau,
                U_logtau_over_tau=u*logtau, logtau=logtau, nmax=nmax)


def periodic(alpha, L, harmonics=40):
    alpha, L = map(mp.mpf, (alpha, L))
    p = [mp.mpf(0) for _ in range(3)]
    for k in range(1, harmonics+1):
        w = 2*mp.pi*k/alpha
        g = mp.gamma(mp.j*w)*mp.exp(mp.j*w*L)
        for d in range(3):
            p[d] += 2*mp.re((mp.j*w)**d*g)/alpha
    c0 = mp.mpf('.5')-mp.euler/alpha
    Q = L+alpha*(c0+p[0])
    u = (1+alpha*p[1])/Q
    C = (1+alpha*p[1]+alpha*p[2])/Q-u*u
    return dict(Z=Q/alpha, U_over_tau=u, C=C,
                U_leading_coefficient=1+alpha*p[1],
                C_leading_coefficient=1+alpha*(p[1]+p[2]))


def bounds(alpha):
    """Rigorous bounds using |Gamma(i*w)| and a geometric tail bound.

    For each majorant summand, the successive ratio is bounded by
    2**1.5*exp(-pi*pi/alpha), since sqrt(1+w*w) grows at most linearly.
    A first-term geometric majorant therefore bounds all harmonics.
    """
    alpha = mp.mpf(alpha)
    w = 2*mp.pi/alpha
    g = mp.sqrt(mp.pi/(w*mp.sinh(mp.pi*w)))
    rho = 2**mp.mpf('1.5')*mp.exp(-mp.pi**2/alpha)
    if rho >= 1:
        raise ValueError('Use a smaller alpha for this simple bound')
    return dict(U_coefficient_bound=2*w*g/(1-rho),
                C_coefficient_bound=2*w*mp.sqrt(1+w*w)*g/(1-rho))


def serial(v):
    if isinstance(v, dict):return {k:serial(x) for k,x in v.items()}
    if isinstance(v, list):return [serial(x) for x in v]
    return mp.nstr(v, 45) if isinstance(v, mp.mpf) else v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'results')
    args = ap.parse_args(argv)
    mp.mp.dps=100
    result = dict(decimal_precision=100, harmonics=40,
                  definition='alpha=nu/2; L=log(tau/A); A=sqrt(r*B)',
                  coefficient_bounds={})
    for nu in ['0.1','0.2','0.3','4']:
        result['coefficient_bounds'][nu]=bounds(mp.mpf(nu)/2)
    result['large_deformation_checks']=[]
    for L in [2000,2001]:
        v=direct('4','4','1',L)
        approx=periodic('2',L)
        alt=direct('4','4','1',L,extra=10)
        with mp.workdps(70):low=direct('4','4','1',L)
        dc=abs(v['C']-approx['C'])
        cutoff=abs(v['C']-alt['C'])
        roundoff=abs(v['C']-low['C'])
        assert dc<mp.mpf('1e-40') and cutoff<mp.mpf('1e-30')
        assert roundoff<mp.mpf('1e-65')
        result['large_deformation_checks'].append(dict(L=L,direct=v,
            periodic=approx,C_absolute_difference=dc,
            extended_nmax=alt['nmax'], cutoff_N_Nplus10_C_absolute_difference=cutoff,
            precision_70_100_C_absolute_difference=roundoff))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'discrete_high_temperature_check.json').write_text(
        json.dumps(serial(result),indent=2)+'\n')
    for nu,b in result['coefficient_bounds'].items():
        print('nu=',nu,'bound_U=',mp.nstr(b['U_coefficient_bound'],8),
              'bound_C=',mp.nstr(b['C_coefficient_bound'],8))
    for row in result['large_deformation_checks']:
        print('L=',row['L'],'C*logtau=',mp.nstr(row['direct']['C_logtau'],16),
              'periodic residual C=',mp.nstr(row['C_absolute_difference'],5))


if __name__=='__main__':
    main()
