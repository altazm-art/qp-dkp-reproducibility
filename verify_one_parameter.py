#!/usr/bin/env python3
"""Independent fixed-level checks of the symmetric spectrum and expansion.

Place next to qp_dkp_reproduce.py and run:
python verify_one_parameter.py --output-dir results
The printed approximation remains unchanged in the five-table reproduction.
"""
import argparse
import json
from pathlib import Path
import mpmath as mp
from qp_dkp_reproduce import peak


def basic(n,eta):
    return mp.fsum(mp.exp((n-1-2*j)*eta) for j in range(n))


def exact(n,eta):
    return mp.sqrt(basic(n+1,eta)+basic(n,eta))


def verify(out):
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    with mp.workdps(70):
        for n in (0,1,5,30):
            y=mp.sqrt(2*n+1)
            coefficient=mp.diff(lambda eta:exact(n,eta),mp.mpf(0),2)/2
            analytic=y*n*(n+1)/12
            printed=y**5/48
            assert abs(coefficient-analytic)<mp.mpf('1e-60')
            assert abs(printed-coefficient-y/48)<mp.mpf('1e-60')
            rows.append(dict(n=n,exact_eta2_coefficient=str(coefficient),
                             printed_eta2_coefficient=str(printed),
                             difference=str(printed-coefficient)))
        worst=mp.mpf(0)
        for eta in map(mp.mpf,('.1','.2','.5')):
            for n in range(31):
                hyperbolic=mp.sqrt(mp.sinh((n+mp.mpf('.5'))*eta)/mp.sinh(eta/2))
                error=abs(hyperbolic/exact(n,eta)-1)
                worst=max(worst,error)
        assert worst<mp.mpf('1e-60')
        gap=mp.sqrt(1+mp.exp(mp.mpf('.1'))+mp.exp(-mp.mpf('.2')))-1
        undeformed=mp.sqrt(3)-1
        result=dict(decimal_precision=70,spectrum_identity_max_relative_error=str(worst),
                    fixed_level_coefficients=rows,off_diagonal_gap=str(gap),
                    undeformed_gap=str(undeformed),
                    source_doi='10.1016/j.physa.2023.129134',
                    source_equations_visually_checked=[19,24,25,26,28],
                    source_pages_checked=[4,5,6,9],
                    expansion_scope='fixed n and tau; no uniform infinite-sum expansion asserted')
    result['r1_peaks']={}
    for mu,nu in ((.1,.1),(.2,.1)):
        tau,c=peak(mu,nu,1,150,.1,10)
        result['r1_peaks'][f'{mu},{nu}']=dict(tau=tau,C=c)
    (out/'one_parameter_source_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('results'))
    verify(parser.parse_args(argv).output_dir)


if __name__=='__main__':main()
