#!/usr/bin/env python3
"""Compute Tables I--V, all 13 figures, and the numerical validation checks.

Usage: python reproduce_all.py --output-dir reproduced
Use --quick to omit the independent arbitrary-precision checks only.
All plot data are recomputed; reference_results is not used as an input.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy
import mpmath
import matplotlib
import qp_dkp_reproduce as q
import plot_all_figures
import verify_one_parameter
import verify_discrete_high_temperature

VERSION=Path(__file__).with_name('VERSION').read_text(encoding='utf-8').strip()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('reproduced'))
    parser.add_argument('--quick',action='store_true',help='Omit independent arbitrary-precision audits')
    args=parser.parse_args(argv)
    out=args.output_dir.resolve();data=out/'data';figures=out/'figures'
    out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    print('1/4 Recomputing tables and numerical data',flush=True)
    q.main(['--output-dir',str(data)]+([] if args.quick else ['--full']))
    if args.quick:q.figure_data(data)
    print('2/4 Checking symmetric spectrum and discrete high-temperature correction',flush=True)
    if not args.quick:
        verify_one_parameter.main(['--output-dir',str(data)])
        verify_discrete_high_temperature.main(['--output-dir',str(data)])
    print('3/4 Computing the remaining plot data and drawing all 13 figures',flush=True)
    coverage=plot_all_figures.main(['--data-dir',str(data),'--output-dir',str(figures)])
    print('4/4 Recording validation and generated-file hashes',flush=True)
    summary=dict(version=VERSION,quick=args.quick,seconds=time.monotonic()-start,
                 python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                 mpmath=mpmath.__version__,matplotlib=matplotlib.__version__,
                 tables=['I','II','III','IV','V'],figure_count=coverage['figure_count'],
                 independent_precision_checks=not args.quick,
                 reference_data_used_as_calculation_input=False)
    (out/'run_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    hashes={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(out.rglob('*')) if f.is_file() and f.name!='generated_manifest.json'}
    (out/'generated_manifest.json').write_text(json.dumps({'sha256':hashes},indent=2)+'\n')
    print('Completed:',out,'in',round(summary['seconds'],1),'seconds',flush=True)


if __name__=='__main__':main()
