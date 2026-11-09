# Numerical prescription

The common kernel is `qp_dkp_reproduce.py`. It evaluates

    [n] = (exp(n*nu) - exp(-n*mu)) / (exp(nu) - exp(-mu))
    A_n = [n+1] + [n] - 1
    E_n = sqrt(1 + r*A_n)
    K_n = (E_n-1)/r = A_n/(sqrt(1+r*A_n)+1).

At the undeformed origin A_n=2n; in the matched NR model K_n=A_n/2.
Logarithms and expm1 avoid cancellation/overflow. The ground level is handled
explicitly. The positive square root and the fixed offset define the model.

With x=tau/r, canonical weights are exp(-K_n/x). The kernel forms scaled
excitation moments using math.fsum and a centered variance for heat capacity.
The unshifted logarithmic partition sum is -1/tau+log(W0). Thus F, U, S and C
are obtained from one consistent canonical sum without temperature finite
differences. Small-r plots report exp(log(W0)) and x*mean(K_n/x).

## Cutoffs

All cutoffs are inclusive: n=0,...,N.

| Calculation | Rule |
|---|---|
| Baseline deformed curves, Figures 3–5 | N=100, r=1, 199 temperatures .1..10 |
| Baseline undeformed curves | ceil(max(100, ((-tau*ln(1e-14))^2-1)/(2*r))) |
| Table I | N=150; peak searches .1<=tau<=10 |
| Tables II–III and small-r, fixed-Sigma, NR/DKP comparisons | N=200 |
| Adaptive Figures 6–7, 12–13 and Table V | Smallest N with E_N/tau>=62 |
| Table IV | Common cutoff sqrt(2N+1)>=1+62*tau; compare 62 with 68 |

The baseline undeformed cutoff reaches 51959 at tau=10. The adaptive
undeformed cutoff reaches 192200 there; the largest deformed cutoff at
10^100 is 4668. The latter very large temperature does not cause an enormous
level count because deformed energies grow exponentially.

Peak searches bracket stationary points on a logarithmic grid and refine the
zero of the third-central-moment condition using Brent's method. At fixed
x the condition is <(K-<K>)^3> - 2*x*Var(K)=0. The largest interior maximum is
selected and compared against search endpoints. The inclusive cutoff is
raised from 200 to 300 for the peak/scanned checks.

## Whole-tail bounds

The terminal threshold is only a cutoff selection rule. For r=1 deformed
levels, E_(n+1)>=exp(nu/2)*E_n gives, with x_*=E_(N+1)/tau,

    B_k = x_*^k*exp(-x_*) + (2/nu)*integral_(x_*)^infinity x^(k-1)*exp(-x) dx.

For undeformed levels the corresponding bound is

    B_k = x_*^k*exp(-x_*) + (tau^2/r)*integral_(x_*)^infinity x^(k+1)*exp(-x) dx.

Here k=0,1,2 and the quoted checks have x_*>=2. If z is the truncated
partition sum and m_k its normalized raw scaled-energy moment, then

    relative error in Z <= B_0/z
    e_k = (B_k + abs(m_k)*B_0)/z
    absolute error in C <= e_2 + 2*abs(m_1)*e_1 + e_1^2.

No geometric r=1 bound is asserted for arbitrary deformed r.

## Precision audits

- Binary64: 53-bit significand, log/expm1 spectra and math.fsum reductions.
- 60 decimal digits: N=100 versus 150 at all 199 baseline temperatures for
  mu=.2, nu=.1.
- 70 digits: matched NR/DKP moments, first-order peak correction, independent
  deformation differentiation and fixed-Sigma endpoint/central peak checks.
- 80 digits: adaptive N versus N+50 for all four pairs at 10^12 and 10^100.
- 100 digits, compared with 70: the discrete high-temperature example at
  mu=nu=4, r=1, L=2000 and 2001; N versus N+10 and a Fourier-series check.
- 110 digits: N=150 versus 200 at mu=.2, nu=.1, r=.01, tau=10.

The scripts construct high-precision inputs from decimal strings. The
recorded checks separate truncation, arithmetic precision and asymptotic
approximations. They do not convert a nominal digit count into a universal
error guarantee.
