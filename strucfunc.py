import numpy as np
import numba
from concurrent import futures

def strucfunc_numpy(vmap, dlogr=0.15, wmap=None, wmin_factor=1e-3):
    """Calculate structure function by numpy.histogram"""
    ny, nx = vmap.shape
    if wmap is None:
        wmap = np.ones_like(vmap)
    wmin = wmin_factor*np.nanmax(wmap)
    maxr = np.hypot(nx, ny)
    nr = int(np.log10(maxr)/dlogr)
    logr = np.arange(nr)*dlogr
    sf = np.zeros_like(logr)
    nsf = np.zeros_like(logr).astype(int)
    wsf = np.zeros_like(logr)
    weight = np.zeros_like(logr)

    # histogram requires an array that is longer by one
    edges = np.arange(nr+1)*dlogr

    # 2D arrays of x and y coordinates
    ii = np.arange(nx)[None, :]
    jj = np.arange(ny)[:, None]
    for j in range(ny):
        for i in range(nx):
            # everything is a 2D array over the map
            r = np.hypot(ii - i, jj - j)
            dvsq = (vmap - vmap[j, i])**2
            w = wmap[j, i]*wmap
            rmask = (r >= 1.0) & (r <= maxr)
            wmask = wmap > wmin
            if wmap[j, i] > wmin:
                mask = wmask & rmask
                # Histogram weighted by dvsq gives sum of dv^2
                hist, _ = np.histogram(np.log10(r[mask]), bins=edges,
                                       weights=dvsq[mask])
                sf += hist
                # Unweighted histogram gives number of points in each bin
                hist, _ = np.histogram(np.log10(r[mask]), bins=edges)
                nsf += hist
            hist, _ = np.histogram(np.log10(r[rmask]), bins=edges,
                                   weights=dvsq[rmask]*w[rmask])
            wsf += hist
            hist, _ = np.histogram(np.log10(r[rmask]), bins=edges,
                                   weights=w[rmask])
            weight += hist

    return {'log10 r': logr,
            'Sum dv^2': sf,
            'Sum weights': weight,
            'Sum w * dv^2': wsf,
            'N pairs': nsf,
            'Unweighted B(r)': sf/nsf,
            'Weighted B(r)': wsf/weight}

def strucfunc_numba_parallel(vmap, dlogr=0.15, wmap=None, wmin_factor=1e-3):
    """Calculate structure function via naive python algorithm"""
    ny, nx = vmap.shape
    if wmap is None:
        wmap = np.ones((ny, nx))
    wmin = wmin_factor*np.nanmax(wmap)
    maxr = np.hypot(nx, ny)
    nr = int(np.log10(maxr)/dlogr)
    logr = np.arange(nr)*dlogr
    sf = np.zeros((nr,))
    nsf = np.zeros((nr,), dtype=np.int64)
    wsf = np.zeros((nr,))
    weight = np.zeros((nr,))
    sf, weight, wsf, nsf = _strucfunc_numba_parallel(
        ny, nx, nr,
        vmap, wmap,
        wmin, dlogr, maxr,
        sf, weight, wsf, nsf
    )
    return {'log10 r': logr,
            'Sum dv^2': sf,
            'Sum weights': weight,
            'Sum w * dv^2': wsf,
            'N pairs': nsf,
            'Unweighted B(r)': sf/nsf,
            'Weighted B(r)': wsf/weight}


@numba.jit(nopython=True, parallel=True)
def _strucfunc_numba_prange(
        ny, nx, nr,
        vmap, wmap,
        wmin, dlogr, maxr,
        sf, weight, wsf, nsf
):
    # THIS ONE DOES NOT WORK !!!!!!
    # Parallel loop on the outer axis
    for j in numba.prange(ny):
        # make per-thread containers to hold partial sums
        _sf = np.zeros(nr)
        _weight = np.zeros(nr)
        _wsf = np.zeros(nr)
        _nsf = np.zeros(nr, dtype=np.int64)
        for i in range(nx):
            for jj in range(ny):
                for ii in range(i+1, nx):
                    r = np.hypot(ii - i, jj - j)
                    ir = int(np.log10(r)/dlogr)
                    if 0 <= ir < nr:
                        dvsq = (vmap[jj, ii] - vmap[j, i])**2
                        if (wmap[j, i] > wmin) and (wmap[jj, ii] > wmin):
                            _sf[ir] += dvsq
                            _nsf[ir] += 1
                        w = wmap[j, i]*wmap[jj, ii]
                        _wsf[ir] += w*dvsq
                        _weight[ir] += w
        # Now add the partial sums into the global sums
        sf += _sf
        weight += _weight
        wsf += _wsf
        nsf += _nsf
    return sf, weight, wsf, nsf


@numba.jit(nopython=True, parallel=True)
def _strucfunc_numba_parallel(
        ny, nx, nr,
        vmap, wmap,
        wmin, dlogr, maxr,
        sf, weight, wsf, nsf
):
    # Use per-j containers to hold partial results 
    _sf = np.zeros((ny, nr))
    _weight = np.zeros((ny, nr))
    _wsf = np.zeros((ny, nr))
    _nsf = np.zeros((ny, nr), dtype=np.int64)
    for j in numba.prange(ny):
        for i in range(nx):
            for jj in range(ny):
                for ii in range(i+1, nx):
                    r = np.hypot(ii - i, jj - j)
                    ir = int(np.log10(r)/dlogr)
                    if 0 <= ir < nr:
                        dvsq = (vmap[jj, ii] - vmap[j, i])**2
                        if (wmap[j, i] > wmin) and (wmap[jj, ii] > wmin):
                            _sf[j, ir] += dvsq
                            _nsf[j, ir] += 1
                        w = wmap[j, i]*wmap[jj, ii]
                        _wsf[j, ir] += w*dvsq
                        _weight[j, ir] += w
    sf = np.sum(_sf, axis=0)
    weight = np.sum(_weight, axis=0)
    wsf = np.sum(_wsf, axis=0)
    nsf = np.sum(_nsf, axis=0)
    return sf, weight, wsf, nsf

def strucfunc_numba(vmap, dlogr=0.15, wmap=None, wmin_factor=1e-3):
    """Calculate structure function via naive python algorithm"""
    ny, nx = vmap.shape
    if wmap is None:
        wmap = np.ones((ny, nx))
    wmin = wmin_factor*np.nanmax(wmap)
    maxr = np.hypot(nx, ny)
    nr = int(np.log10(maxr)/dlogr)
    logr = np.arange(nr)*dlogr
    sf = np.zeros((nr,))
    nsf = np.zeros((nr,), dtype=np.int64)
    wsf = np.zeros((nr,))
    weight = np.zeros((nr,))
    sf, weight, wsf, nsf = _strucfunc_numba(
        ny, nx, nr,
        vmap, wmap,
        wmin, dlogr, maxr,
        sf, weight, wsf, nsf
    )
    return {'log10 r': logr,
            'Sum dv^2': sf,
            'Sum weights': weight,
            'Sum w * dv^2': wsf,
            'N pairs': nsf,
            'Unweighted B(r)': sf/nsf,
            'Weighted B(r)': wsf/weight}

@numba.jit(nopython=True)
def _strucfunc_numba(
        ny, nx, nr,
        vmap, wmap,
        wmin, dlogr, maxr,
        sf, weight, wsf, nsf
):
    for j in range(ny):
        for i in range(nx):
            for jj in range(ny):
                for ii in range(i+1, nx):
                    r = np.hypot(ii - i, jj - j)
                    ir = int(np.log10(r)/dlogr)
                    if 0 <= ir < nr:
                        dvsq = (vmap[jj, ii] - vmap[j, i])**2
                        if (wmap[j, i] > wmin) and (wmap[jj, ii] > wmin):
                            sf[ir] += dvsq
                            nsf[ir] += 1
                        w = wmap[j, i]*wmap[jj, ii]
                        wsf[ir] += w*dvsq
                        weight[ir] += w
    return sf, weight, wsf, nsf

def strucfunc_python(vmap, dlogr=0.15, wmap=None, wmin_factor=1e-3):
    """Calculate structure function via naive python algorithm"""
    ny, nx = vmap.shape
    if wmap is None:
        wmap = np.ones_like(vmap)
    wmin = wmin_factor*np.nanmax(wmap)
    maxr = np.hypot(nx, ny)
    nr = int(np.log10(maxr)/dlogr)
    logr = np.arange(nr)*dlogr
    sf = np.zeros_like(logr)
    nsf = np.zeros_like(logr).astype(int)
    wsf = np.zeros_like(logr)
    weight = np.zeros_like(logr)

    for j in range(ny):
        for i in range(nx):
            for jj in range(ny):
                for ii in range(i+1, nx):
                    r = np.hypot(ii - i, jj - j)
                    ir = int(np.log10(r)/dlogr)
                    if 0 <= ir < nr:
                        dvsq = (vmap[jj, ii] - vmap[j, i])**2
                        if (wmap[j, i] > wmin) and (wmap[jj, ii] > wmin):
                            sf[ir] += dvsq
                            nsf[ir] += 1
                        w = wmap[j, i]*wmap[jj, ii]
                        wsf[ir] += w*dvsq
                        weight[ir] += w

    return {'log10 r': logr,
            'Sum dv^2': sf,
            'Sum weights': weight,
            'Sum w * dv^2': wsf,
            'N pairs': nsf,
            'Unweighted B(r)': sf/nsf,
            'Weighted B(r)': wsf/weight}

def test_strucfunc(n=50, func=strucfunc_python, **kwds):
    """Set up arrays for stucture function and run it"""
    ny, nx = n, n
    vels = np.random.normal(size=(ny, nx))
    bright = np.ones((ny, nx))
    rslt = func(vmap=vels, wmap=bright, **kwds)
    return ['{} :: {}'.format(k, list(v)) for (k, v) in rslt.items()]
