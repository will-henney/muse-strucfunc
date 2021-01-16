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

def strucfunc_numba_futures(vmap, dlogr=0.15,
                            wmap=None, wmin_factor=1e-3, numthreads=4):
    """Calculate structure function via jitted, threaded python algorithm"""
    ny, nx = vmap.shape
    if wmap is None:
        wmap = np.ones((ny, nx))
    wmin = wmin_factor*np.nanmax(wmap)
    maxr = np.hypot(nx, ny)
    nr = int(np.log10(maxr)/dlogr)
    logr = np.arange(nr)*dlogr

    # arrays to hold the total accumulation sums
    sf = np.zeros_like(logr)
    nsf = np.zeros_like(logr).astype(int)
    wsf = np.zeros_like(logr)
    weight = np.zeros_like(logr)

    @numba.jit(nogil=True, nopython=True)
    def _strucfunc_chunk(j1, j2):
        """Pure python function designed to be jitted and threaded.

        Finds partial contribution to structure function from a chunk
        (vmap[j1:j2, :] - vmap[:, :])^2
        """
        # Per-thread arrays of partial accumulation sums
        _sf = np.zeros(nr)
        _nsf = np.zeros(nr, dtype=np.int64)
        _wsf = np.zeros(nr)
        _weight = np.zeros(nr)
        # outer loop is over only those rows that are in chunk 
        for j in range(j1, j2):
            for i in range(nx):
                for jj in range(ny):
                    for ii in range(i+1, nx):
                        r = np.hypot(ii - i, jj - j)
                        ir = int(np.log10(r)/dlogr)
                        if 0 <= ir < nr:
                            dvsq = (vmap[jj, ii] - vmap[j, i])**2
                            if (wmap[j, i] > wmin) and (wmap[jj, ii] > wmin):
                                _sf[ir] += dvsq
                                _nsf[ir] += 1.0
                            w = 1.0 # wmap[j, i]*wmap[jj, ii]
                            _wsf[ir] = _wsf[ir] + w*dvsq
                            _weight[ir] = _weight[ir] + w
        # return the partial accumulated sums for this chunk
        return _sf, _nsf, _wsf, _weight

    # Do integer division with remainder
    chunklen, extras = divmod(ny, numthreads)
    # So "normal" chunks have length chunklen
    # While the first `extras` chunks have length (chunklen  + 1)
    long_chunks = [[i*(chunklen + 1), (i+1)*(chunklen + 1)]
                   for i in range(extras)]
    i0 = extras*(chunklen + 1)
    normal_chunks = [[i*chunklen + i0, (i+1)*chunklen + i0]
                     for i in range(chunklen - extras)]
    chunklimits = long_chunks + normal_chunks

    # Spawn one thread per chunk
    with futures.ThreadPoolExecutor(max_workers=numthreads) as ex:
        chunks = [
            ex.submit(_strucfunc_chunk, j1, j2)
            for j1, j2 in chunklimits
        ]
        for f in futures.as_completed(chunks):
            # As each thread finishes, add its partial result into the
            # arrays
            _sf, _nsf, _wsf, _weight = f.result()
            sf += np.array(_sf)
            nsf += np.array(_nsf)
            wsf += np.array(_wsf)
            weight += np.array(_weight)

    return {'log10 r': logr,
            'Sum dv^2': sf,
            'Sum weights': weight,
            'Sum w * dv^2': wsf,
            'N pairs': nsf,
            'Unweighted B(r)': sf/nsf,
            'Weighted B(r)': wsf/weight}

@numba.jit
def strucfunc_numba(vmap, dlogr=0.15, wmap=None, wmin_factor=1e-3):
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
