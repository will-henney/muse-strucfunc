import numpy as np

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
