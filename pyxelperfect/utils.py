import numpy as np
def getPatch(arr, idx, radius=3, fill=None):
    """
    Gets surrounding elements from a numpy array

    Parameters:
    arr (ndarray of rank N): Input array
    idx (N-Dimensional Index): The index at which to get surrounding elements. If None is specified for a particular axis,
        the entire axis is returned.
    radius (array-like of rank N or scalar): The radius across each axis. If None is specified for a particular axis,
        the entire axis is returned.
    fill (scalar or None): The value to fill the array for indices that are out-of-bounds.
        If value is None, only the surrounding indices that are within the original array are returned.

    Returns:
    ndarray: The surrounding elements at the specified index
    """

    assert len(idx) == len(arr.shape)

    if np.isscalar(radius):
        radius = tuple([radius for i in range(len(arr.shape))])

    slices = []
    paddings = []
    for axis in range(len(arr.shape)):
        if idx[axis] is None or radius[axis] is None:
            slices.append(slice(0, arr.shape[axis]))
            paddings.append((0, 0))
            continue

        r = radius[axis]
                l = idx[axis] - r
        r = idx[axis] + r

        pl = 0 if l > 0 else abs(l)
        pr = 0 if r < arr.shape[axis] else r - arr.shape[axis] + 1

        slices.append(slice(max(0, l), min(arr.shape[axis], r+1)))
        paddings.append((pl, pr))

    if fill is None:
        return arr[tuple(slices)]
    return np.pad(arr[tuple(slices)], paddings, 'constant', constant_values=fill)


