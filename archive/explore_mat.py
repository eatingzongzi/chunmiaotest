"""Explore one cell's detailed data to understand feature extraction."""
import h5py
import numpy as np

path = 'data/mit_stanford/2017-05-12_batchdata_updated_struct_errorcorrect.mat'
f = h5py.File(path, 'r')
batch = f['batch']

# Explore first cell's summary data
summ_ref = batch['summary'][0, 0]
summ = f[summ_ref]
print('=== Summary group ===')
for k in summ.keys():
    item = summ[k]
    if isinstance(item, h5py.Dataset):
        print(f'{k}: shape={item.shape}, dtype={item.dtype}')
        vals = item[()]
        print(f'  first 5: {vals.flatten()[:5]}')
        print(f'  last 5: {vals.flatten()[-5:]}')
    else:
        print(f'{k}: Group')

# Explore first cell's cycles data
cyc_ref = batch['cycles'][0, 0]
cyc = f[cyc_ref]
print('\n=== Cycles group ===')
for k in cyc.keys():
    item = cyc[k]
    if isinstance(item, h5py.Dataset):
        print(f'{k}: shape={item.shape}, dtype={item.dtype}')
        vals = item[()]
        # Check if it's a cell array or regular array
        if vals.dtype == h5py.ref_dtype or vals.dtype == np.dtype('O'):
            nrefs = len(vals.flatten())
            print(f'  references: {nrefs} entries')
            # Look at first entry
            if nrefs > 0:
                ref0 = vals.flatten()[0]
                deref = f[ref0]
                if isinstance(deref, h5py.Dataset):
                    print(f'  ref[0]: Dataset, shape={deref.shape}')
                    print(f'    first 10: {deref[()].flatten()[:10]}')
                else:
                    print(f'  ref[0]: Group, keys={list(deref.keys())}')
        else:
            print(f'  values (first 10): {vals.flatten()[:10]}')
    else:
        print(f'{k}: Group, keys={list(item.keys())}')

# Cycle life
cl_ref = batch['cycle_life'][0, 0]
cl = f[cl_ref]
print(f'\n=== Cycle life ===')
print(f'Value: {cl[()]}')

# How many cells total
print(f'\nTotal cells in batch: {batch["summary"].shape[0]}')

# Check number of cycles for first cell
print(f'\nSummary cycle count: {summ["cycle"].shape}')

# Explore discharge_dQdV structure in cycles
print('\n=== discharge_dQdV structure (first 3 cycles) ===')
for i in range(min(3, len(cyc['discharge_dQdV'][()].flatten()))):
    ref = cyc['discharge_dQdV'][()].flatten()[i]
    dqdv = f[ref]
    if isinstance(dqdv, h5py.Dataset):
        vals = dqdv[()]
        print(f'Cycle {i}: shape={dqdv.shape}, values (first 20): {vals.flatten()[:20]}')
    else:
        print(f'Cycle {i}: Group, keys={list(dqdv.keys())}')

f.close()
