import pywt
import matplotlib.pyplot as plt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis

max_level = 10
shape = (480,480)
ii,rem = divmod(10,2)
# figure size for 3 as reference
ref_size = [14,8]
fig_size = [s*50 for s in ref_size]
f,ax = plt.subplots(figsize=ref_size)
for level in range(1,max_level+1,1):
    print(level)
    ax.clear()
    draw_2d_wp_basis(shape,wavedec2_keys(level),ax=ax,
                     label_levels=max_level)
    ax.set_title('{} level\ndecomposition'.format(level))
    f.savefig('wavelet-decomp-label-{}-level.png'.format(level))
