import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import glob
import pandas as pd
from scipy.signal import wiener
from scipy.ndimage.filters import median_filter
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-i", "--interactive",
                  action="store_true", dest="interactive", default=False,
                  help="if flag present, plot to screen, not to file")

(options, args) = parser.parse_args()

# deprecated; now done in the .in files since this can change per-column
# sq1_fb to bias current conversion factor
# scale_factor=1.0794e-9 # nA/DAC w/ DAC=sq1_fb

datasets = []

in_file = open(sys.argv[1], 'r')
a = in_file.readlines()

title = None
islog = 'xy'
ylabel = None
ymin = None
ymax = None
xmin = None
xmax = None
filter_size = 25
color_order = 1
figfilename = None

lines = []

for line in a:
    split_line = line.split()
    if '#' in line:
        continue
    elif "title=" in line:
        title = line.lstrip('title=').rstrip()
    elif "ylabel=" in line:
        ylabel = line.lstrip('ylabel=').rstrip()
    elif "figfilename=" in line:
        figfilename = line.lstrip('figfilename=').rstrip()
    elif "ymax=" in line:
        ymax = float(line.lstrip('ymax=').rstrip())
    elif "ymin=" in line:
        ymin = float(line.lstrip('ymin=').rstrip())
    elif "xmax=" in line:
        xmax = float(line.lstrip('xmax=').rstrip())
    elif "xmin=" in line:
        xmin = float(line.lstrip('xmin=').rstrip())
    elif "islog=" in line:
        islog = line.lstrip('islog=').rstrip()
    elif "color_order=" in line:
        color_order = float(line.lstrip('color_order=').rstrip())
    elif 'filter_size=' in line:
        filter_size = float(line.lstrip('filter_size=').rstrip())
    elif '.meanfft' in split_line[0]:
        dataset = split_line[0].lstrip().rstrip()
        scale_factor = float(split_line[1].lstrip().rstrip())
        label = ' '.join(split_line[2:]).lstrip().rstrip()
        print('dataset=', dataset, '\tlabel=', label)
        datasets.append((dataset, scale_factor, label))
    elif (split_line[0] == 'line'):
        dct = {}

        for p in split_line[1:]:
            pk = p.split('=')[0]
            pv = p.split('=')[1].lstrip().rstrip()
            dct[pk] = pv
        lines.append(dct)
    elif 'daqoutfile(s)=' in line:
        daqoutfiles = (line.lstrip(
            'daqoutfile(s)=').rstrip().lstrip()).split(',')

print('lines=', lines)
print('daqoutfile(s)=', daqoutfiles)

rawctimes = '_'.join([os.path.basename(daqof)[4:-4] for daqof in daqoutfiles])
outdir = 'raw_'+rawctimes
path = 'output/%s/' % (outdir)
print('path=', path)

print('datasets=', datasets)

print('title=', title)
print('islog=', islog)
print('ylabel=', ylabel)
print('ymin=', ymin)
print('ymax=', ymax)
print('xmin=', xmin)
print('xmax=', xmax)
print('color_order=', color_order)
print('filter_size=', filter_size)
print('figfilename=', figfilename)

if ymin is None or ymax is None:
    ymin = 1e-13
    ymax = 1e-6

if title != None:
    print('here')
    plt.suptitle(title, fontsize=8)

# fig=plt.figure(figsize=(12,14))

upper_freq = 200.

print(datasets)

counter = 0
# cmap=plt.get_cmap('Spectral')
cmap = plt.get_cmap('cool')
for (data, scale_factor, label) in datasets:
    print('* plotting', 'data=%s' % data, 'scale_factor=%s' %
          scale_factor, 'label=%s' % label, '...')
    if len(datasets) > 1:
        xcolor = float(datasets.index(
            (data, scale_factor, label)))/float(len(datasets)-1)
    else:
        xcolor = 1.
    if color_order < 0:
        xcolor = 1-xcolor
    color = cmap(xcolor)
    print(('xcolor=%0.3f, color=' % xcolor), color)

    # load and plot normal

    #
    # check if file was compressed
    compression = None
    # find data
    data = glob.glob('output/*/%s' % (data))[0]
    if data.endswith('.bz2'):
        compression = 'bz2'
    # done checking if file was compressed
    #

    datadf = pd.read_csv(data, delim_whitespace=True, error_bad_lines=False, index_col=False,
                         header=None, compression=compression, names=[u'freq', u'Pxx_den'])
    datadf = datadf.dropna(axis=1, how='all')

    # for filtering - http://www.nehalemlabs.net/prototype/blog/2013/04/09/an-introduction-to-smoothing-time-series-in-python-part-ii-wiener-filter-and-smoothing-splines/
    print(np.array(datadf[u'Pxx_den'].values)[0])
    data_asd = np.sqrt(np.array(datadf[u'Pxx_den'].values))*scale_factor
    # plt.loglog(datadf['freq'].values,data_psd,colors[idx],alpha=0.75,label=labels[idx])
    # plt.loglog(datadf['freq'].values,wiener(data_psd,mysize=50),colors[idx],alpha=0.75,label=labels[idx])

    data_rms_x1e6 = (1.e6/1.e9)*scale_factor*np.sqrt(np.sum(
        np.array(datadf[u'Pxx_den'].values))*np.median(np.diff(datadf['freq'].values)))

    print('* Not filtering...')
    # filt_data_asd=median_filter(data_asd,size=filter_size)
    filt_data_asd = data_asd
    if 'x' in islog and 'y' in islog:
        plt.loglog(datadf['freq'].values, filt_data_asd,
                   color=color, alpha=0.8, label=label + f' rms={data_rms_x1e6:.2f} uV')
    elif 'x' in islog:
        plt.semilogx(datadf['freq'].values, filt_data_asd,
                     color=color, alpha=0.8, label=label)
    elif 'y' in islog:
        plt.semilogy(datadf['freq'].values, filt_data_asd,
                     color=color, alpha=0.8, label=label)
    else:
        plt.plot(datadf['freq'].values, filt_data_asd,
                 color=color, alpha=0.8, label=label)

    max_freq = np.max(datadf['freq'].values)
    if max_freq > upper_freq:
        upper_freq = max_freq
    # done loading and plotting data

    # np.savetxt('tmp%d.out'%(counter),(datadf['freq'].values,median_filter(data_asd,size=25)))
    counter = counter+1


plt.xlabel('Frequency (Hz)')
if ylabel != None:
    plt.ylabel(ylabel)
else:
    plt.ylabel('ASD (DAC/rt.Hz)')

plt.gca().set_title(','.join(daqoutfiles), fontsize=6)

if xmin is None:
    xmin = 0.1
if xmax is None:
    xmax = upper_freq
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# noise predictions from https://phy-wiki.princeton.edu/advactwiki/pmwiki.php?n=AdvACTDetector20141125.Uploads?action=download&upname=almnbolo1_update20141125.pdf
# plt.plot([xmin,xmax],[0.9,0.9],'k--',label='AD797A typ. noise @ 1kHz (0.9 nV/rt.Hz -> 3.5 pA/rt.Hz?)')

for line_dict in lines:
    x = np.linspace(xmin, xmax)
    y = float(line_dict['m'])*x+float(line_dict['b'])
    plt.plot(x, y, color=line_dict['c'],
             ls=line_dict['ls'], label=line_dict['label'])

handles, labels = plt.gca().get_legend_handles_labels()
leg = plt.legend(ncol=1, loc='best')
leg.get_frame().set_alpha(0)
leg.get_frame().set_edgecolor('white')

if figfilename is not None and not options.interactive:
    plt.savefig(figfilename)
elif options.interactive:
    plt.show()
else:
    print('* figfilename=', figfilename, ' and --interactive=',
          options.interactive, ', so doing nothing ...')
