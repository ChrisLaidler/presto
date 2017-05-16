#!/usr/bin/env python

import matplotlib
import matplotlib as mpl
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from numpy.random import randn
import csv
import pylab
from mpl_toolkits.mplot3d import Axes3D
import os.path
from matplotlib.collections import PolyCollection
import shutil
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import os
#from matplotlib.colors import LogNorm
import itertools

from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def read_2D(fname) :
  nox    = 0
  noy    = 0
  lineNo = 0
  rownum = 0
  harm   = 0
  X=[]
  Y=[]
  Z=[]
  vv=0
  with open(fname, 'rb') as f:
    reader = csv.reader(f, dialect='excel-tab')
    for row in reader :
      colnum = 0
      vector = []
      for col in row :
        if rownum == 0 :
          if colnum == 0 :
            harm = float(col)
          else :
            X.append(float(col))
            
        else :
          if colnum == 0 :
            Y.append(float(col))
          else :
            if col == "NaN" :
              vector.append(np.nan)
              vv+=1
            else :
              vector.append(float(col))
        colnum = colnum +1
      if rownum != 0 :
        Z.append(vector)
      rownum = rownum + 1 
    X = np.array(X)
    Y = np.array(Y)
    X1, Y1 = np.meshgrid(X,Y)
    Z = np.ma.array(Z, mask=np.isnan(Z))
    return (harm, X1, Y1, Z)

def read_rdat(fname) :
  nox    = 0
  noy    = 0
  lineNo = 0
  rownum = 0  
  VALS=[]
  with open(fname, 'rb') as f:
    reader = csv.reader(f, dialect='excel-tab')
    for row in reader :
      vector1 = []
      vector2 = []
      ncol = 0
      for col in row :
        vector1.append(float(col))
        ncol+=1
        if (ncol==2) :
          vector2.append(vector1)
          vector1=[]
          ncol=0
      VALS.append(vector2)
      rownum = rownum + 1 
    Z = np.array(VALS)
    return (Z)

lrg=sys.argv[1]

if len(sys.argv) > 2 :
  sz=float(sys.argv[2])
else:
  sz=15
  
if len(sys.argv) > 3 :
  ratio=float(sys.argv[3])
else:
  ratio=1.1

fontSz = np.round(sz*1.4)
mpl.rcParams.update({'font.size'		: fontSz   })
mpl.rcParams.update({'lines.markeredgewidth'	: sz * 1.5 })
mpl.rcParams.update({'lines.markersize'		: sz * 1.0 })	#size of ticks 
mpl.rcParams.update({'xtick.major.size'		: sz * 0.3 })
mpl.rcParams.update({'ytick.major.size'		: sz * 0.3 })


fnm=os.path.basename(lrg)
dnm=os.path.dirname(lrg)
filename, file_extension  = os.path.splitext( fnm )

harm,xLrg,yLrg,zLrg = read_2D(lrg)

vMin = vmin=zLrg.min()
vMax = vmax=zLrg.max()

z=yLrg[:,0]
r=xLrg[0,:]
zFine=np.arange(z[-1],z[1],0.1)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(sz*ratio,sz))
x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)

#lvls = np.logspace(0,vMax,20)
#print "Max: %.7f"%(zLrg.max())

axLW=1.5
ax1 = plt.subplot(1,1,1)
#CS = ax1.contour(xLrg, yLrg, zLrg, 20, linewidths=0.01, colors='k', vmax=vMax, vmin=vMin)
CS = ax1.contourf(xLrg, yLrg, zLrg, 280, cmap=plt.cm.rainbow, vmax=vMax, vmin=vMin )
#CS = ax1.contourf(xLrg, yLrg, zLrg, 40, cmap=plt.cm.rainbow, levels = lvls, norm = LogNorm() )
ln = ax1.plot([r[0],r[-1]], [0,0], 'k-', linewidth=axLW)
ln = ax1.plot([0,0], [z[0],z[-1]], 'k-', linewidth=axLW)

lw=2.0

ax1.xaxis.set_major_formatter(x_formatter)
ax1.yaxis.set_major_formatter(y_formatter)

ax1.set_ylim([yLrg[0][0],yLrg[yLrg.shape[0]-1][0]])
ax1.set_xlim([xLrg[0][0],xLrg[0][xLrg.shape[1]-1]])
plt.ylabel('z - Frequency derivative')
plt.xlabel('r - FFT bin')

if yLrg[0][0] > yLrg[yLrg.shape[0]-1][0] :
  plt.gca().invert_yaxis()

if (0): ## Sampling point ticks
  xticks = np.arange(np.floor(xLrg[0][0]), np.ceil(xLrg[0][xLrg.shape[1]-1]), 0.5/harm)                                              
  yticks = np.arange(-200, 200, 2/harm)
  gridpoints = list( itertools.product(xticks, yticks) )
  plt.scatter(*zip(*gridpoints), marker = '.', color='k', s=100)

divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "2%", pad="1%")
cbar = plt.colorbar(CS, cax=cax, orientation="vertical" )
cbar.formatter.set_useOffset(False) 
cbar.update_ticks() 

fig.set_tight_layout(True)

nm="%s/%s.png" % ( dnm, filename )
plt.savefig(nm)

plt.close('all')

