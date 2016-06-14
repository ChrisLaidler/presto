import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from numpy.random import randn
import csv
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import os.path
from matplotlib.collections import PolyCollection
import shutil
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import os


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
            harm = int(col)
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
fnm=os.path.basename(lrg)
dnm=os.path.dirname(lrg)
filename, file_extension  = os.path.splitext( fnm )

harm,xLrg,yLrg,zLrg = read_2D(lrg)

vMin = vmin=zLrg.min()
vMax = vmax=zLrg.max()
#if vMin > 0 :
#  vMin = 0

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(22,12))
x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)

ax1 = plt.subplot(1,1,1)
CS = ax1.contour(xLrg, yLrg, zLrg, 20, linewidths=0.5, colors='k', vmax=vMax, vmin=vMin)
CS = ax1.contourf(xLrg, yLrg, zLrg, 20, cmap=plt.cm.rainbow, vmax=vMax, vmin=vMin)
ln = ax1.plot([xLrg[0][0],xLrg[0][xLrg.shape[1]-1]], [0,0], 'k-')

ax1.xaxis.set_major_formatter(x_formatter)
ax1.yaxis.set_major_formatter(y_formatter)

ax1.set_ylim([yLrg[0][0],yLrg[yLrg.shape[0]-1][0]])
ax1.set_xlim([xLrg[0][0],xLrg[0][xLrg.shape[1]-1]])
plt.ylabel('z - frequency derivative')
plt.xlabel('r - FFT bin')

if yLrg[0][0] > yLrg[yLrg.shape[0]-1][0] :
  plt.gca().invert_yaxis()

divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "2%", pad="1%")
#plt.colorbar(CS, cax=cax, orientation="vertical", format='%.5f')
plt.colorbar(CS, cax=cax, orientation="vertical" )

fig.set_tight_layout(True)

nm="%s/%s.png" % ( dnm, filename )
plt.savefig(nm)

plt.close('all')

