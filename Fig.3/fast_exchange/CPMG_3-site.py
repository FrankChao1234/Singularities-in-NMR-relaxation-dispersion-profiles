# 
#

import sys
from string import *
import math
import numpy
import random


#------------------------------------------------------------------------------
# calculate the perturbation

def cal_perturb(za,ya,xa,zb,yb,xb,zc,yc,xc,incre,Wa,Wb,Wc,kab,kba,kbc,kcb,kac,kca,R1a,R1b,R1c,R2a,R2b,R2c):
  const1 = 2*3.1415926
  inter_za = -((kab+kac)*const1+R1a)*incre*za +kba*incre*const1*zb +kca*incre*const1*zc
  inter_ya = Wa*incre*const1*xa -((kab+kac)*const1+R2a)*incre*ya +kba*incre*const1*yb +kca*incre*const1*yc
  inter_xa = -((kab+kac)*const1+R2a)*incre*xa +kba*incre*const1*xb +kca*incre*const1*xc -Wa*incre*const1*ya 
  inter_zb = kab*incre*const1*za +kcb*incre*const1*zc -((kba+kbc)*const1+R1b)*incre*zb 
  inter_yb = Wb*incre*const1*xb +kab*incre*const1*ya +kcb*incre*const1*yc -((kba+kbc)*const1+R2b)*incre*yb
  inter_xb = kab*incre*const1*xa +kcb*incre*const1*xc -((kba+kbc)*const1+R2b)*incre*xb -Wb*incre*const1*yb 
  inter_zc = kac*incre*const1*za +kbc*incre*const1*zb -((kcb+kca)*const1+R1c)*incre*zc 
  inter_yc = Wc*incre*const1*xc +kac*incre*const1*ya +kbc*incre*const1*yb -((kcb+kca)*const1+R2c)*incre*yc
  inter_xc = kac*incre*const1*xa +kbc*incre*const1*xb -((kcb+kca)*const1+R2c)*incre*xc -Wc*incre*const1*yc
  return inter_za, inter_ya, inter_xa, inter_zb, inter_yb, inter_xb, inter_zc, inter_yc, inter_xc


#------------------------------------------------------------------------------
# calculate the bulk magnetization (using Bloch-McConnell equation with second order perturbation)

def cal_relax(a,b,c,d,e,f,g,h,q1,q2,t,m):

  const1 = 2*3.1415926
  n1 = 1000
  n3 = 10
  Wa = a   # offset frequency of state A (kHz)
  Wb = b   # offset frequency of state B (kHz)
  Wc = c   # offset frequency of state C (kHz)
  kexab = d   # exchange rate between A and B (kHz)
  kexbc = e   # exchange rate between B and C (kHz)
  kexac = f   # exchange rate between A and C (kHz)
  pb = g
  pc = h
  pa = 1-pb-pc
  kab = kexab*pb/float(pa+pb)   # forward reaction rate A->B (kHz)
  kba = kexab*pa/float(pa+pb)   # reverse reaction rate B->A (kHz)
  kbc = kexbc*pc/float(pb+pc)   # forward reaction rate B->C (kHz)
  kcb = kexbc*pb/float(pb+pc)   # reverse reaction rate C->B (kHz)
  kac = kexac*pc/float(pa+pc)   # forward reaction rate A->C (kHz)
  kca = kexac*pa/float(pa+pc)   # reverse reaction rate C->A (kHz)
  R1a = q1   # R1 for state A (ks-1)
  R1b = q1   # R1 for state B (ks-1)
  R1c = q1   # R1 for state C (ks-1)
  R2a = q2   # R2 for state A (ks-1)
  R2b = q2   # R2 for state B (ks-1)
  R2c = q2   # R2 for state C (ks-1)
  Tot = int(t*n1*n3) 
  incre = t/float(Tot)
  for n in range(m):
    coor_za = {}
    coor_ya = {}
    coor_xa = {}
    coor_zb = {}
    coor_yb = {}
    coor_xb = {}
    coor_zc = {}
    coor_yc = {}
    coor_xc = {}
    for i in range(Tot+1):
      coor_za[i] = []
      coor_ya[i] = []
      coor_xa[i] = []
      coor_zb[i] = []
      coor_yb[i] = []
      coor_xb[i] = []
      coor_zc[i] = []
      coor_yc[i] = []
      coor_xc[i] = []
      if i == 0:
        if n == 0:
          coor_za[0] = 0  # initial point
          coor_ya[0] = pa  # initial point
          coor_xa[0] = 0  # initial point
          coor_zb[0] = 0  # initial point
          coor_yb[0] = pb  # initial point
          coor_xb[0] = 0  # initial point
          coor_zc[0] = 0  # initial point
          coor_yc[0] = pc  # initial point
          coor_xc[0] = 0  # initial point
        else:
          coor_za[0] = coor_za_fa
          coor_ya[0] = coor_ya_fa
          coor_xa[0] = coor_xa_fa
          coor_zb[0] = coor_zb_fa
          coor_yb[0] = coor_yb_fa
          coor_xb[0] = coor_xb_fa
          coor_zc[0] = coor_zc_fa
          coor_yc[0] = coor_yc_fa
          coor_xc[0] = coor_xc_fa
      else:
        za = coor_za[i-1]
        ya = coor_ya[i-1]
        xa = coor_xa[i-1]
        zb = coor_zb[i-1]
        yb = coor_yb[i-1]
        xb = coor_xb[i-1]
        zc = coor_zc[i-1]
        yc = coor_yc[i-1]
        xc = coor_xc[i-1]
        inter_za,inter_ya,inter_xa,inter_zb,inter_yb,inter_xb,inter_zc,inter_yc,inter_xc = cal_perturb(za,ya,xa,zb,yb,xb,zc,yc,xc,incre,Wa,Wb,Wc,kab,kba,kbc,kcb,kac,kca,R1a,R1b,R1c,R2a,R2b,R2c)
        inter_za1,inter_ya1,inter_xa1,inter_zb1,inter_yb1,inter_xb1,inter_zc1,inter_yc1,inter_xc1 = cal_perturb(inter_za,inter_ya,inter_xa,inter_zb,inter_yb,inter_xb,inter_zc,inter_yc,inter_xc,incre,Wa,Wb,Wc,kab,kba,kbc,kcb,kac,kca,R1a,R1b,R1c,R2a,R2b,R2c)
        coor_za[i] = za+inter_za+0.5*inter_za1
        coor_ya[i] = ya+inter_ya+0.5*inter_ya1
        coor_xa[i] = xa+inter_xa+0.5*inter_xa1
        coor_zb[i] = zb+inter_zb+0.5*inter_zb1
        coor_yb[i] = yb+inter_yb+0.5*inter_yb1
        coor_xb[i] = xb+inter_xb+0.5*inter_xb1
        coor_zc[i] = zc+inter_zc+0.5*inter_zc1
        coor_yc[i] = yc+inter_yc+0.5*inter_yc1
        coor_xc[i] = xc+inter_xc+0.5*inter_xc1

    for i in range(Tot+1):
      i = i + Tot+1
      coor_za[i] = []
      coor_ya[i] = []
      coor_xa[i] = []
      coor_zb[i] = []
      coor_yb[i] = []
      coor_xb[i] = []
      coor_zc[i] = []
      coor_yc[i] = []
      coor_xc[i] = []
      if i == Tot+1:
        coor_za[Tot+1] = -coor_za[Tot]
        coor_ya[Tot+1] = coor_ya[Tot]
        coor_xa[Tot+1] = -coor_xa[Tot]
        coor_zb[Tot+1] = -coor_zb[Tot]
        coor_yb[Tot+1] = coor_yb[Tot]
        coor_xb[Tot+1] = -coor_xb[Tot]
        coor_zc[Tot+1] = -coor_zc[Tot]
        coor_yc[Tot+1] = coor_yc[Tot]
        coor_xc[Tot+1] = -coor_xc[Tot]
      else:
        za = coor_za[i-1]
        ya = coor_ya[i-1]
        xa = coor_xa[i-1]
        zb = coor_zb[i-1]
        yb = coor_yb[i-1]
        xb = coor_xb[i-1]
        zc = coor_zc[i-1]
        yc = coor_yc[i-1]
        xc = coor_xc[i-1]
        inter_za,inter_ya,inter_xa,inter_zb,inter_yb,inter_xb,inter_zc,inter_yc,inter_xc = cal_perturb(za,ya,xa,zb,yb,xb,zc,yc,xc,incre,Wa,Wb,Wc,kab,kba,kbc,kcb,kac,kca,R1a,R1b,R1c,R2a,R2b,R2c)
        inter_za1,inter_ya1,inter_xa1,inter_zb1,inter_yb1,inter_xb1,inter_zc1,inter_yc1,inter_xc1 = cal_perturb(inter_za,inter_ya,inter_xa,inter_zb,inter_yb,inter_xb,inter_zc,inter_yc,inter_xc,incre,Wa,Wb,Wc,kab,kba,kbc,kcb,kac,kca,R1a,R1b,R1c,R2a,R2b,R2c)
        coor_za[i] = za+inter_za+0.5*inter_za1
        coor_ya[i] = ya+inter_ya+0.5*inter_ya1
        coor_xa[i] = xa+inter_xa+0.5*inter_xa1
        coor_zb[i] = zb+inter_zb+0.5*inter_zb1
        coor_yb[i] = yb+inter_yb+0.5*inter_yb1
        coor_xb[i] = xb+inter_xb+0.5*inter_xb1
        coor_zc[i] = zc+inter_zc+0.5*inter_zc1
        coor_yc[i] = yc+inter_yc+0.5*inter_yc1
        coor_xc[i] = xc+inter_xc+0.5*inter_xc1

    coor_za_fa = coor_za[2*Tot+1]
    coor_ya_fa = coor_ya[2*Tot+1]
    coor_xa_fa = coor_xa[2*Tot+1]
    coor_zb_fa = coor_zb[2*Tot+1]
    coor_yb_fa = coor_yb[2*Tot+1]
    coor_xb_fa = coor_xb[2*Tot+1]
    coor_zc_fa = coor_zc[2*Tot+1]
    coor_yc_fa = coor_yc[2*Tot+1]
    coor_xc_fa = coor_xc[2*Tot+1]

  return coor_za_fa, coor_ya_fa, coor_xa_fa, coor_zb_fa, coor_yb_fa, coor_xb_fa, coor_zc_fa, coor_yc_fa, coor_xc_fa


#------------------------------------------------------------------------------
# main script

const = 2*3.1415926*1000
R1 = 1/float(1000)
R2 = 10/float(1000)

Wa = 0
Wb = 300/float(const)
kexab = (500)/float(const)
kexbc = 2*(10**(2+1))/float(const)
kexac = 0
pa = 0.85
pb = 0.05
pc = 0.15

for j in range(40):
  for i in range(40):
    Wc = (500-(25*j))/float(const)
    tcp = 10/float(1+i)
    cyc = int(20/float(tcp))

    coor_za, coor_ya, coor_xa, coor_zb, coor_yb, coor_xb, coor_zc, coor_yc, coor_xc = cal_relax(Wa,Wb,Wc,kexab,kexbc,kexac,pb,pc,R1,R2,tcp,cyc)
    trans_relax = math.log(math.sqrt(coor_ya**2+coor_xa**2)+math.sqrt(coor_yb**2+coor_xb**2)+math.sqrt(coor_yc**2+coor_xc**2))/float(-2*0.001*cyc*tcp)

    print (1000/float(4*tcp), -25*j, trans_relax)
  print ( )





