#
# 
#

from sys import *
from string import *
import math
import random
import numpy as np
from scipy.optimize import minimize


#------------------------------------------------------------------------------
# read the data
#

def read_data(file):

  relax = {}
  code = []
  relax['R1rho'] = {}
  relax['R2rho'] = {}
  f = open(file,'r')
  lines = f.readlines()
  f.close()
  for line in lines:
    entry = line.split()
    if entry[2] not in code:
      code.append(entry[2])
    if entry[0] == 'R1rho':
      value = (float(entry[3]),float(entry[4]),float(entry[5]),float(entry[6]),float(entry[7]),float(entry[8]),float(entry[9]),float(entry[10]),float(entry[11]),float(entry[12]),float(entry[13]),float(entry[14]),float(entry[15]))
      relax['R1rho'][entry[2]] = value
    if entry[0] == 'R2rho':
      value = (float(entry[3]),float(entry[4]),float(entry[5]),float(entry[6]),float(entry[7]),float(entry[8]),float(entry[9]),float(entry[10]),float(entry[11]),float(entry[12]),float(entry[13]),float(entry[14]),float(entry[15]))
      relax['R2rho'][entry[2]] = value

  return relax, code


#------------------------------------------------------------------------------
# read the relaxation library
#

def read_library(file,offset1):

  relax = {}
  offset_set = []
  offset1 = offset1*4
  offset_int1 = int(offset1)
  if abs(offset1-offset_int1) > 0.5:
    if offset1 > 0:
      offset_int1 = offset_int1 + 1
    else:
      offset_int1 = offset_int1 - 1

  if offset_int1 in (-20,20):
    if offset_int1 > 0:
      offset_set.append(offset_int1)
      offset_set.append(offset_int1-1)
      offset_set.append(offset_int1-2)
    else:
      offset_set.append(offset_int1)
      offset_set.append(offset_int1+1)
      offset_set.append(offset_int1+2)
  else:
    offset_set.append(offset_int1)
    offset_set.append(offset_int1+1)
    offset_set.append(offset_int1-1)
  f = open(file,'r')
  lines = f.readlines()
  f.close()
  for line in lines:
    entry = line.split()
    if int(entry[0]) not in offset_set:
      continue
    id = (int(entry[0]),int(entry[1]),int(entry[2]),int(entry[3]))
    value = (float(entry[4]),float(entry[5]),float(entry[6]),float(entry[7]))
    relax[id] = value

  return relax


def read_J(file,offset1):

  relax = {}
  offset_set = []
  offset1 = offset1*4
  offset_int1 = int(offset1)
  if abs(offset1-offset_int1) > 0.5:
    if offset1 > 0:
      offset_int1 = offset_int1 + 1
    else:
      offset_int1 = offset_int1 - 1

  if offset_int1 in (-20,20):
    if offset_int1 > 0:
      offset_set.append(offset_int1)
      offset_set.append(offset_int1-1)
      offset_set.append(offset_int1-2)
    else:
      offset_set.append(offset_int1)
      offset_set.append(offset_int1+1)
      offset_set.append(offset_int1+2)
  else:
    offset_set.append(offset_int1)
    offset_set.append(offset_int1+1)
    offset_set.append(offset_int1-1)
  f = open(file,'r')
  lines = f.readlines()
  f.close()
  for line in lines:
    entry = line.split()
    if int(entry[0]) not in offset_set:
      continue
    id = (int(entry[0]),int(entry[1]),int(entry[2]),int(entry[3]))
    value = (float(entry[4]),float(entry[5]),float(entry[6]))
    relax[id] = value

  return relax


#------------------------------------------------------------------------------
# calculate relaxation difference

def cal_diff(peak1,peak2):

  if abs(peak1[0]-peak2[0]) < peak1[4]:
    d0 = 0
  else:
    d0 = abs(peak1[0]-peak2[0])-peak1[4]
  if abs(peak1[1]-peak2[1]) < peak1[5]:
    d1 = 0
  else:
    d1 = abs(peak1[1]-peak2[1])-peak1[5]
  if abs(peak1[2]-peak2[2]) < peak1[6]:
    d2 = 0
  else:
    d2 = abs(peak1[2]-peak2[2])-peak1[6]
  if abs(peak1[3]-peak2[3]) < peak1[7]:
    d3 = 0
  else:
    d3 = abs(peak1[3]-peak2[3])-peak1[7]

  dist = 0.25*((d0/float(peak1[0]))**2+(d1/float(peak1[1]))**2+(d2/float(peak1[2]))**2+(d3/float(peak1[3]))**2)

  return dist


def X_diff(peak1,peak2):

  if abs(peak1[0]-peak2[0]) < peak1[4]:
    d0 = 0
  else:
    d0 = abs(peak1[0]-peak2[0])-peak1[4]
  if abs(peak1[1]-peak2[1]) < peak1[5]:
    d1 = 0
  else:
    d1 = abs(peak1[1]-peak2[1])-peak1[5]
  if abs(peak1[2]-peak2[2]) < peak1[6]:
    d2 = 0
  else:
    d2 = abs(peak1[2]-peak2[2])-peak1[6]
  if abs(peak1[3]-peak2[3]) < peak1[7]:
    d3 = 0
  else:
    d3 = abs(peak1[3]-peak2[3])-peak1[7]

  dist = 0.25*((d0/float(peak1[4]))**2+(d1/float(peak1[5]))**2+(d2/float(peak1[6]))**2+(d3/float(peak1[7]))**2)

  return dist


#------------------------------------------------------------------------
# polynomial approximation

def poly_relax_vert(id,id_int,d1,d2,lib0,lib1,lib2):

  const1 = 2*math.pi
  u1 = id[0]-id_int[0]
  u2 = id[1]-id_int[1]
  u3 = id[2]-id_int[2]
  u4 = id[3]-id_int[3]
  a0,a1,a2,a3 = 1,1,1,1
  if u1 < 0:
    a0 = -1
    u1 = -u1
  if u2 < 0:
    a1 = -1
    u2 = -u2
  if u3 < 0:
    a2 = -1
    u3 = -u3
  if u4 < 0 and id_int[3] != 0:
    a3 = -1
    u4 = -u4
  if id_int[0] in (-20,20):
    if id_int[0] > 0:
      a0 = -1
    else:
      a0 = 1
  if id_int[1] in (0,80):
    if id_int[1] > 0:
      a1 = -1
    else:
      a1 = 1
  if id_int[2] in (0,40):
    if id_int[2] > 0:
      a2 = -1
    else:
      a2 = 1
  if id_int[3] in (0,9):
    if id_int[3] > 0:
      a3 = -1
    else:
      a3 = 1

  if id_int[0] in (-20,20):
    relax_p1 = lib0[(id_int[0]+2*a0,id_int[1],id_int[2],id_int[3])]
    relax_n1 = lib0[(id_int[0]+a0,id_int[1],id_int[2],id_int[3])]
    add_11 = 0.5*(relax_p1[0]-2*relax_n1[0]+lib0[id_int][0])*(u1**2) - 0.5*(relax_p1[0]-4*relax_n1[0]+3*lib0[id_int][0])*u1
    add_21 = 0.5*(relax_p1[1]-2*relax_n1[1]+lib0[id_int][1])*(u1**2) - 0.5*(relax_p1[1]-4*relax_n1[1]+3*lib0[id_int][1])*u1
    add_41 = 0.5*(relax_p1[2]-2*relax_n1[2]+lib0[id_int][2])*(u1**2) - 0.5*(relax_p1[2]-4*relax_n1[2]+3*lib0[id_int][2])*u1
    add_61 = 0.5*(relax_p1[3]-2*relax_n1[3]+lib0[id_int][3])*(u1**2) - 0.5*(relax_p1[3]-4*relax_n1[3]+3*lib0[id_int][3])*u1
  else:
    relax_p1 = lib0[(id_int[0]+a0,id_int[1],id_int[2],id_int[3])]
    relax_n1 = lib0[(id_int[0]-a0,id_int[1],id_int[2],id_int[3])]
    add_11 = 0.5*(relax_p1[0]+relax_n1[0]-2*lib0[id_int][0])*(u1**2) + 0.5*(relax_p1[0]-relax_n1[0])*u1
    add_21 = 0.5*(relax_p1[1]+relax_n1[1]-2*lib0[id_int][1])*(u1**2) + 0.5*(relax_p1[1]-relax_n1[1])*u1
    add_41 = 0.5*(relax_p1[2]+relax_n1[2]-2*lib0[id_int][2])*(u1**2) + 0.5*(relax_p1[2]-relax_n1[2])*u1
    add_61 = 0.5*(relax_p1[3]+relax_n1[3]-2*lib0[id_int][3])*(u1**2) + 0.5*(relax_p1[3]-relax_n1[3])*u1
  if id_int[1] in (0,80):
    relax_p2 = lib0[(id_int[0],id_int[1]+2*a1,id_int[2],id_int[3])]
    relax_n2 = lib0[(id_int[0],id_int[1]+a1,id_int[2],id_int[3])]
    add_12 = 0.5*(relax_p2[0]-2*relax_n2[0]+lib0[id_int][0])*(u2**2) - 0.5*(relax_p2[0]-4*relax_n2[0]+3*lib0[id_int][0])*u2
    add_22 = 0.5*(relax_p2[1]-2*relax_n2[1]+lib0[id_int][1])*(u2**2) - 0.5*(relax_p2[1]-4*relax_n2[1]+3*lib0[id_int][1])*u2
    add_42 = 0.5*(relax_p2[2]-2*relax_n2[2]+lib0[id_int][2])*(u2**2) - 0.5*(relax_p2[2]-4*relax_n2[2]+3*lib0[id_int][2])*u2
    add_62 = 0.5*(relax_p2[3]-2*relax_n2[3]+lib0[id_int][3])*(u2**2) - 0.5*(relax_p2[3]-4*relax_n2[3]+3*lib0[id_int][3])*u2
  else:
    relax_p2 = lib0[(id_int[0],id_int[1]+a1,id_int[2],id_int[3])]
    relax_n2 = lib0[(id_int[0],id_int[1]-a1,id_int[2],id_int[3])]
    add_12 = 0.5*(relax_p2[0]+relax_n2[0]-2*lib0[id_int][0])*(u2**2) + 0.5*(relax_p2[0]-relax_n2[0])*u2
    add_22 = 0.5*(relax_p2[1]+relax_n2[1]-2*lib0[id_int][1])*(u2**2) + 0.5*(relax_p2[1]-relax_n2[1])*u2
    add_42 = 0.5*(relax_p2[2]+relax_n2[2]-2*lib0[id_int][2])*(u2**2) + 0.5*(relax_p2[2]-relax_n2[2])*u2
    add_62 = 0.5*(relax_p2[3]+relax_n2[3]-2*lib0[id_int][3])*(u2**2) + 0.5*(relax_p2[3]-relax_n2[3])*u2
  if id_int[2] in (0,40):
    relax_p3 = lib0[(id_int[0],id_int[1],id_int[2]+2*a2,id_int[3])]
    relax_n3 = lib0[(id_int[0],id_int[1],id_int[2]+a2,id_int[3])]
    add_13 = 0.5*(relax_p3[0]-2*relax_n3[0]+lib0[id_int][0])*(u3**2) - 0.5*(relax_p3[0]-4*relax_n3[0]+3*lib0[id_int][0])*u3
    add_23 = 0.5*(relax_p3[1]-2*relax_n3[1]+lib0[id_int][1])*(u3**2) - 0.5*(relax_p3[1]-4*relax_n3[1]+3*lib0[id_int][1])*u3
    add_43 = 0.5*(relax_p3[2]-2*relax_n3[2]+lib0[id_int][2])*(u3**2) - 0.5*(relax_p3[2]-4*relax_n3[2]+3*lib0[id_int][2])*u3
    add_63 = 0.5*(relax_p3[3]-2*relax_n3[3]+lib0[id_int][3])*(u3**2) - 0.5*(relax_p3[3]-4*relax_n3[3]+3*lib0[id_int][3])*u3
  else:
    relax_p3 = lib0[(id_int[0],id_int[1],id_int[2]+a2,id_int[3])]
    relax_n3 = lib0[(id_int[0],id_int[1],id_int[2]-a2,id_int[3])]
    add_13 = 0.5*(relax_p3[0]+relax_n3[0]-2*lib0[id_int][0])*(u3**2) + 0.5*(relax_p3[0]-relax_n3[0])*u3
    add_23 = 0.5*(relax_p3[1]+relax_n3[1]-2*lib0[id_int][1])*(u3**2) + 0.5*(relax_p3[1]-relax_n3[1])*u3
    add_43 = 0.5*(relax_p3[2]+relax_n3[2]-2*lib0[id_int][2])*(u3**2) + 0.5*(relax_p3[2]-relax_n3[2])*u3
    add_63 = 0.5*(relax_p3[3]+relax_n3[3]-2*lib0[id_int][3])*(u3**2) + 0.5*(relax_p3[3]-relax_n3[3])*u3
  if id_int[3] in (0,9):
    relax_p4 = lib0[(id_int[0],id_int[1],id_int[2],id_int[3]+2*a3)]
    relax_n4 = lib0[(id_int[0],id_int[1],id_int[2],id_int[3]+a3)]
    add_14 = 0.5*(relax_p4[0]-2*relax_n4[0]+lib0[id_int][0])*(u4**2) - 0.5*(relax_p4[0]-4*relax_n4[0]+3*lib0[id_int][0])*u4
    add_24 = 0.5*(relax_p4[1]-2*relax_n4[1]+lib0[id_int][1])*(u4**2) - 0.5*(relax_p4[1]-4*relax_n4[1]+3*lib0[id_int][1])*u4
    add_44 = 0.5*(relax_p4[2]-2*relax_n4[2]+lib0[id_int][2])*(u4**2) - 0.5*(relax_p4[2]-4*relax_n4[2]+3*lib0[id_int][2])*u4
    add_64 = 0.5*(relax_p4[3]-2*relax_n4[3]+lib0[id_int][3])*(u4**2) - 0.5*(relax_p4[3]-4*relax_n4[3]+3*lib0[id_int][3])*u4
  else:
    relax_p4 = lib0[(id_int[0],id_int[1],id_int[2],id_int[3]+a3)]
    relax_n4 = lib0[(id_int[0],id_int[1],id_int[2],id_int[3]-a3)]
    add_14 = 0.5*(relax_p4[0]+relax_n4[0]-2*lib0[id_int][0])*(u4**2) + 0.5*(relax_p4[0]-relax_n4[0])*u4
    add_24 = 0.5*(relax_p4[1]+relax_n4[1]-2*lib0[id_int][1])*(u4**2) + 0.5*(relax_p4[1]-relax_n4[1])*u4
    add_44 = 0.5*(relax_p4[2]+relax_n4[2]-2*lib0[id_int][2])*(u4**2) + 0.5*(relax_p4[2]-relax_n4[2])*u4
    add_64 = 0.5*(relax_p4[3]+relax_n4[3]-2*lib0[id_int][3])*(u4**2) + 0.5*(relax_p4[3]-relax_n4[3])*u4
  relax_p1 = lib0[(id_int[0]+a0,id_int[1],id_int[2],id_int[3])]
  relax_p2 = lib0[(id_int[0],id_int[1]+a1,id_int[2],id_int[3])]
  relax_p3 = lib0[(id_int[0],id_int[1],id_int[2]+a2,id_int[3])]
  relax_p4 = lib0[(id_int[0],id_int[1],id_int[2],id_int[3]+a3)]
  relax_12 = lib0[(id_int[0]+a0,id_int[1]+a1,id_int[2],id_int[3])]
  relax_13 = lib0[(id_int[0]+a0,id_int[1],id_int[2]+a2,id_int[3])]
  relax_14 = lib0[(id_int[0]+a0,id_int[1],id_int[2],id_int[3]+a3)]
  relax_23 = lib0[(id_int[0],id_int[1]+a1,id_int[2]+a2,id_int[3])]
  relax_24 = lib0[(id_int[0],id_int[1]+a1,id_int[2],id_int[3]+a3)]
  relax_34 = lib0[(id_int[0],id_int[1],id_int[2]+a2,id_int[3]+a3)]

  HS1 = lib0[id_int][0] + add_11 + add_12 + add_13 + add_14 + (relax_12[0]-relax_p1[0]-relax_p2[0]+lib0[id_int][0])*u1*u2 + (relax_13[0]-relax_p1[0]-relax_p3[0]+lib0[id_int][0])*u1*u3 + (relax_14[0]-relax_p1[0]-relax_p4[0]+lib0[id_int][0])*u1*u4 + (relax_23[0]-relax_p2[0]-relax_p3[0]+lib0[id_int][0])*u2*u3 + (relax_24[0]-relax_p2[0]-relax_p4[0]+lib0[id_int][0])*u2*u4 + (relax_34[0]-relax_p3[0]-relax_p4[0]+lib0[id_int][0])*u3*u4 + (d1/float(0.005))*(lib1[id_int][0]-lib0[id_int][0]) + (d2/float(0.05))*(lib2[id_int][0]-lib0[id_int][0])

  HS2 = lib0[id_int][1] + add_21 + add_22 + add_23 + add_24 + (relax_12[1]-relax_p1[1]-relax_p2[1]+lib0[id_int][1])*u1*u2 + (relax_13[1]-relax_p1[1]-relax_p3[1]+lib0[id_int][1])*u1*u3 + (relax_14[1]-relax_p1[1]-relax_p4[1]+lib0[id_int][1])*u1*u4 + (relax_23[1]-relax_p2[1]-relax_p3[1]+lib0[id_int][1])*u2*u3 + (relax_24[1]-relax_p2[1]-relax_p4[1]+lib0[id_int][1])*u2*u4 + (relax_34[1]-relax_p3[1]-relax_p4[1]+lib0[id_int][1])*u3*u4 + (d1/float(0.005))*(lib1[id_int][1]-lib0[id_int][1]) + (d2/float(0.05))*(lib2[id_int][1]-lib0[id_int][1])

  HS4 = lib0[id_int][2] + add_41 + add_42 + add_43 + add_44 + (relax_12[2]-relax_p1[2]-relax_p2[2]+lib0[id_int][2])*u1*u2 + (relax_13[2]-relax_p1[2]-relax_p3[2]+lib0[id_int][2])*u1*u3 + (relax_14[2]-relax_p1[2]-relax_p4[2]+lib0[id_int][2])*u1*u4 + (relax_23[2]-relax_p2[2]-relax_p3[2]+lib0[id_int][2])*u2*u3 + (relax_24[2]-relax_p2[2]-relax_p4[2]+lib0[id_int][2])*u2*u4 + (relax_34[2]-relax_p3[2]-relax_p4[2]+lib0[id_int][2])*u3*u4 + (d1/float(0.005))*(lib1[id_int][2]-lib0[id_int][2]) + (d2/float(0.05))*(lib2[id_int][2]-lib0[id_int][2])

  HS8 = lib0[id_int][3] + add_61 + add_62 + add_63 + add_64 + (relax_12[3]-relax_p1[3]-relax_p2[3]+lib0[id_int][3])*u1*u2 + (relax_13[3]-relax_p1[3]-relax_p3[3]+lib0[id_int][3])*u1*u3 + (relax_14[3]-relax_p1[3]-relax_p4[3]+lib0[id_int][3])*u1*u4 + (relax_23[3]-relax_p2[3]-relax_p3[3]+lib0[id_int][3])*u2*u3 + (relax_24[3]-relax_p2[3]-relax_p4[3]+lib0[id_int][3])*u2*u4 + (relax_34[3]-relax_p3[3]-relax_p4[3]+lib0[id_int][3])*u3*u4 + (d1/float(0.005))*(lib1[id_int][3]-lib0[id_int][3]) + (d2/float(0.05))*(lib2[id_int][3]-lib0[id_int][3])

  return (HS1,HS2,HS4,HS8)



def poly_relax_trans(id,id_int,d1,d2,d3,lib0,lib1,lib2,lib3):

  const1 = 2*math.pi
  u1 = id[0]-id_int[0]
  u2 = id[1]-id_int[1]
  u3 = id[2]-id_int[2]
  u4 = id[3]-id_int[3]
  a0,a1,a2,a3 = 1,1,1,1
  if u1 < 0:
    a0 = -1
    u1 = -u1
  if u2 < 0:
    a1 = -1
    u2 = -u2
  if u3 < 0:
    a2 = -1
    u3 = -u3
  if u4 < 0 and id_int[3] != 0:
    a3 = -1
    u4 = -u4
  if id_int[0] in (-20,20):
    if id_int[0] > 0:
      a0 = -1
    else:
      a0 = 1
  if id_int[1] in (0,80):
    if id_int[1] > 0:
      a1 = -1
    else:
      a1 = 1
  if id_int[2] in (0,40):
    if id_int[2] > 0:
      a2 = -1
    else:
      a2 = 1
  if id_int[3] in (0,9):
    if id_int[3] > 0:
      a3 = -1
    else:
      a3 = 1

  if id_int[0] in (-20,20):
    relax_p1 = lib0[(id_int[0]+2*a0,id_int[1],id_int[2],id_int[3])]
    relax_n1 = lib0[(id_int[0]+a0,id_int[1],id_int[2],id_int[3])]
    add_11 = 0.5*(relax_p1[0]-2*relax_n1[0]+lib0[id_int][0])*(u1**2) - 0.5*(relax_p1[0]-4*relax_n1[0]+3*lib0[id_int][0])*u1
    add_21 = 0.5*(relax_p1[1]-2*relax_n1[1]+lib0[id_int][1])*(u1**2) - 0.5*(relax_p1[1]-4*relax_n1[1]+3*lib0[id_int][1])*u1
    add_41 = 0.5*(relax_p1[2]-2*relax_n1[2]+lib0[id_int][2])*(u1**2) - 0.5*(relax_p1[2]-4*relax_n1[2]+3*lib0[id_int][2])*u1
    add_61 = 0.5*(relax_p1[3]-2*relax_n1[3]+lib0[id_int][3])*(u1**2) - 0.5*(relax_p1[3]-4*relax_n1[3]+3*lib0[id_int][3])*u1
  else:
    relax_p1 = lib0[(id_int[0]+a0,id_int[1],id_int[2],id_int[3])]
    relax_n1 = lib0[(id_int[0]-a0,id_int[1],id_int[2],id_int[3])]
    add_11 = 0.5*(relax_p1[0]+relax_n1[0]-2*lib0[id_int][0])*(u1**2) + 0.5*(relax_p1[0]-relax_n1[0])*u1
    add_21 = 0.5*(relax_p1[1]+relax_n1[1]-2*lib0[id_int][1])*(u1**2) + 0.5*(relax_p1[1]-relax_n1[1])*u1
    add_41 = 0.5*(relax_p1[2]+relax_n1[2]-2*lib0[id_int][2])*(u1**2) + 0.5*(relax_p1[2]-relax_n1[2])*u1
    add_61 = 0.5*(relax_p1[3]+relax_n1[3]-2*lib0[id_int][3])*(u1**2) + 0.5*(relax_p1[3]-relax_n1[3])*u1
  if id_int[1] in (0,80):
    relax_p2 = lib0[(id_int[0],id_int[1]+2*a1,id_int[2],id_int[3])]
    relax_n2 = lib0[(id_int[0],id_int[1]+a1,id_int[2],id_int[3])]
    add_12 = 0.5*(relax_p2[0]-2*relax_n2[0]+lib0[id_int][0])*(u2**2) - 0.5*(relax_p2[0]-4*relax_n2[0]+3*lib0[id_int][0])*u2
    add_22 = 0.5*(relax_p2[1]-2*relax_n2[1]+lib0[id_int][1])*(u2**2) - 0.5*(relax_p2[1]-4*relax_n2[1]+3*lib0[id_int][1])*u2
    add_42 = 0.5*(relax_p2[2]-2*relax_n2[2]+lib0[id_int][2])*(u2**2) - 0.5*(relax_p2[2]-4*relax_n2[2]+3*lib0[id_int][2])*u2
    add_62 = 0.5*(relax_p2[3]-2*relax_n2[3]+lib0[id_int][3])*(u2**2) - 0.5*(relax_p2[3]-4*relax_n2[3]+3*lib0[id_int][3])*u2
  else:
    relax_p2 = lib0[(id_int[0],id_int[1]+a1,id_int[2],id_int[3])]
    relax_n2 = lib0[(id_int[0],id_int[1]-a1,id_int[2],id_int[3])]
    add_12 = 0.5*(relax_p2[0]+relax_n2[0]-2*lib0[id_int][0])*(u2**2) + 0.5*(relax_p2[0]-relax_n2[0])*u2
    add_22 = 0.5*(relax_p2[1]+relax_n2[1]-2*lib0[id_int][1])*(u2**2) + 0.5*(relax_p2[1]-relax_n2[1])*u2
    add_42 = 0.5*(relax_p2[2]+relax_n2[2]-2*lib0[id_int][2])*(u2**2) + 0.5*(relax_p2[2]-relax_n2[2])*u2
    add_62 = 0.5*(relax_p2[3]+relax_n2[3]-2*lib0[id_int][3])*(u2**2) + 0.5*(relax_p2[3]-relax_n2[3])*u2
  if id_int[2] in (0,40):
    relax_p3 = lib0[(id_int[0],id_int[1],id_int[2]+2*a2,id_int[3])]
    relax_n3 = lib0[(id_int[0],id_int[1],id_int[2]+a2,id_int[3])]
    add_13 = 0.5*(relax_p3[0]-2*relax_n3[0]+lib0[id_int][0])*(u3**2) - 0.5*(relax_p3[0]-4*relax_n3[0]+3*lib0[id_int][0])*u3
    add_23 = 0.5*(relax_p3[1]-2*relax_n3[1]+lib0[id_int][1])*(u3**2) - 0.5*(relax_p3[1]-4*relax_n3[1]+3*lib0[id_int][1])*u3
    add_43 = 0.5*(relax_p3[2]-2*relax_n3[2]+lib0[id_int][2])*(u3**2) - 0.5*(relax_p3[2]-4*relax_n3[2]+3*lib0[id_int][2])*u3
    add_63 = 0.5*(relax_p3[3]-2*relax_n3[3]+lib0[id_int][3])*(u3**2) - 0.5*(relax_p3[3]-4*relax_n3[3]+3*lib0[id_int][3])*u3
  else:
    relax_p3 = lib0[(id_int[0],id_int[1],id_int[2]+a2,id_int[3])]
    relax_n3 = lib0[(id_int[0],id_int[1],id_int[2]-a2,id_int[3])]
    add_13 = 0.5*(relax_p3[0]+relax_n3[0]-2*lib0[id_int][0])*(u3**2) + 0.5*(relax_p3[0]-relax_n3[0])*u3
    add_23 = 0.5*(relax_p3[1]+relax_n3[1]-2*lib0[id_int][1])*(u3**2) + 0.5*(relax_p3[1]-relax_n3[1])*u3
    add_43 = 0.5*(relax_p3[2]+relax_n3[2]-2*lib0[id_int][2])*(u3**2) + 0.5*(relax_p3[2]-relax_n3[2])*u3
    add_63 = 0.5*(relax_p3[3]+relax_n3[3]-2*lib0[id_int][3])*(u3**2) + 0.5*(relax_p3[3]-relax_n3[3])*u3
  if id_int[3] in (0,9):
    relax_p4 = lib0[(id_int[0],id_int[1],id_int[2],id_int[3]+2*a3)]
    relax_n4 = lib0[(id_int[0],id_int[1],id_int[2],id_int[3]+a3)]
    add_14 = 0.5*(relax_p4[0]-2*relax_n4[0]+lib0[id_int][0])*(u4**2) - 0.5*(relax_p4[0]-4*relax_n4[0]+3*lib0[id_int][0])*u4
    add_24 = 0.5*(relax_p4[1]-2*relax_n4[1]+lib0[id_int][1])*(u4**2) - 0.5*(relax_p4[1]-4*relax_n4[1]+3*lib0[id_int][1])*u4
    add_44 = 0.5*(relax_p4[2]-2*relax_n4[2]+lib0[id_int][2])*(u4**2) - 0.5*(relax_p4[2]-4*relax_n4[2]+3*lib0[id_int][2])*u4
    add_64 = 0.5*(relax_p4[3]-2*relax_n4[3]+lib0[id_int][3])*(u4**2) - 0.5*(relax_p4[3]-4*relax_n4[3]+3*lib0[id_int][3])*u4
  else:
    relax_p4 = lib0[(id_int[0],id_int[1],id_int[2],id_int[3]+a3)]
    relax_n4 = lib0[(id_int[0],id_int[1],id_int[2],id_int[3]-a3)]
    add_14 = 0.5*(relax_p4[0]+relax_n4[0]-2*lib0[id_int][0])*(u4**2) + 0.5*(relax_p4[0]-relax_n4[0])*u4
    add_24 = 0.5*(relax_p4[1]+relax_n4[1]-2*lib0[id_int][1])*(u4**2) + 0.5*(relax_p4[1]-relax_n4[1])*u4
    add_44 = 0.5*(relax_p4[2]+relax_n4[2]-2*lib0[id_int][2])*(u4**2) + 0.5*(relax_p4[2]-relax_n4[2])*u4
    add_64 = 0.5*(relax_p4[3]+relax_n4[3]-2*lib0[id_int][3])*(u4**2) + 0.5*(relax_p4[3]-relax_n4[3])*u4
  relax_p1 = lib0[(id_int[0]+a0,id_int[1],id_int[2],id_int[3])]
  relax_p2 = lib0[(id_int[0],id_int[1]+a1,id_int[2],id_int[3])]
  relax_p3 = lib0[(id_int[0],id_int[1],id_int[2]+a2,id_int[3])]
  relax_p4 = lib0[(id_int[0],id_int[1],id_int[2],id_int[3]+a3)]
  relax_12 = lib0[(id_int[0]+a0,id_int[1]+a1,id_int[2],id_int[3])]
  relax_13 = lib0[(id_int[0]+a0,id_int[1],id_int[2]+a2,id_int[3])]
  relax_14 = lib0[(id_int[0]+a0,id_int[1],id_int[2],id_int[3]+a3)]
  relax_23 = lib0[(id_int[0],id_int[1]+a1,id_int[2]+a2,id_int[3])]
  relax_24 = lib0[(id_int[0],id_int[1]+a1,id_int[2],id_int[3]+a3)]
  relax_34 = lib0[(id_int[0],id_int[1],id_int[2]+a2,id_int[3]+a3)]

  HS1 = lib0[id_int][0] + add_11 + add_12 + add_13 + add_14 + (relax_12[0]-relax_p1[0]-relax_p2[0]+lib0[id_int][0])*u1*u2 + (relax_13[0]-relax_p1[0]-relax_p3[0]+lib0[id_int][0])*u1*u3 + (relax_14[0]-relax_p1[0]-relax_p4[0]+lib0[id_int][0])*u1*u4 + (relax_23[0]-relax_p2[0]-relax_p3[0]+lib0[id_int][0])*u2*u3 + (relax_24[0]-relax_p2[0]-relax_p4[0]+lib0[id_int][0])*u2*u4 + (relax_34[0]-relax_p3[0]-relax_p4[0]+lib0[id_int][0])*u3*u4 + (d1/float(0.005))*(lib1[id_int][0]-lib0[id_int][0]) + (d2/float(0.05))*(lib2[id_int][0]-lib0[id_int][0]) + (d3/float(0.01))*(lib3[id_int][0]-lib0[id_int][0])

  HS2 = lib0[id_int][1] + add_21 + add_22 + add_23 + add_24 + (relax_12[1]-relax_p1[1]-relax_p2[1]+lib0[id_int][1])*u1*u2 + (relax_13[1]-relax_p1[1]-relax_p3[1]+lib0[id_int][1])*u1*u3 + (relax_14[1]-relax_p1[1]-relax_p4[1]+lib0[id_int][1])*u1*u4 + (relax_23[1]-relax_p2[1]-relax_p3[1]+lib0[id_int][1])*u2*u3 + (relax_24[1]-relax_p2[1]-relax_p4[1]+lib0[id_int][1])*u2*u4 + (relax_34[1]-relax_p3[1]-relax_p4[1]+lib0[id_int][1])*u3*u4 + (d1/float(0.005))*(lib1[id_int][1]-lib0[id_int][1]) + (d2/float(0.05))*(lib2[id_int][1]-lib0[id_int][1]) + (d3/float(0.01))*(lib3[id_int][1]-lib0[id_int][1])

  HS4 = lib0[id_int][2] + add_41 + add_42 + add_43 + add_44 + (relax_12[2]-relax_p1[2]-relax_p2[2]+lib0[id_int][2])*u1*u2 + (relax_13[2]-relax_p1[2]-relax_p3[2]+lib0[id_int][2])*u1*u3 + (relax_14[2]-relax_p1[2]-relax_p4[2]+lib0[id_int][2])*u1*u4 + (relax_23[2]-relax_p2[2]-relax_p3[2]+lib0[id_int][2])*u2*u3 + (relax_24[2]-relax_p2[2]-relax_p4[2]+lib0[id_int][2])*u2*u4 + (relax_34[2]-relax_p3[2]-relax_p4[2]+lib0[id_int][2])*u3*u4 + (d1/float(0.005))*(lib1[id_int][2]-lib0[id_int][2]) + (d2/float(0.05))*(lib2[id_int][2]-lib0[id_int][2]) + (d3/float(0.01))*(lib3[id_int][2]-lib0[id_int][2])

  HS8 = lib0[id_int][3] + add_61 + add_62 + add_63 + add_64 + (relax_12[3]-relax_p1[3]-relax_p2[3]+lib0[id_int][3])*u1*u2 + (relax_13[3]-relax_p1[3]-relax_p3[3]+lib0[id_int][3])*u1*u3 + (relax_14[3]-relax_p1[3]-relax_p4[3]+lib0[id_int][3])*u1*u4 + (relax_23[3]-relax_p2[3]-relax_p3[3]+lib0[id_int][3])*u2*u3 + (relax_24[3]-relax_p2[3]-relax_p4[3]+lib0[id_int][3])*u2*u4 + (relax_34[3]-relax_p3[3]-relax_p4[3]+lib0[id_int][3])*u3*u4 + (d1/float(0.005))*(lib1[id_int][3]-lib0[id_int][3]) + (d2/float(0.05))*(lib2[id_int][3]-lib0[id_int][3]) + (d3/float(0.01))*0.344990445496

  return (HS1,HS2,HS4,HS8)



#------------------------------------------------------------------------
# library search

def linear_relax_vert(id,d1,d2,lib0,lib1,lib2):

  const1 = 2*math.pi
  HS1 = lib0[id][0] + (d1/float(0.005))*(lib1[id][0]-lib0[id][0]) + (d2/float(0.05))*(lib2[id][0]-lib0[id][0])
  HS2 = lib0[id][1] + (d1/float(0.005))*(lib1[id][1]-lib0[id][1]) + (d2/float(0.05))*(lib2[id][1]-lib0[id][1])
  HS4 = lib0[id][2] + (d1/float(0.005))*(lib1[id][2]-lib0[id][2]) + (d2/float(0.05))*(lib2[id][2]-lib0[id][2])
  HS8 = lib0[id][3] + (d1/float(0.005))*(lib1[id][3]-lib0[id][3]) + (d2/float(0.05))*(lib2[id][3]-lib0[id][3])

  return (HS1,HS2,HS4,HS8)


def linear_relax_trans(id,d1,d2,d3,lib0,lib1,lib2,lib3):

  const1 = 2*math.pi
  HS1 = lib0[id][0] + (d1/float(0.005))*(lib1[id][0]-lib0[id][0]) + (d2/float(0.05))*(lib2[id][0]-lib0[id][0]) + (d3/float(0.01))*(lib3[id][0]-lib0[id][0])
  HS2 = lib0[id][1] + (d1/float(0.005))*(lib1[id][1]-lib0[id][1]) + (d2/float(0.05))*(lib2[id][1]-lib0[id][1]) + (d3/float(0.01))*(lib3[id][1]-lib0[id][1])
  HS4 = lib0[id][2] + (d1/float(0.005))*(lib1[id][2]-lib0[id][2]) + (d2/float(0.05))*(lib2[id][2]-lib0[id][2]) + (d3/float(0.01))*(lib3[id][2]-lib0[id][2])
  HS8 = lib0[id][3] + (d1/float(0.005))*(lib1[id][3]-lib0[id][3]) + (d2/float(0.05))*(lib2[id][3]-lib0[id][3]) + (d3/float(0.01))*0.344990445496

  return (HS1,HS2,HS4,HS8)



#------------------------------------------------------------------------
#

def func(d1,d2):

  if d1 == 0:
    add = (d2,0,0,0,0,0)
  if d1 == 1:
    add = (0,d2,0,0,0,0)
  if d1 == 2:
    add = (0,0,d2,0,0,0)
  if d1 == 3:
    add = (0,0,0,d2,0,0)
  if d1 == 4:
    add = (0,0,0,0,d2,0)
  if d1 == 5:
    add = (0,0,0,0,0,d2)

  return add


#------------------------------------------------------------------------------
# calculate the perturbation

def cal_perturb(za,ya,xa,zb,yb,xb,incre,Wa,Wb,ka,kb,R1a,R1b,R2a,R2b):
  const1 = 2*math.pi
  inter_za = -(ka*const1+R1a)*incre*za +kb*incre*const1*zb
  inter_ya = Wa*incre*const1*xa -(ka*const1+R2a)*incre*ya +kb*incre*const1*yb
  inter_xa = -(ka*const1+R2a)*incre*xa +kb*incre*const1*xb -Wa*incre*const1*ya
  inter_zb = ka*incre*const1*za -(kb*const1+R1b)*incre*zb
  inter_yb = Wb*incre*const1*xb +ka*incre*const1*ya -(kb*const1+R2b)*incre*yb
  inter_xb = ka*incre*const1*xa -(kb*const1+R2b)*incre*xb -Wb*incre*const1*yb
  return inter_za, inter_ya, inter_xa, inter_zb, inter_yb, inter_xb


#------------------------------------------------------------------------------
# calculate the bulk magnetization (using Bloch-McConnell equation with second order perturbation)

def cal_relax(a,b,c,d,e,f,t,m):

  const1 = 2*math.pi
  n1 = 1000
  n3 = 10
  Wa = a   # offset frequency of state A (kHz)
  Wb = b   # offset frequency of state B (kHz)
  ka = c   # forward reaction rate A->B (kHz)
  kb = d   # reverse reaction rate B->A (kHz)
  kex = ka + kb   # exchange rate (kHz)
  R1a = e   # R1 for state A (ks-1)
  R1b = e   # R1 for state B (ks-1)
  R2a = f   # R2 for state A (ks-1)
  R2b = f   # R2 for state B (ks-1)
  Tot = int(t*n1*n3)
  incre = 1/float(n1*n3)
  for n in range(m):
    coor_za = {}
    coor_ya = {}
    coor_xa = {}
    coor_zb = {}
    coor_yb = {}
    coor_xb = {}
    for i in range(Tot+1):
      coor_za[i] = []
      coor_ya[i] = []
      coor_xa[i] = []
      coor_zb[i] = []
      coor_yb[i] = []
      coor_xb[i] = []
      if i == 0:
        if n == 0:
          coor_za[0] = 0  # initial point
          coor_ya[0] = kb/float(kex)  # initial point
          coor_xa[0] = 0  # initial point
          coor_zb[0] = 0  # initial point
          coor_yb[0] = ka/float(kex)  # initial point
          coor_xb[0] = 0  # initial point
        else:
          coor_za[0] = coor_za_fa
          coor_ya[0] = coor_ya_fa
          coor_xa[0] = coor_xa_fa
          coor_zb[0] = coor_zb_fa
          coor_yb[0] = coor_yb_fa
          coor_xb[0] = coor_xb_fa
      else:
        za = coor_za[i-1]
        ya = coor_ya[i-1]
        xa = coor_xa[i-1]
        zb = coor_zb[i-1]
        yb = coor_yb[i-1]
        xb = coor_xb[i-1]
        inter_za,inter_ya,inter_xa,inter_zb,inter_yb,inter_xb = cal_perturb(za,ya,xa,zb,yb,xb,incre,Wa,Wb,ka,kb,R1a,R1b,R2a,R2b)
        inter_za1,inter_ya1,inter_xa1,inter_zb1,inter_yb1,inter_xb1 = cal_perturb(inter_za,inter_ya,inter_xa,inter_zb,inter_yb,inter_xb,incre,Wa,Wb,ka,kb,R1a,R1b,R2a,R2b)
        coor_za[i] = za+inter_za+0.5*inter_za1
        coor_ya[i] = ya+inter_ya+0.5*inter_ya1
        coor_xa[i] = xa+inter_xa+0.5*inter_xa1
        coor_zb[i] = zb+inter_zb+0.5*inter_zb1
        coor_yb[i] = yb+inter_yb+0.5*inter_yb1
        coor_xb[i] = xb+inter_xb+0.5*inter_xb1

    for i in range(Tot+1):
      i = i + Tot+1
      coor_za[i] = []
      coor_ya[i] = []
      coor_xa[i] = []
      coor_zb[i] = []
      coor_yb[i] = []
      coor_xb[i] = []
      if i == Tot+1:
        coor_za[Tot+1] = -coor_za[Tot]
        coor_ya[Tot+1] = coor_ya[Tot]
        coor_xa[Tot+1] = -coor_xa[Tot]
        coor_zb[Tot+1] = -coor_zb[Tot]
        coor_yb[Tot+1] = coor_yb[Tot]
        coor_xb[Tot+1] = -coor_xb[Tot]
      else:
        za = coor_za[i-1]
        ya = coor_ya[i-1]
        xa = coor_xa[i-1]
        zb = coor_zb[i-1]
        yb = coor_yb[i-1]
        xb = coor_xb[i-1]
        inter_za,inter_ya,inter_xa,inter_zb,inter_yb,inter_xb = cal_perturb(za,ya,xa,zb,yb,xb,incre,Wa,Wb,ka,kb,R1a,R1b,R2a,R2b)
        inter_za1,inter_ya1,inter_xa1,inter_zb1,inter_yb1,inter_xb1 = cal_perturb(inter_za,inter_ya,inter_xa,inter_zb,inter_yb,inter_xb,incre,Wa,Wb,ka,kb,R1a,R1b,R2a,R2b)
        coor_za[i] = za+inter_za+0.5*inter_za1
        coor_ya[i] = ya+inter_ya+0.5*inter_ya1
        coor_xa[i] = xa+inter_xa+0.5*inter_xa1
        coor_zb[i] = zb+inter_zb+0.5*inter_zb1
        coor_yb[i] = yb+inter_yb+0.5*inter_yb1
        coor_xb[i] = xb+inter_xb+0.5*inter_xb1

    coor_za_fa = coor_za[2*Tot+1]
    coor_ya_fa = coor_ya[2*Tot+1]
    coor_xa_fa = coor_xa[2*Tot+1]
    coor_zb_fa = coor_zb[2*Tot+1]
    coor_yb_fa = coor_yb[2*Tot+1]
    coor_xb_fa = coor_xb[2*Tot+1]

  return coor_za_fa,coor_ya_fa,coor_xa_fa,coor_zb_fa,coor_yb_fa,coor_xb_fa


#------------------------------------------------------------------------------
# main script

data, code = read_data(argv[1])
number_round_1 = 10000
number_round_2 = 0
number_round = number_round_1 + number_round_2
bs = ((0,80), (0,40), (-0.2,9), (0,1), (0,1), (0,2))


for i in code:
  off_lib_R1rho = read_library('library/offset_profile_R1rho',data['R1rho'][i][0])
  R1_lib_R1rho = read_library('library/R1_profile_R1rho',data['R1rho'][i][0])
  R2_lib_R1rho = read_library('library/R2_profile_R1rho',data['R1rho'][i][0])
  off_lib_R2rho = read_library('library/offset_profile_R2rho',data['R2rho'][i][0])
  R1_lib_R2rho = read_library('library/R1_profile_R2rho',data['R2rho'][i][0])
  R2_lib_R2rho = read_library('library/R2_profile_R2rho',data['R2rho'][i][0])
  J_lib_R2rho = read_J('library/J_profile_R2rho',data['R2rho'][i][0])
  relax_R1rho = []
  relax_R2rho = []
  relax_R1rho = (data['R1rho'][i][3],data['R1rho'][i][4],data['R1rho'][i][5],data['R1rho'][i][6],data['R1rho'][i][9],data['R1rho'][i][10],data['R1rho'][i][11],data['R1rho'][i][12])
  relax_R2rho = (data['R2rho'][i][3],data['R2rho'][i][4],data['R2rho'][i][5],data['R2rho'][i][6],data['R2rho'][i][9],data['R2rho'][i][10],data['R2rho'][i][11],data['R2rho'][i][12])
  offset1 = data['R1rho'][i][0]*4
  if abs(offset1) > 20:
    continue
  R1 = (data['R1rho'][i][1],data['R1rho'][i][7])
  R2 = (data['R1rho'][i][2],data['R1rho'][i][8])
  R1ex = -(data['R1rho'][i][1]+2*data['R1rho'][i][7])

  offset_int1 = int(offset1)
  if abs(offset1-offset_int1) > 0.5:
    if offset1 > 0:
      offset_int1 = offset_int1 + 1
    else:
      offset_int1 = offset_int1 - 1

  para_set = {}
  for j in range(100):
    rand1 = random.random()
    rand2 = random.random()
    rand3 = random.random()
    rand4 = random.random()
    rand5 = random.random()
    rand6 = random.random()

    rand1 = 41 * rand1
    rand1 = int(rand1) + 20
    rand2 = 41 * rand2
    rand2 = int(rand2)
    rand3 = 10 * rand3
    rand3 = int(rand3)
    rand4 = 51 * rand4
    S1 = 0.02*int(rand4)    
    rand6 = 21 * rand6
    S3 = 0.05*int(rand6)

    id_ini1 = (offset_int1,rand1,rand2,rand3)
    es_R1rho1 = linear_relax_vert(id_ini1,R1[0],2*S1*R2[0],off_lib_R1rho,R1_lib_R1rho,R2_lib_R1rho)
    es_R2rho1 = linear_relax_trans(id_ini1,R1[0],2*S1*R2[0],S3*R1ex,off_lib_R2rho,R1_lib_R2rho,R2_lib_R2rho,J_lib_R2rho)
    distance = math.sqrt(cal_diff(relax_R1rho,es_R1rho1)+cal_diff(relax_R2rho,es_R2rho1))
    temp = (rand1,rand2,rand3,S1,S3,distance)
    memo = (rand1,rand2,rand3,S1,S3,distance) 
    for k in range(number_round):
      B = 3*((1-(float(k)/float(number_round)))**2)
      rand7 = random.random()
      rand8 = random.random()
      rand9 = random.random()
      rand7 = 5 * rand7
      rand7 = int(rand7)
      rand8 = 2 * rand8
      rand8 = 2*int(rand8)-1
      ad = func(rand7,rand8)
      id_new1 = (offset_int1,temp[0]+ad[0],temp[1]+ad[1],temp[2]+ad[2])
      S1 = temp[3]+0.02*ad[3]
      S3 = temp[4]+0.05*ad[4]
      if id_new1[1] > 80 or id_new1[1] < 0:
        ad = func(rand7,-rand8)
        id_new1 = (offset_int1,temp[0]+ad[0],temp[1]+ad[1],temp[2]+ad[2])
      if id_new1[2] > 40 or id_new1[2] < 0:
        ad = func(rand7,-rand8)
        id_new1 = (offset_int1,temp[0]+ad[0],temp[1]+ad[1],temp[2]+ad[2])
      if id_new1[3] > 9 or id_new1[3] < 0:
        ad = func(rand7,-rand8)
        id_new1 = (offset_int1,temp[0]+ad[0],temp[1]+ad[1],temp[2]+ad[2])
      if S1 > 1 or S1 < 0:
        ad = func(rand7,-rand8)
        S1 = temp[3]+0.02*ad[3]
      if S3 > 1 or S3 < 0:
        ad = func(rand7,-rand8)
        S3 = temp[4]+0.05*ad[4]
      es_R1rho1 = linear_relax_vert(id_new1,R1[0],2*S1*R2[0],off_lib_R1rho,R1_lib_R1rho,R2_lib_R1rho)
      es_R2rho1 = linear_relax_trans(id_new1,R1[0],2*S1*R2[0],S3*R1ex,off_lib_R2rho,R1_lib_R2rho,R2_lib_R2rho,J_lib_R2rho)
      distance = math.sqrt(cal_diff(relax_R1rho,es_R1rho1)+cal_diff(relax_R2rho,es_R2rho1))

      if distance < temp[5] or distance-temp[5] < B*rand9:
        if distance < memo[5]:
          memo = (id_new1[1],id_new1[2],id_new1[3],S1,S3,distance)
        temp = (id_new1[1],id_new1[2],id_new1[3],S1,S3,distance)

    def dist(x1,x2,x3,x4,x5,x6,r1,r2,R1,R2,offset1,off_lib_R1rho,R1_lib_R1rho,R2_lib_R1rho,off_lib_R2rho,R1_lib_R2rho,R2_lib_R2rho,J_lib_R2rho):
      R1ex = -(R1[0]+2*R1[1])
      offset_int1 = int(offset1)
      if abs(offset1-offset_int1) > 0.5:
        if offset1 > 0:
          offset_int1 = offset_int1 + 1
        else:
          offset_int1 = offset_int1 - 1

      id_new1 = (offset1,x1,x2,x3)

      id_int1 = (offset_int1,int(id_new1[1]),int(id_new1[2]),int(id_new1[3]))
      ad1, ad2, ad3 = 0, 0, 0
      if id_new1[1]-id_int1[1] > 0.5:
        ad1 = 1
      if id_new1[2]-id_int1[2] > 0.5:
        ad2 = 1
      if id_new1[3]-id_int1[3] > 0.5:
        ad3 = 1
      id_int1 = (id_int1[0],id_int1[1]+ad1,id_int1[2]+ad2,id_int1[3]+ad3)

      es_R1rho1 = poly_relax_vert(id_new1,id_int1,x6*R1[0],2*x4*R2[0],off_lib_R1rho,R1_lib_R1rho,R2_lib_R1rho)
      es_R2rho1 = poly_relax_trans(id_new1,id_int1,x6*R1[0],2*x4*R2[0],x5*R1ex,off_lib_R2rho,R1_lib_R2rho,R2_lib_R2rho,J_lib_R2rho)
      dif_R1 = ((1-x6)*R1[0]/float(R1[1]))**2
      dif_R2 = ((abs(2*x4*R2[0]-x6*R1[0])-(2*x4*R2[0]-x6*R1[0]))/float(R2[1]+R1[1]))**2 + ((abs(1-2*x4)-(1-2*x4))*R2[0]/float(R2[1]))**2 
      distance = math.sqrt(X_diff(r1,es_R1rho1)+X_diff(r2,es_R2rho1)+dif_R1+dif_R2)

      return distance

    def fun(x):
      distance = dist(x[0],x[1],x[2],x[3],x[4],x[5],relax_R1rho,relax_R2rho,R1,R2,offset1,off_lib_R1rho,R1_lib_R1rho,R2_lib_R1rho,off_lib_R2rho,R1_lib_R2rho,R2_lib_R2rho,J_lib_R2rho)
      return distance

    x0 = np.array([memo[0], memo[1], memo[2], memo[3], memo[4], 1])
    res = minimize(fun, x0, method='L-BFGS-B', bounds=bs, options={'disp': False})
    para_set[(res['x'][0], res['x'][1], res['x'][2], res['x'][3], res['x'][4], res['x'][5], res['fun'])] = res['fun']
  items = para_set.items()
  backitems = [[v[1],v[0]] for v in items]
  backitems.sort()
  sortedlist = [backitems[j][1] for j in range(len(backitems))]

  Rex = []
  for j in range(10):
    Wb = abs(0.2*(sortedlist[j][0]-40)/float(2*math.pi))  
    kex = 10**(2+0.1*sortedlist[j][1])/float(2*math.pi*1000)
    pa = 0.01*(100-5*sortedlist[j][2]-1)
    coor_za, coor_ya, coor_xa, coor_zb, coor_yb, coor_xb = cal_relax(0,Wb,kex*(1-pa),kex*pa,R1[0],R2[0],5,4)
    trans_relax_L = math.log(math.sqrt(coor_ya**2+coor_xa**2)+math.sqrt(coor_yb**2+coor_xb**2))/float(-0.04)
    coor_za, coor_ya, coor_xa, coor_zb, coor_yb, coor_xb = cal_relax(0,Wb,kex*(1-pa),kex*pa,R1[0],R2[0],0.25,80)
    trans_relax_H = math.log(math.sqrt(coor_ya**2+coor_xa**2)+math.sqrt(coor_yb**2+coor_xb**2))/float(-0.04)
    Rex.append(float(trans_relax_L-trans_relax_H))

  print(i,np.mean(Rex),np.std(Rex))

