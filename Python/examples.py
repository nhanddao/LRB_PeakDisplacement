# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 20:29:23 2024
@author: nhand
"""
import numpy as np
import LRB_PeakDisp

# Isolation system from Naeim, F., Kelly, J. M. Design of Seismic Isolated Structures: From Theory to Practice", John Wiley and Sons, NY, USA, 1999.
fd= 0.04275
Td= 3.016
k1d= 9.782
# 1s-spectral acceleration
S1= 0.7 # measured in g. i.e, A1= 0.7g
# assume that the design spectrum follows a hyperbole curve for periods larger than 1.0s
# then spectral acceleration at 1.0 s, 1.5 s, 2.0 s,...,5.0 s are
S1To5= S1/np.arange(1.0,5.5,0.5)
S3= S1To5[4] # 3s-period spectral acceleration
print('============================================')
print('PEAK DISPLACEMENT PER DIFFERENT PREDICTORS\n(in meters)')
print('===========================================')
#=====================================
print('Practical equation derived from Equivalent Linear Force:\n    ' + str(
      LRB_PeakDisp.ELF(fd, Td, k1d, S1)
      )+ '\n')
#=====================================
print('=============================')
print('PRACTICAL EQUATIONS USING S3:\n')
#=====================================
print('[+] Mixed ground motions, 50 percentile:\n    ' + str(
      LRB_PeakDisp.D50_MixedGM(fd, Td, k1d, S3)
      )+ '\n')
#=====================================
print('[+] Mixed ground motions, 90 percentile:\n    ' + str(
      LRB_PeakDisp.D90_MixedGM(fd, Td, k1d, S3)
      )+ '\n')
#=====================================
print('[+] Mixed ground motions, 95 percentile:\n    ' + str(
      LRB_PeakDisp.D95_MixedGM(fd, Td, k1d, S3)
      )+ '\n')
#=====================================
print('[+] Pulse-like ground motions, 50 percentile:\n    ' + str(
      LRB_PeakDisp.D50_PulseLikeGM(fd, Td, k1d, S3)
      )+ '\n')
#=====================================
print('[+] Pulse-like ground motions, 90 percentile:\n    ' + str(
      LRB_PeakDisp.D90_PulseLikeGM(fd, Td, k1d, S3)
      )+ '\n')
#=====================================
print('[+] Pulse-like ground motions, 95 percentile:\n    ' + str(
      LRB_PeakDisp.D95_PulseLikeGM(fd, Td, k1d, S3)
      )+ '\n')
#=====================================
print('[+] No-pulse ground motions, 50 percentile:\n    ' + str(
      LRB_PeakDisp.D50_NoPulseGM(fd, Td, k1d, S3)
      )+ '\n')
#=====================================
print('[+] No-pulse ground motions, 90 percentile:\n    ' + str(
      LRB_PeakDisp.D90_NoPulseGM(fd, Td, k1d, S3)
      )+ '\n')
#=====================================
print('[+] No-pulse ground motions, 95 percentile:\n    ' + str(
      LRB_PeakDisp.D95_NoPulseGM(fd, Td, k1d, S3)
      )+ '\n')
#=====================================
print('[+] Any motion type, 50 percentile:\n    ' + str(
      LRB_PeakDisp.D50(fd, Td, k1d, S3)
      )+ '\n')
#=====================================
print('[+] Any motion type, 90 percentile:\n    ' + str(
      LRB_PeakDisp.D90(fd, Td, k1d, S3)
      )+ '\n')
#=====================================
print('[+] Any motion type, 95 percentile:\n    ' + str(
      LRB_PeakDisp.D95(fd, Td, k1d, S3)
      )+ '\n')
#=====================================
print('=============================')
print('ANN MODELS USING S1 TO S5:\n')
#=====================================
print('[+] Mixed ground motion:\n    ' + str(
      LRB_PeakDisp.ANN_MixedGM(fd,Td,k1d, S1To5)
      )+ '\n')
#=====================================
print('[+] Pulse-like ground motion:\n    ' + str(
      LRB_PeakDisp.ANN_PulseLikeGM(fd,Td,k1d, S1To5)
      )+ '\n')
#=====================================
print('[+] No-pulse ground motion:\n    ' + str(
      LRB_PeakDisp.ANN_NoPulseGM(fd,Td,k1d, S1To5)
      )+ '\n')
#=====================================
print('[+] Any motion type:\n    ' + str(
      LRB_PeakDisp.ANN_AllGMTypes(fd,Td,k1d, S1To5)
      )+ '\n')
