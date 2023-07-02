#!/usr/bin/env python3

from register_sensor import imu, gnss
from kinematics_kf import kalman_filter
import numpy as np
from math import pi
from matplotlib.pyplot import plot

## IMU objects (name, topic, frame)
#sbg = imu('mavros-imu', '/mavros/imu/data', 'ENU')
sbg = imu('mavrosimu', '/mavros/imu/data', 'ENU')
## GPS objects (name, topic, origin)
#gps = gnss('xsens-gps', '/xsens/gnss')
gps = gnss('cubegps', '/mavros/global_position/raw/fix')
# gps = gnss('ardusimple-gps', '/ardusimple/gnss')  
kf = kalman_filter('kalman_filter_node', 10)
kf.register_sensors([sbg], [gps])

num_states = 8
num_measurements = 6

''' Initial states and Input parameters '''
''' (x,y,psi,u,v,r,ax,ay) '''
# initial_state = np.array([65,-29,0.0,0,0,0,0,0])                        # East concrete pillar
# initial_state = np.array([-82.51,-2.47,0.0,0,0,0,0,0])                # West concrete pillar wrt old rusted tank
initial_state = np.array([0,0,pi/2,0,0,0,0,0])            # 
# initial_state = np.array([0, 0, 0, 1, 0, 0, 0, 0])

initial_covariance = 1e-6*np.identity(num_states)                  ###### WAS INITIALLY 1e-6,
initial_covariance[0,0] = initial_covariance[1,1] = 0.1
# initial_covariance[2,2] = 5*1e-4
initial_covariance[2,2] = 0.5

'''
( 0 , 1,  2,  3, 4, 5)
( x,  y, psi, r, ax, ay)
'''
measurement_noise = np.zeros((num_measurements,num_measurements))
measurement_noise[0,0] =  0.05   #0.03  # 2.386235916299923e-13
measurement_noise[1,1] = 0.05     #0.03 # 7.084323962612105e-13
measurement_noise[2,2] = 7.131028243600414e-05    #3*1e-5 # 0.06
measurement_noise[3,3] = 2.605696980476807e-07     #8*1e-7 #1.52e-6#
measurement_noise[4,4] = 3.178294051157069e-05    #4*1e-6 #0.01872 #0.0002#
measurement_noise[5,5] = 2.14477382446546e-05    #4*1e-6 #0.00235 #0.0002#

'''
(0, 1,  2 , 3, 4, 5, 6 , 7 )
(x, y, psi, u, v, r, ax, ay)
'''
process_noise = np.zeros((num_states,num_states))
process_noise = np.zeros((num_states,num_states))
process_noise[0,0] = process_noise[1,1] = 0.01
process_noise[3,3] = process_noise[4,4] = 0.02           ####  lin. velocities  
process_noise[6,6] = process_noise[7,7] = 0.02           ####  accelerations
process_noise[2,2] = 0.06
process_noise[5,5] = 0.02

kf.filtered_state = initial_state
kf.state_covariance = initial_covariance
kf.measurement_noise = measurement_noise
kf.process_noise = process_noise


kf.filter()