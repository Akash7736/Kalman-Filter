import numpy as np
from tf.transformations import euler_from_quaternion
from math import pi, radians, cos, sin
import pyproj

'''
Base class for defining sensor measurements
'''

class imu():

    def __init__(self, name, topic, frame):
        self.name = name
        self.topic_name = topic
        self.frame = frame
        self.flag = 0

        self.lin_acceleration = np.zeros(3)
        self.orientation_q = []
        self.angular_velocity = np.zeros(3)
        self.orientation_e = np.zeros(3)

    def gps_to_xy_ENU(self, origin_lat, origin_long, goal_lat, goal_long):
    # Calculate distance and azimuth between GPS points
        geodesic = pyproj.Geod(ellps='WGS84')
        azimuth,back_azimuth,distance = geodesic.inv(origin_long, origin_lat, goal_long, goal_lat)
        # Convert polar (distance and azimuth) to x,y translation in meters (needed for ROS) 
        # by finding side lengths of a right-angle triangle
        # Convert azimuth to radians
        azimuth = radians(azimuth)
        # To get (x,y) in ENU frame
        self.y = adjacent = cos(azimuth) * distance
        self.x = opposite = sin(azimuth) * distance
        # print("xy: ", self.x, self.y))
        self.flag = 0
    
    def callback(self, msg):
        self.angular_velocity[2] = msg.angular_velocity.z

        self.lin_acceleration[0] = msg.linear_acceleration.x
        self.lin_acceleration[1] = msg.linear_acceleration.y

        self.orientation_q = msg.orientation
        orientation_list = [self.orientation_q.x, self.orientation_q.y, self.orientation_q.z, self.orientation_q.w]
        self.orientation_e[0] = euler_from_quaternion(orientation_list)[0]
        self.orientation_e[1] = euler_from_quaternion(orientation_list)[1]
        self.orientation_e[2] = euler_from_quaternion(orientation_list)[2]
        # self.transfrom_to_ENU()
        # self.transform_to_NED()
        self.flag = 1

    def transform_to_ENU(self):
        if(self.frame.casefold() == 'ENU'.casefold()):
            pass
        elif(self.frame.casefold() == 'NWU'.casefold()):
            self.orientation_e[2] = self.orientation_e[2] + pi/2 
        elif(self.frame.casefold() == 'NED'.casefold()):
            self.orientation_e[2] = self.orientation_e[2] - pi/2
            self.angular_velocity[2] = -self.angular_velocity[2]
            self.lin_acceleration[1] = -self.lin_acceleration[1]

    def transform_to_NED(self):
        if(self.frame.casefold() == 'ENU'.casefold()):
            self.orientation_e[2] = -self.orientation_e[2] + pi/2
            self.angular_velocity[2] = -self.angular_velocity[2]
            self.lin_acceleration[1] = -self.lin_acceleration[1]
        elif(self.frame.casefold() == 'NWU'.casefold()):
            self.orientation_e[2] = -self.orientation_e[2]
            self.angular_velocity[2] = -self.angular_velocity[2]
            self.lin_acceleration[1] = -self.lin_acceleration[1]
        elif(self.frame.casefold() == 'NED'.casefold()):
            exit()
    

class gnss():

    def __init__(self, name, topic, origin=[]):
        self.name = name
        self.topic_name = topic
        self.origin_flag = 0
        if len(origin) != 0:
            self.origin_gps = origin
            self.origin_flag = 1
        else:
            self.origin_gps = []
        self.flag = 0
        self.lat = 0
        self.long = 0
        self.x = 0
        self.y = 0
        self.dist = np.zeros(2)

    def callback(self, msg):
        self.lat = msg.latitude
        self.long = msg.longitude

        if self.origin_flag == 0:
            # if origin is provided that is taken as datum, else the first gps message is used as datum.
            self.origin_gps = [self.lat, self.long]
            self.origin_flag = 1
            # print("origin: ", self.origin_gps[0], self.origin_gps[1])
        self.gps_to_xy_ENU(self.origin_gps[0], self.origin_gps[1], self.lat, self.long)
        # self.gps_to_xy_NED(self.origin_gps[0], self.origin_gps[1], self.lat, self.long)
        self.flag = 1

    def gps_to_xy_ENU(self, origin_lat, origin_long, goal_lat, goal_long):
    # Calculate distance and azimuth between GPS points
        geodesic = pyproj.Geod(ellps='WGS84')
        azimuth,back_azimuth,distance = geodesic.inv(origin_long, origin_lat, goal_long, goal_lat)
        # Convert polar (distance and azimuth) to x,y translation in meters (needed for ROS) 
        # by finding side lengths of a right-angle triangle
        # Convert azimuth to radians
        azimuth = radians(azimuth)
        # To get (x,y) in ENU frame
        self.y = adjacent = cos(azimuth) * distance
        self.x = opposite = sin(azimuth) * distance
        # print("xy: ", self.x, self.y)

    def gps_to_xy_NED(self, origin_lat, origin_long, goal_lat, goal_long):
    # Calculate distance and azimuth between GPS points
        geodesic = pyproj.Geod(ellps='WGS84')
        azimuth,back_azimuth,distance = geodesic.inv(origin_long, origin_lat, goal_long, goal_lat)
        # Convert polar (distance and azimuth) to x,y translation in meters (needed for ROS) 
        # by finding side lengths of a right-angle triangle
        # Convert azimuth to radians
        azimuth = radians(azimuth)
        # To get (x,y) in NED frame
        self.y = adjacent = sin(azimuth) * distance
        self.x = opposite = cos(azimuth) * distance
        # print("xy: ", self.x, self.y)


if __name__ == '__main__':
    print("This is just the class, please instantiate the objects of this class along with ekf.py file")



