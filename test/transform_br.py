#!/usr/bin/env python  
import rospy
import tf
import tf.transformations as tft
from geometry_msgs.msg import TransformStamped

import numpy as np


if __name__ == '__main__':
    rospy.init_node('tf_publisher')

    br = tf.TransformBroadcaster()
    freq = 10.0
    rate = rospy.Rate(freq)  # 10 Hz
    
    t = 0.0
    dt = 1.0 / freq
    
    counter = 0

    while not rospy.is_shutdown():         
        
        # Example: translate by (1, 2, 0), rotate about Z axis by 45 deg
        translation = [1.0, 2.0, 0.0] + 0.03 * np.random.randn(3) + [0.1 * np.sin(t), 0, 0.1 * np.cos(t)]
        rotation = tft.quaternion_from_euler(0, 0, 0.785398)  # 45 degrees in radians
        
        # Add random outliers
        if counter % 100 <= 2:
            translation[0] += 1.0
            translation[2] += 1.0

        # Broadcast transform
        br.sendTransform(
            translation,
            rotation,
            rospy.Time.now(),
            "t1_beam1",   # child frame
            "world"         # parent frame
        )

        t += dt
        
        counter += 1
        
        rate.sleep()