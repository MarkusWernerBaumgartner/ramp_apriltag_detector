#!/usr/bin/env python3

import rospy

from lti_filters.discrete_filters import (DiscreteManualLTI, LTIBaseParams)

from ramp_apriltag_detector.filter_tfs import FilterTf
from ramp_apriltag_detector.server import Beamtracker


def extract_tag_names(tag_list):
    tag_names = []

    for tag_data in tag_list:
        tag_names.append(tag_data['name'])
    return tag_names


if __name__ == "__main__":
    rospy.init_node("filter_tfs")
    
    # Get beam names
    beam_names = Beamtracker().beam_names

    standalone_tags = rospy.get_param('/apriltag_ros_continuous_node/standalone_tags')
    tag_names = extract_tag_names(standalone_tags)

    tf_to_filter = beam_names + tag_names

    # Create an instance of a filter
    # Filter dim must be 6
    dim = 6
    # Sampling frequency 
    dt = 1.0 / 30.0 # Seconds
    damping_ratio = 2.0 # Slightly under damped with no ringing
    rise_time = 0.8 # Seconds
    
    params = LTIBaseParams(rise_time, damping_ratio)
    filter = DiscreteManualLTI(dim, params, dt)

    # parent_frame = "camera_link"
    parent_frame = "base_link"
    
    # Create the tf filter
    tf_filter = FilterTf(tf_to_filter, filter, parent_frame)
    
    # Filter tfs
    tf_filter()
