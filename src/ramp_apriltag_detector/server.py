import rospy
import rospkg
from geometry_msgs.msg import TransformStamped, Transform, Vector3, Quaternion
import yaml
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R
import tf2_ros
import logging

logger = logging.Logger(__name__)


@dataclass
class ScTf:
    """Scipy Transform"""

    lin: np.ndarray
    rot: np.ndarray

    def __str__(self) -> str:
        p_str = ", ".join(f"p.{c}: {v:.3f}" for c, v in zip(["x", "y", "z"], self.lin))
        q_str = ", ".join(
            f"q.{c}: {v:.3f}" for c, v in zip(["x", "y", "z", "w"], self.rot)
        )
        return f"{p_str}, {q_str}"

    def apply(self, right: np.ndarray) -> np.ndarray:
        """left compose self with arg transform"""

        left = self

        if isinstance(right, ScTf):
            # transform composition

            return ScTf(
                lin=apply_rot(left.rot, right.lin) + left.lin,
                rot=apply_rot(left.rot, right.rot),
            )
        elif isinstance(right, np.ndarray):
            # transform applied to vector
            return apply_rot(left.rot, right) + left.lin
        raise ValueError("argument type not supported")

    def inv(self):
        inv_rot = R.from_quat(self.rot).inv().as_quat()
        return ScTf(lin=apply_rot(inv_rot, -self.lin), rot=inv_rot)


def apply_rot(rot_a: R, rot_b: R) -> R:
    """return composition of rot_b then rot_a"""

    # assert isinstance(rot_a, R), "must be scipy.transform.rotation.R"
    # return copy.deepcopy(rot_a * rot_b)

    assert isinstance(rot_a, np.ndarray)

    if rot_b.shape[-1] == 4:
        return (R.from_quat(rot_a) * R.from_quat(rot_b)).as_quat()
    elif rot_b.shape[-1] == 3:
        return R.from_quat(rot_a).apply(rot_b)
    raise ValueError("argument type not supported")


def compose(*transforms):
    '''apply transforms in sequence order'''

    if len(transforms) == 0:
        raise ValueError

    # solve edge case of not unpacking transforms
    if isinstance(transforms[0], list):
        if len(transforms) > 1:
            raise ValueError
        transforms = transforms[0]


    transform = transforms[-1]
    for T in reversed(transforms[:-1]):
        transform = T.apply(transform)
    return transform

def rot_inv(rot: R) -> R:
    return rot.inv()


def rot_diff(p, q) -> R:
    """returns pq^-1"""
    # this is the rotation from q to p
    return apply_rot(p, rot_inv(q))


def mean_quaternion(q_list: list):
    # https://ntrs.nasa.gov/api/citations/20070017872/downloads/20070017872.pdf
    # Page 4

    n = len(q_list)
    qq_sum = sum(q[:, None] @ q[None, :] for q in q_list) / n

    eig_val, v = np.linalg.eig(qq_sum)
    max_col = np.argmax(np.real(eig_val))

    q_avg = v[:, max_col]

    return q_avg


def dct_to_NumpyTransform(dct):
    if dct is None:
        return None
    x = dct
    lin = np.array((x["x"], x["y"], x["z"]))
    rot = np.array((x["qx"], x["qy"], x["qz"], x["qw"]))
    return ScTf(lin, rot)


@dataclass
class Beam:
    name: str
    beam_tags: list  # list[Tag]
    transform: ScTf

    def offsets(self):
        if all(t.beam_to_tag is not None for t in self.beam_tags):
            return {t.name: t.beam_to_tag for t in self.beam_tags}

    def calc_origin(self) -> ScTf:
        """compute beam origin from tags, requires all tags to be present"""

        n = len(self.beam_tags)

        tag_transforms = [tag.world_transform for tag in self.beam_tags]
        if any(t is None for t in tag_transforms):
            return

        # get mean of tag positions
        mean_lin = np.array(
            [0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        for t in tag_transforms:
            mean_lin += t.lin
        mean_lin /= n

        # use the rotation  of the first tag
        rotation = self.beam_tags[0].world_transform.rot
        # q = np.array([0.0, 0.0, 0.0, 1.0])

        self.transform = ScTf(mean_lin, rotation)

        return self.transform

    def beam_to_tags(self) -> dict:
        """requires calc_origin to be called_first"""
        """returns dictionary of offsets"""
        if self.transform is None:
            return

        world_to_beam = self.transform

        # this SHOULD be a dictionary of tag_id: beam_to_tag transform
        return {
            tag.name: compose(world_to_beam.inv(), tag.world_transform) for tag in self.beam_tags
        }


@dataclass
class Tag:
    name: str
    world_transform: ScTf
    parent: Beam = None
    beam_to_tag: ScTf = None


class Beamtracker:
    def __init__(self) -> None:
        import os

        self.path = os.path.join(rospkg.RosPack().get_path('ramp_apriltag_detector'), "config", "beams.yaml")
        self.beam_config = yaml.safe_load(open(self.path))

        self.beams = {}
        self.beam_tags = {}
        
        self._beam_names = []

        for b in self.beam_config["beams"]:
            beam_obj = Beam(b["name"], None, None)
            tags = [
                (
                    t["name"],
                    Tag(
                        t["name"],
                        None,
                        beam_obj,
                        beam_to_tag=dct_to_NumpyTransform(t.get("offset", None)),
                    ),
                )
                for t in b["tags"]
            ]
            beam_obj.beam_tags = list(zip(*tags))[1]

            self.beams[b["name"]] = beam_obj
            for t, T in tags:
                self.beam_tags[t] = T

        self.link_tags = {
            link["name"]: Tag(link["name"], None) for link in self.beam_config["links"]
        }
    
    @property
    def beam_names(self):
        self._beam_names = list(beam.name for beam in self.beams.values())
        return self._beam_names

    def tags(self):
        return tuple(list(self.beam_tags.keys()) + list(self.link_tags.keys()))


def tf_to_ScTf(tf):
    p = tf.translation
    q = tf.rotation
    return ScTf(np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w]))


def ScTf_to_tf(npt: ScTf):
    tf = Transform()
    tf.translation = Vector3(*npt.lin)
    tf.rotation = Quaternion(*npt.rot)
    return tf

def ScTf_to_dct(tf: ScTf) -> dict:
    keys = ("x", "y", "z", "qx", "qy", "qz", "qw")
    return {k: float(v) for k, v in zip(keys, (*tf.lin, *tf.rot))}


def configure_beam(data: TransformStamped, beam_tracker: Beamtracker):
    tf = data.transform
    tag_id = data.child_frame_id

    is_link = bool(tag_id in beam_tracker.link_tags)
    # store position
    # if both positions exist publish relative transform
    # filter with Finite MA(?)

    # is this a peg? if so return None, store transform
    if is_link:
        beam_tracker.link_tags[tag_id].world_transform = tf_to_ScTf(tf)
        return

    # if beam....

    tag = beam_tracker.beam_tags[tag_id]
    tag.world_transform = tf_to_ScTf(tf)

    beam: Beam = tag.parent
    res = beam.calc_origin()
    if res is not None:
        offsets = beam.beam_to_tags()
        overwrite_beamconfig(beam_tracker.path, beam.name, offsets)
        offsets = {t: str(v) for t, v in offsets.items()}
        logging.info(f"{beam.name}: {offsets}")


def get_beamposition(
    data: TransformStamped,
    beam_tracker: Beamtracker,
    #   tf_broadcaster: tf2_ros.TransformBroadcaster
):
    tf = data.transform
    tag_id = data.child_frame_id

    is_link = bool(tag_id in beam_tracker.link_tags)
    # store position
    # if both positions exist publish relative transform
    # filter with Finite MA(?)

    if is_link:
        beam_tracker.link_tags[tag_id].world_transform = tf_to_ScTf(tf)
        return

    tag = beam_tracker.beam_tags[tag_id]

    tag_world = tf_to_ScTf(tf)

    beam: Beam = tag.parent

    beam_tfstamped = TransformStamped()
    beam_tfstamped.header.stamp = rospy.Time.now()
    beam_tfstamped.header.frame_id = data.header.frame_id
    beam_tfstamped.child_frame_id = beam.name

    beam_to_tag: ScTf = beam.offsets()[tag_id]

    beam_pose = compose(tag_world, beam_to_tag.inv())

    beam_tfstamped.transform = ScTf_to_tf(beam_pose)

    return beam_tfstamped


def overwrite_beamconfig(config_file, beam_name, offsets):
    with open(config_file) as f:
        beams_dct = yaml.safe_load(f)
    beam_list: list = beams_dct.get("beams")
    beam_index = [beam.get("name") for beam in beam_list].index(beam_name)

    for tag_dct in beam_list[beam_index].get("tags"):
        for k, v in offsets.items():
            if k == tag_dct.get("name"):
                tag_dct["offset"] = ScTf_to_dct(v)
    
    try:
        with open(config_file, "w") as f:
            yaml.safe_dump(beams_dct, f)
            
    except KeyboardInterrupt as e:
        with open(config_file, "w") as f:
            yaml.safe_dump(beams_dct, f)
        raise KeyboardInterrupt

def listen_to(beam_tracker: Beamtracker, mode: str = "detect"):
    tags_names = beam_tracker.tags()

    print(tags_names)

    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)
    
    parent_frame = "world"

    if mode == "detect":
        br = tf2_ros.TransformBroadcaster()

    while not rospy.is_shutdown():
        rate = rospy.Rate(30.0)
        transforms = []
        for tf_id in tags_names:
            try:
                transforms.append(
                tf_buffer.lookup_transform(parent_frame, tf_id, rospy.Time(0))
                )
            except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as e:
                continue
                
        # tf in a list transforms
        for tf in transforms:
            if mode == "detect":
                # skip stale TFs
                if rospy.Time.now() - tf.header.stamp > rospy.Duration(5.0):
                    continue

                tf_to_publish = get_beamposition(tf, beam_tracker)
                if tf_to_publish is not None:
                    #
                    br.sendTransform(tf_to_publish)

            elif mode == "config":
                configure_beam(tf, beam_tracker)

        rate.sleep()

