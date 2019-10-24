import math
from transformations import *
from numpy.linalg import norm, inv
from humanoid_base import HumanoidBase



def remap_coordinates(joints):
    for joint in joints:
        x = joint[0]
        y = joint[1]
        z = joint[2]
        joint[2] = y
        joint[1] = -x
        joint[0] = -z

class Humanoid(HumanoidBase):
    def __init__(self, portAX='COM8', portRX='COM11'):
        super().__init__(portAX, portRX)
        self.angles = {'RAy': 0.0, 'RAx': 0.0, 'RFAy': 0.0,
                       'LAy': 0.0, 'LAx': 0.0, 'LFAy': 0.0,
                       'Hz': 0}
        # wartosci, ktore beda wyslane do dynamixelli
        self.dynamixel_values = [0.0] * 20
        self.dimensions = {'arm': 1.0, 'forearm': 1.0, 'shoulder': 0.5, 'spine': 1}
        max_torque = [255] * 20
        self.sync_write_2(self.ADR_TOR_LIM, max_torque)

    def send_signals(self, k=0.05):
        # ustawienie wartosci wywyslanych do dynamixeli, z filtrem low pass
        self.dynamixel_values[13] = self.dynamixel_values[13]*(1-k) + k*(512 - int(self.angles['RAy']*1024/np.deg2rad(300)))
        self.dynamixel_values[14] = self.dynamixel_values[14]*(1-k) + k*(512 - int(self.angles['RAx'] * 1024 / np.deg2rad(300)))
        self.dynamixel_values[15] = self.dynamixel_values[15]*(1-k) + k*(512 + int(self.angles['RFAy'] * 1024 / np.deg2rad(300)))

        self.dynamixel_values[16] = self.dynamixel_values[16]*(1-k) + k*(512 + int(self.angles['LAy'] * 1024 / np.deg2rad(300)))
        self.dynamixel_values[17] = self.dynamixel_values[17]*(1-k) + k*(512 - int(self.angles['LAx'] * 1024 / np.deg2rad(300)))
        self.dynamixel_values[18] = self.dynamixel_values[18]*(1-k) + k*(512 - int(self.angles['LFAy'] * 1024 / np.deg2rad(300)))

        self.dynamixel_values[19] = self.dynamixel_values[19]*(1-k) + k*(512 + int((self.angles['Hz'] + np.deg2rad(80)) * 1024 / np.deg2rad(300)))

        torque = [1] * 13 + [1]*7

        self.sync_write_1(self.ADR_TOR_EN, torque)
        self.sync_write_1(self.ADR_SLOPE_CW, self.slope_CW)
        self.sync_write_1(self.ADR_SLOPE_CCW, self.slope_CCW)
        data_to_send = [int(element) for element in self.dynamixel_values]
        self.sync_write_2(30, data_to_send)



    def getLeftHandPosition(self):
        R1 = rotation_y(self.angles['LAy'])
        R2 = rotation_x(self.angles['LAx'])
        T2 = translation([0, 0, -self.dimensions['arm']])
        R3 = rotation_y(self.angles['LFAy'])
        T3 = translation([0, 0, -self.dimensions['forearm']])
        return R1.dot(R2).dot(T2).dot(R3).dot(T3)

    def getRightHandPosition(self):
        R1 = rotation_y(self.angles['RAy'])
        R2 = rotation_x(self.angles['RAx'])
        T2 = translation([0, 0, -self.dimensions['arm']])
        R3 = rotation_y(self.angles['RFAy'])
        T3 = translation([0, 0, -self.dimensions['forearm']])
        return R1.dot(R2).dot(T2).dot(R3).dot(T3)

    # funkcja ustawiajaca katy na stawach robota w taki sposob, aby koncowka lewej reki znajdywala sie w zadanej pozycji
    #  wzgledem lewego barku
    def setLeftHand(self, desired_pos):
        d = math.sqrt(desired_pos[0] ** 2 + desired_pos[1] ** 2 + desired_pos[2] ** 2)
        arm = self.dimensions['arm']
        forearm = self.dimensions['forearm']
        x = desired_pos[0]
        y = desired_pos[1]
        z = desired_pos[2]
        if y < 0:
            y = 0
        if x == 0:
            x = 0.000001
        gamma = -math.acos((d**2 - arm**2 - forearm**2)/(2*arm*forearm))
        beta = math.asin(y/(arm+forearm*math.cos(gamma)))
        N1 = -forearm*math.cos(beta)*math.cos(gamma)-arm*math.cos(beta)
        N2 = forearm * math.sin(gamma)
        delta = 4*z**2*N1**2-4*(N1**2+N2**2)*(z**2-N2**2)
        cos1 = (2*z*N1-math.sqrt(delta))/(2*N1**2+2*N2**2)
        cos2 = (2 * z * N1 + math.sqrt(delta)) / (2 * N1 ** 2 + 2 * N2 ** 2)

        delta = 4 * z ** 2 * N2 ** 2 - 4 * (N1 ** 2 + N2 ** 2) * (z ** 2 - N1 ** 2)
        sin1 = (2 * z * N2 - math.sqrt(delta)) / (2 * N1 ** 2 + 2 * N2 ** 2)
        sin2 = (2 * z * N2 + math.sqrt(delta)) / (2 * N1 ** 2 + 2 * N2 ** 2)

        # wszystkie mozliwosci
        alphas = [math.atan2(sin1, cos1), math.atan2(sin2, cos2), math.atan2(sin2, cos1), math.atan2(sin1, cos2)]
        R2 = rotation_x(beta)
        T2 = translation([0, 0, -arm])
        R3 = rotation_y(gamma)
        T3 = translation([0,0,-forearm])

        transformation = R2.dot(T2).dot(R3).dot(T3)

        min_distance = 999
        alpha = 0
        for angle in alphas:
            hand_position = (rotation_y(angle).dot(transformation))[0:3, 3]
            distance = np.linalg.norm(np.subtract(hand_position, desired_pos))
            if distance < min_distance:
                min_distance = distance
                alpha = angle
                # print(hand_position)

        self.angles['LAy'] = alpha
        self.angles['LAx'] = beta
        self.angles['LFAy'] = gamma

    def setRightHand(self, desired_pos):
        d = math.sqrt(desired_pos[0] ** 2 + desired_pos[1] ** 2 + desired_pos[2] ** 2)
        arm = self.dimensions['arm']
        forearm = self.dimensions['forearm']
        x = desired_pos[0]
        y = desired_pos[1]
        z = desired_pos[2]
        if y > 0:
            y = 0
        if x == 0:
            x = 0.000001

        gamma = - math.acos((d ** 2 - arm ** 2 - forearm ** 2) / (2 * arm * forearm))
        beta = math.asin(y / (arm + forearm * math.cos(gamma)))
        N1 = -forearm * math.cos(beta) * math.cos(gamma) - arm * math.cos(beta)
        N2 = forearm * math.sin(gamma)
        delta = 4 * z ** 2 * N1 ** 2 - 4 * (N1 ** 2 + N2 ** 2) * (z ** 2 - N2 ** 2)
        cos1 = (2 * z * N1 - math.sqrt(delta)) / (2 * N1 ** 2 + 2 * N2 ** 2)
        cos2 = (2 * z * N1 + math.sqrt(delta)) / (2 * N1 ** 2 + 2 * N2 ** 2)

        delta = 4 * z ** 2 * N2 ** 2 - 4 * (N1 ** 2 + N2 ** 2) * (z ** 2 - N1 ** 2)
        sin1 = (2 * z * N2 - math.sqrt(delta)) / (2 * N1 ** 2 + 2 * N2 ** 2)
        sin2 = (2 * z * N2 + math.sqrt(delta)) / (2 * N1 ** 2 + 2 * N2 ** 2)

        # wszystkie mozliwosci
        alphas = [math.atan2(sin1, cos1), math.atan2(sin2, cos2), math.atan2(sin2, cos1), math.atan2(sin1, cos2)]
        R2 = rotation_x(beta)
        T2 = translation([0, 0, -arm])
        R3 = rotation_y(gamma)
        T3 = translation([0, 0, -forearm])

        transformation = R2.dot(T2).dot(R3).dot(T3)

        min_distance = 999
        alpha = 0
        for angle in alphas:
            hand_position = (rotation_y(angle).dot(transformation))[0:3, 3]
            distance = np.linalg.norm(np.subtract(hand_position, desired_pos))
            if distance < min_distance:
                min_distance = distance
                alpha = angle
                # print(hand_position)

        self.angles['RAy'] = alpha
        self.angles['RAx'] = beta
        self.angles['RFAy'] = gamma

    # funckja zamieniajaca osie tak, aby zgadzaly sie z robotem


    def mimic_kinect(self, joints):
        joint_mapping = {'SpineBase': 0, 'SpineMid': 1, 'Neck': 2, 'Head': 3, 'ShoulderLeft': 4, 'ElbowLeft': 5,
                         'WristLeft': 6,
                         'HandLeft': 7, 'ShoulderRight': 8, 'ElbowRight': 9, 'WristRight': 10, 'HandRight': 11,
                         'HipLeft': 12,
                         'KneeLeft': 13, 'AnkleLeft': 14, 'FootLeft': 15, 'HipRight': 16, 'KneeRight': 17,
                         'AnkleRight': 18,
                         'FootRight': 19, 'SpineShoulder': 20, 'HandTipLeft': 21, 'ThumbLeft': 22,
                         'HandTipRight': 23,
                         'ThumbRight': 24, 'Count': 25}

        remap_coordinates(joints)
        arm_left = joints[joint_mapping['ElbowLeft']] - joints[joint_mapping['ShoulderLeft']]
        arm_right = joints[joint_mapping['ElbowRight']] - joints[joint_mapping['ShoulderRight']]
        forearm_left = joints[joint_mapping['WristLeft']] - joints[joint_mapping['ElbowLeft']]
        forearm_right = joints[joint_mapping['WristRight']] - joints[joint_mapping['ElbowRight']]
        spine = joints[joint_mapping['Neck']] - joints[joint_mapping['SpineBase']]

        # right - to - left shoulder vector
        shoulder_line = joints[joint_mapping['ShoulderLeft']] - joints[joint_mapping['ShoulderRight']]
        # print(shoulder_line)
        chest_x_vector = np.cross(shoulder_line, spine)
        chest_orientation = np.eye(3)
        chest_orientation[:, 0] = chest_x_vector / norm(chest_x_vector)
        chest_orientation[:, 1] = shoulder_line / norm(shoulder_line)
        chest_orientation[:, 2] = np.cross(chest_orientation[:, 0], chest_orientation[:, 1])
        self.angles['Hz'] = math.atan2(-shoulder_line[0], shoulder_line[1])

        hand_left_versor = (joints[joint_mapping['WristLeft']] - joints[joint_mapping['ShoulderLeft']]) / \
                           (np.linalg.norm(arm_left) + np.linalg.norm(forearm_left))
        # hand_left_versor = np.append(hand_left_versor, [1])
        hand_right_versor = (joints[joint_mapping['WristRight']] - joints[joint_mapping['ShoulderRight']]) / \
                            (np.linalg.norm(arm_right) + np.linalg.norm(forearm_right))
        # hand_right_versor = np.append(hand_right_versor, [1])

        hand_left_rescaled = np.linalg.inv(chest_orientation).dot(
            hand_left_versor * (self.dimensions['arm'] + self.dimensions['forearm']))
        hand_right_rescaled = np.linalg.inv(chest_orientation).dot(
            hand_right_versor * (self.dimensions['arm'] + self.dimensions['forearm']))

        # print(hand_right_versor)
        # print(hand_right_rescaled)
        self.setLeftHand(hand_left_rescaled)
        self.setRightHand(hand_right_rescaled)

if __name__ == '__main__':
    h = Humanoid()
    h.setLeftHand(np.array([0.00001, 0.5, -0.999999]))
    print(h.angles)
