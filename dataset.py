"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import sys
import numpy as np 
from enum import Enum
import cv2
import os
import glob
import time
import datetime as dt

from collections import namedtuple
from multiprocessing import Process, Queue, Value 
from utils import Printer 


class DatasetType(Enum):
    NONE = 1
    KITTI = 2
    TUM = 3
    VIDEO = 4
    LIVE = 6


def dataset_factory(settings):
    type=DatasetType.NONE
    associations = None    
    path = None 
    is_color = None  # used for kitti datasets

    type = settings['type']
    name = settings['name']    
    
    path = settings['base_path'] 
    path = os.path.expanduser(path)
    
    if 'associations' in settings:
        associations = settings['associations']
    if 'is_color' in settings:
        is_color = settings['is_color'].lower() == 'true'

    dataset = None 
    if type == 'kitti':
        dataset = KittiDataset(path, name, associations, DatasetType.KITTI)
        dataset.set_is_color(is_color)   
    if type == 'tum':
        dataset = TumDataset(path, name, associations, DatasetType.TUM)
    if type == 'video':
        dataset = VideoDataset(path, name, associations, DatasetType.VIDEO)
    if type == 'live':
        dataset = LiveDataset(path, name, associations, DatasetType.LIVE)   
                
    return dataset 


class Dataset(object):
    def __init__(self, path, name, fps=None, associations=None, type=DatasetType.NONE):
        self.path=path 
        self.name=name 
        self.type=type    
        self.is_ok = True
        self.fps = fps   
        if fps is not None:       
            self.Ts = 1./fps 
        else: 
            self.Ts = None 
          
        self.timestamps = None
        self.oxts = None
        self._timestamp = None       # current timestamp if available [s]
        self._next_timestamp = None  # next timestamp if available otherwise an estimate [s]

    def isOk(self):
        return self.is_ok

    def getOxts(self, frame_id):
        try:
            return self.getOxts(frame_id)
        except:
            # raise IOError('Cannot read latest Oxts measurement ')
            Printer.red('Cannot read latest Oxts measurement.')
            return None

    def getImage(self, frame_id):
        return None 

    #def getImage1(self, frame_id):
    #    return None

    def getDepth(self, frame_id):
        return None        

    def getImageColor(self, frame_id):
        try: 
            img = self.getImage(frame_id)
            if img.ndim == 2:
                return cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)     
            else:
                return img             
        except:
            img = None  
            #raise IOError('Cannot open dataset: ', self.name, ', path: ', self.path)        
            Printer.red('Cannot open dataset: ', self.name, ', path: ', self.path)
            return img    
        
    def getTimestamp(self):
        return self._timestamp
    
    def getNextTimestamp(self):
        return self._next_timestamp    

# --------------------------------- V I D E O  --------------------------------- #

class VideoDataset(Dataset): 
    def __init__(self, path, name, associations=None, type=DatasetType.VIDEO): 
        super().__init__(path, name, None, associations, type)    
        self.filename = path + '/' + name 
        #print('video: ', self.filename)
        self.cap = cv2.VideoCapture(self.filename)
        if not self.cap.isOpened():
            raise IOError('Cannot open movie file: ', self.filename)
        else: 
            print('Processing Video Input')
            self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
            self.Ts = 1./self.fps 
            print('num frames: ', self.num_frames)  
            print('fps: ', self.fps)              
        self.is_init = False   
            
    def getImage(self, frame_id):
        # retrieve the first image if its id is > 0 
        if self.is_init is False and frame_id > 0:
            self.is_init = True 
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        self.is_init = True
        ret, image = self.cap.read()
        #self._timestamp = time.time()  # rough timestamp if nothing else is available 
        self._timestamp = float(self.cap.get(cv2.CAP_PROP_POS_MSEC)*1000)
        self._next_timestamp = self._timestamp + self.Ts


        if ret is False:
            print('ERROR while reading from file: ', self.filename)
        return image       


# --------------------------------- L I V E  --------------------------------- #

class LiveDataset(Dataset): 
    def __init__(self, path, name, associations=None, type=DatasetType.VIDEO): 
        super().__init__(path, name, None, associations, type)    
        self.camera_num = name # use name for camera number
        print('opening camera device: ', self.camera_num)
        self.cap = cv2.VideoCapture(self.camera_num)   
        if not self.cap.isOpened():
            raise IOError('Cannot open camera') 
        else:
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
            self.Ts = 1./self.fps             
            print('fps: ', self.fps)    
            
    def getImage(self, frame_id):
        ret, image = self.cap.read()
        self._timestamp = time.time()  # rough timestamp if nothing else is available 
        self._next_timestamp = self._timestamp + self.Ts         
        if ret is False:
            print('ERROR in reading from camera: ', self.camera_num)
        return image           


'''
class FolderDatasetParallelStatus:
    def __init__(self, i, maxlen, listing, skip):
        self.i = i
        self.maxlen = maxlen
        self.listing = listing 
        self.skip = skip  

# this is experimental 
class FolderDatasetParallel(Dataset): 
    def __init__(self, path, name, fps=None, associations=None, type=DatasetType.VIDEO): 
        super().__init__(path, name, fps, associations, type)    
        print('fps: ', self.fps)  
        self.Ts = 1./self.fps    
        self._timestamp = 0     
        self.skip=1
        self.listing = []    
        self.maxlen = 1000000    
        print('Processing Image Directory Input')
        self.listing = glob.glob(path + '/' + self.name)
        self.listing.sort()
        self.listing = self.listing[::self.skip]
        #print('list of files: ', self.listing)
        self.maxlen = len(self.listing)
        self.i = 0        
        if self.maxlen == 0:
          raise IOError('No images were found in folder: ', path)     

        self.is_running = Value('i',1)  
        
        self.folder_status = FolderDatasetParallelStatus(i,maxlen,listing,skip)

        self.q = Queue(maxsize=10)    
        self.q.put(self.folder_status)   # pass the folder status with the initialization  
        
        self.vp = Process(target=self._update_image, args=(self.q,))
        self.vp.daemon = True                 
            
    # create thread for reading images
    def start(self):
        self.vp.start()        
        
    def quit(self):
        print('webcam closing...') 
        self.is_running.value = 0
        self.vp.join(timeout=3)     
                    
    def _update_image(self, q):
        folder_status = q.get()  
        while is_running.value == 1:
            while not q.full():
                self.current_frame = self._get_image(folder_status)
                self.q.put(self.current_frame)
                #print('q.size: ', self.q.qsize())
        time.sleep(0.005)

    def _get_image(self, folder_status):
        if self.i == folder_status.maxlen:
            return (None, False)
        image_file = folder_status.listing[self.i]
        img = cv2.imread(image_file)         
        if img is None: 
            raise IOError('error reading file: ', image_file)               
        # Increment internal counter.
        self.i = self.i + 1
        return img 
    
    # get the current frame
    def getImage(self):
        img = None 
        while not self.q.empty():  # get the last one
            self._timestamp += self.Ts
            self._next_timestamp = self._timestamp + self.Ts                  
            img = self.q.get()         
        return img    

'''

# --------------------------------- W E B C A M  --------------------------------- #
class Webcam(object):
    def __init__(self, camera_num=0):
        self.cap = cv2.VideoCapture(camera_num)
        self.current_frame = None 
        self.ret = None 
        
        self.is_running = Value('i',1)        
        self.q = Queue(maxsize=2)        
        self.vp = Process(target=self._update_frame, args=(self.q,self.is_running,))
        self.vp.daemon = True

    # create thread for capturing images
    def start(self):
        self.vp.start()        
        
    def quit(self):
        print('webcam closing...') 
        self.is_running.value = 0
        self.vp.join(timeout=3)               
        
    # process function     
    def _update_frame(self, q, is_running):
        while is_running.value == 1:
            self.ret, self.current_frame = self.cap.read()
            if self.ret is True: 
                #self.current_frame= self.cap.read()[1]
                if q.full():
                    old_frame = self.q.get()
                self.q.put(self.current_frame)
                print('q.size: ', self.q.qsize())           
        time.sleep(0.005)
                  
    # get the current frame
    def get_current_frame(self):
        img = None 
        while not self.q.empty():  # get last available image
            img = self.q.get()         
        return img


# --------------------------------- K I T T Y  --------------------------------- #

class KittiDataset(Dataset):
    def __init__(self, path, name, associations=None, type=DatasetType.KITTI): 
        super().__init__(path, name, 10, associations, type)
        self.fps = 10
        self.image_left_path = '/image_0/'
        #self.image_right_path = '/image_1/'
        self.timestamps = np.loadtxt(self.path + '/sequences/' + self.name + '/times.txt')
        self.odometry_timestamps = None
        self.oxts = None
        self.load_oxts_packets_and_poses()

        self.max_frame_id = len(self.timestamps)
        print('Processing KITTI Sequence of lenght: ', len(self.timestamps))

    def load_odometry_timestamps(self):
        """Load timestamps from file."""
        timestamp_file = os.path.join(
            self.path, 'oxts', self.name, 'timestamps.txt')

        # Read and parse the timestamps
        odometry_timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                # NB: datetime only supports microseconds, but KITTI timestamps
                # give nanoseconds, so need to truncate last 4 characters to
                # get rid of \n (counts as 1) and extra 3 digits
                t = dt.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                odometry_timestamps.append(t)
        return odometry_timestamps

    def load_oxts_packets_and_poses(self):
        """Generator to read OXTS ground truth data.
           Poses are given in an East-North-Up coordinate system
           whose origin is the first GPS position.
        """

        def rotx(t):
            """Rotation about the x-axis."""
            c = np.cos(t)
            s = np.sin(t)
            return np.array([[1, 0, 0],
                             [0, c, -s],
                             [0, s, c]])

        def roty(t):
            """Rotation about the y-axis."""
            c = np.cos(t)
            s = np.sin(t)
            return np.array([[c, 0, s],
                             [0, 1, 0],
                             [-s, 0, c]])

        def rotz(t):
            """Rotation about the z-axis."""
            c = np.cos(t)
            s = np.sin(t)
            return np.array([[c, -s, 0],
                             [s, c, 0],
                             [0, 0, 1]])

        def pose_from_oxts_packet(packet, scale):
            """Helper method to compute a SE(3) pose matrix from an OXTS packet.
            """
            er = 6378137.  # earth radius (approx.) in meters

            # Use a Mercator projection to get the translation vector
            tx = scale * packet.lon * np.pi * er / 180.
            ty = scale * er * \
                 np.log(np.tan((90. + packet.lat) * np.pi / 360.))
            tz = packet.alt
            t = np.array([tx, ty, tz])

            # Use the Euler angles to get the rotation matrix
            Rx = rotx(packet.roll)
            Ry = roty(packet.pitch)
            Rz = rotz(packet.yaw)
            R = Rz.dot(Ry.dot(Rx))

            # Combine the translation and rotation into a homogeneous transform
            return R, t

        def transform_from_rot_trans(R, t):
            """Transforation matrix from rotation matrix and translation vector."""
            R = R.reshape(3, 3)
            t = t.reshape(3, 1)
            return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

        # Per dataformat.txt
        OxtsPacket = namedtuple('OxtsPacket',
                                'lat, lon, alt, ' +
                                'roll, pitch, yaw, ' +
                                'vn, ve, vf, vl, vu, ' +
                                'ax, ay, az, af, al, au, ' +
                                'wx, wy, wz, wf, wl, wu, ' +
                                'pos_accuracy, vel_accuracy, ' +
                                'navstat, numsats, ' +
                                'posmode, velmode, orimode, ' +
                                'timestamp, ' + 'sec_since_start ')

        odometry_timestamps = self.load_odometry_timestamps()

        # Bundle into an easy-to-access structure
        OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')

        oxts_files = sorted(glob.glob(
            os.path.join(self.path, 'oxts', self.name, 'data', '*.txt')))

        # Scale for Mercator projection (from first lat value)
        scale = None
        # Origin of the global coordinate system (first GPS position)
        origin = None

        self.oxts = []

        for indx, filename in enumerate(oxts_files):
            with open(filename, 'r') as f:
                for line in f.readlines():
                    line = line.split()
                    # Last five entries are flags and counts
                    line[:-5] = [float(x) for x in line[:-5]]
                    line[-5:] = [int(float(x)) for x in line[-5:]]

                    sec_since_start =  (odometry_timestamps[indx] - odometry_timestamps[0]).total_seconds()

                    packet = OxtsPacket(*line + [odometry_timestamps[indx], sec_since_start])

                    if scale is None:
                        scale = np.cos(packet.lat * np.pi / 180.)

                    R, t = pose_from_oxts_packet(packet, scale)

                    if origin is None:
                        origin = t

                    T_w_imu = transform_from_rot_trans(R, t - origin)
                    self.oxts.append(OxtsData(packet, T_w_imu))
        print("VIDEO STARTTIME: ", self.oxts[0].packet.timestamp)
        print("ORIGIN GPS POS (lat, lon): ", self.oxts[0].packet.lat, " ", self.oxts[0].packet.lon)
        self.oxts = iter(self.oxts)



    # TODO: resample GPS (lower frequency -> every 10th frame only as 10 FPS video + add noise)

    def set_is_color(self,val):
        self.is_color = val 
        if self.is_color:
            print('dataset in color!')            
            self.image_left_path = '/image_2/'
            #self.image_right_path = '/image_3/'
        
    def getImage(self, frame_id):
        img = None
        if frame_id < self.max_frame_id:
            try: 
                img = cv2.imread(self.path + '/sequences/' + self.name + self.image_left_path + str(frame_id).zfill(6) + '.png')
                self._timestamp = self.timestamps[frame_id]
            except:
                print('could not retrieve image: ', frame_id, ' in path ', self.path )
            if frame_id+1 < self.max_frame_id:   
                self._next_timestamp = self.timestamps[frame_id+1]
            else:
                self._next_timestamp = self.timestamps            
        self.is_ok = (img is not None)
        return img 
    """ We only use Mono Vision, so we don't need the right image:
    def getImage1(self, frame_id):
        img = None
        if frame_id < self.max_frame_id:        
            try: 
                img = cv2.imread(self.path + '/sequences/' + self.name + self.image_right_path + str(frame_id).zfill(6) + '.png') 
                self._timestamp = self.timestamps[frame_id]        
            except:
                print('could not retrieve image: ', frame_id, ' in path ', self.path )   
            if frame_id+1 < self.max_frame_id:   
                self._next_timestamp = self.timestamps[frame_id+1]
            else:
                self._next_timestamp = self.timestamps                  
        self.is_ok = (img is not None)        
        return img
    #"""
    #"""
    def getOxts(self, frame_id):

        if frame_id < self.max_frame_id:
            while True:
                next_oxts = next(self.oxts)
                if self._timestamp < next_oxts.packet.sec_since_start:
                    return next_oxts
        else:
            return None
    #"""
# --------------------------------- T U M  --------------------------------- #

class TumDataset(Dataset):
    def __init__(self, path, name, associations, type=DatasetType.TUM): 
        super().__init__(path, name, 30, associations, type)
        self.fps = 30
        print('Processing TUM Sequence')        
        self.base_path=self.path + '/' + self.name + '/'
        associations_file=self.path + '/' + self.name + '/' + associations
        with open(associations_file) as f:
            self.associations = f.readlines()
            self.max_frame_id = len(self.associations)           
        if self.associations is None:
            sys.exit('ERROR while reading associations file!')    

    def getImage(self, frame_id):
        img = None
        if frame_id < self.max_frame_id:
            file = self.base_path + self.associations[frame_id].strip().split()[1]
            img = cv2.imread(file)
            self.is_ok = (img is not None)
            self._timestamp = float(self.associations[frame_id].strip().split()[0])
            if frame_id +1 < self.max_frame_id: 
                self._next_timestamp = float(self.associations[frame_id+1].strip().split()[0])
            else:
                self._next_timestamp = self.timestamps             
        else:
            self.is_ok = False     
            self._timestamp = None                  
        return img 

    def getDepth(self, frame_id):
        img = None
        if frame_id < self.max_frame_id:
            file = self.base_path + self.associations[frame_id].strip().split()[3]
            img = cv2.imread(file)
            self.is_ok = (img is not None)
            self._timestamp = float(self.associations[frame_id].strip().split()[0])
            if frame_id +1 < self.max_frame_id: 
                self._next_timestamp = float(self.associations[frame_id+1].strip().split()[0])
            else:
                self._next_timestamp = self.timestamps               
        else:
            self.is_ok = False      
            self._timestamp = None                       
        return img 
