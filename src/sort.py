"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
from filterpy.kalman import KalmanFilter
np.random.seed(0)

def linear_assignment(cost_matrix):
  try:
    # Use the Jonker-Volgenant algorithm to compute an assignment with minimum costs.
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    # Module for Jonker-Volgenant algrithm could not be found. Use the linear sum assignment from scipy.
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  
  # Computes the Intersection-Over-Union value between two bounding boxes of shape [x1 y1 x2 y2].
  
  # Expand dimensions.
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  # Compute intersection of test and groundtruth box.
  x1_intersection = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  y1_intersection = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  x2_intersection = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  y2_intersection = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w_intersection = np.maximum(0., x2_intersection - x1_intersection)
  h_intersection = np.maximum(0., y2_intersection - y1_intersection)
  intersection = w_intersection * h_intersection
  
  # Compute union of test and groundtruth box.
  w_test = bb_test[..., 2] - bb_test[..., 0]
  h_test = bb_test[..., 3] - bb_test[..., 1]
  area_test = w_test * h_test
  w_gt = bb_gt[..., 2] - bb_gt[..., 0]
  h_gt = bb_gt[..., 3] - bb_gt[..., 1]
  area_gt = w_gt * h_gt
  union = area_test + (area_gt - intersection)
  
  # Compute IOU.
  iou = intersection / union                                        
  
  return (iou)  


def f1_batch(bb_test, bb_gt):
  
  # Expand dimensions.
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  # Compute intersection of test and groundtruth box.
  x1_intersection = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  y1_intersection = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  x2_intersection = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  y2_intersection = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w_intersection = np.maximum(0., x2_intersection - x1_intersection)
  h_intersection = np.maximum(0., y2_intersection - y1_intersection)
  intersection = w_intersection * h_intersection
  
  # Compute area of test box.
  w_test = bb_test[..., 2] - bb_test[..., 0]
  h_test = bb_test[..., 3] - bb_test[..., 1]
  area_test = w_test * h_test
  
  # Compute area of groundtruth box.
  w_gt = bb_gt[..., 2] - bb_gt[..., 0]
  h_gt = bb_gt[..., 3] - bb_gt[..., 1]
  area_gt = w_gt * h_gt
  
  # Compute F1-measure.
  f1 = 2 * intersection / (area_gt + area_test)
  
  return (f1)
  

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  
  # Converts a bounding box of shape [x1 y1 x2 y2] to a measurement vector of the Kalman filter.
  
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
    
  # Converts a state vector of the Kalman filter to a bounding box of shape [x1 y1 x2 y2].
    
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  
  # Static variable to count number of instances and to generate IDs.
  count = 0
  
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    # Instantiate Kalman filter. dim_x denotes the number of components in the state. dim_z denotes
    # the number of components in the measurements.
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    
    # STATE (7x1): [x y s r vx vy vs]
    # x: x-coordinate of center point of bounding box
    # y: y-coordinate of center point of bounding box
    # s: scale of bounding box
    # r: aspect ratio of boundig box
    # vx: velocity of x-coordinate of center point
    # vy: velocity of y-coordinate of center point
    # vs: velocity of scale
    
    # MEASUREMENT (4x1): [x y s r]
    # x: x-coordinate of center point of bounding box
    # y: y-coordinate of center point of bounding box
    # s: scale of bounding box
    # r: aspect ratio of boundig box
    
    # Define constant velocity model for filter.
    self.kf.F = np.array([[1,0,0,0,1,0,0],
                          [0,1,0,0,0,1,0],
                          [0,0,1,0,0,0,1],
                          [0,0,0,1,0,0,0],
                          [0,0,0,0,1,0,0],
                          [0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],
                          [0,1,0,0,0,0,0],
                          [0,0,1,0,0,0,0],
                          [0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    # Set initial state to be the given bounding box.
    self.kf.x[:4] = convert_bbox_to_z(bbox)
  
    # Assign some variables.
    self.time_since_update = 0
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    
    # Assign ID.
    self.id = KalmanBoxTracker.count    
    # Increment instance counter.
    KalmanBoxTracker.count += 1

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    # Reset time since update.
    self.time_since_update = 0
    
    # Reset prediction history.
    self.history = []
    
    # Increment number of hits.
    self.hits += 1
    
    # Increment number of hit streaks.
    self.hit_streak += 1
    
    # Update the Kalman filter with the given measurement.
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    # Returns the current state of the Kalman filter as a bounding box.
    return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, cost_type, cost_threshold):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  # If no trackers exist, return empty matches and unmatched trackers, but return an umatched detection for each given detection.
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  # Compute costs between each given detection and tracker using the given cost type. A matrix of shape MxN is returned
  # where M denotes the number of detections and N the number of trackers. Therefore, each 
  # combination of a detection and a tracker is considered and their costs are computed.
  switcher = {
    'iou' : iou_batch(detections, trackers),
    'f1' : f1_batch(detections, trackers)
  }
  cost_matrix = switcher.get(cost_type)

  # Check whether any costs were computed.
  if min(cost_matrix.shape) > 0:
    # Apply the cost_threshold to the cost_matrix. Returns mask indicating which value of the
    # cost_matrix is greater than (1) or equal/less (0) the cost_threshold.
    a = (cost_matrix > cost_threshold).astype(np.int32)
    
    # Check whether there is only one match for each detection and whether there is only one match for each tracker.
    # Compute matches between detections and trackers.
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        # If so, the matches were perfect. Extract the indices.
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      # If not, match the indices using the Jonker-Volgenant algorithm to minimize the assignment costs.
      matched_indices = linear_assignment(-cost_matrix)
  else:
    # If no costs were computed, then there are no matched indices and therefore no matched detections.
    matched_indices = np.empty(shape=(0,2))

  # Compute unmatched detections: Detections for which no tracker was associated.
  unmatched_detections = []
  for d, det in enumerate(detections):
    # Check if index of current detection is contained in the detection indices of the matched indices.
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  
  # Compute unmatched trackers: Trackers for which no detection was associated.
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    # Check if index of current tracker is contained in the tracker indices of the matched indices.
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  # Filter out matched with low cost.
  matches = []
  for m in matched_indices:
    if(cost_matrix[m[0], m[1]]<cost_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
      
  # If there are no matches, construct default value for matches.
  # Otherwise, concatenate all matches vertically.
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  # Return matches, unmatched detections and unmatched trackers.
  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age, min_hits, cost_type, cost_threshold):
    """
    Sets key parameters for SORT
    """
    
    # Initializes a new SORT instance.
  
    # Maximum age (in frames) of a tracker.
    self.max_age = max_age
    
    # TODO: Find out what that means.
    self.min_hits = min_hits
    
    # Minimum cost to associate a detection to a track.
    self.cost_threshold = cost_threshold
    
    # Type of cost computed for association.
    self.cost_type = cost_type
        
    # Trackers.
    self.trackers = []
    
    # Number of frames already processed.
    self.frame_count = 0

  def update(self, dets=np.empty((0, 5))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    # Update frame count.
    self.frame_count += 1
    
    # Initialize some variables.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    
    # Get predicted locations from already existing trackers.
    for t, trk in enumerate(trks):
      # Get predicted position of current tracker.
      pos = self.trackers[t].predict()[0]
      
      # Assign predicted position to current tracker.
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      
      # If position is invalid, annotate tracker for deletion.
      if np.any(np.isnan(pos)):
        to_del.append(t)
        
    # Filter new tracks. Only involve tracks that contain valid values (non-infinite and non-nan).    
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    
    # If any of the predictions for the existing trackers yielded invalid positions, delete the corresponding tracker.
    for t in reversed(to_del):
      self.trackers.pop(t)
      
    # Try to associate detections for current frame with already existing trackers. Also return unmatched trackers 
    # and unmatched detections.
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.cost_type, self.cost_threshold)

    # Update matched trackers with associated detections.
    for m in matched:
      # Index 0: Index of detection.
      # Index 1: Index of tracker.
      self.trackers[m[1]].update(dets[m[0], :])

    # Create and initialize new trackers for unmatched detections.
    for i in unmatched_dets:
        # Initialize new tracker with current detection.        
        trk = KalmanBoxTracker(dets[i,:])
        
        # Append list of trackers with new tracker.
        self.trackers.append(trk)
        
    # Reverse iterate over trackers and check if they are still alive.
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        # Get current state of current tracker.
        d = trk.get_state()[0]
        
        # TODO: Finde out what is checked here.
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive IDs.
          
        i -= 1
        
        # If tracker is dead, remove it.
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
          
    # Return new tracker states if there are any, otherwise return a default value.
    if(len(ret)>0):
      return np.concatenate(ret)
    else:
      return np.empty((0,5))