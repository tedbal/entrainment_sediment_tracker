# Bedload Entrainment Particle Tracking, Sizing, and PIV functions
# Code written by Ted Balabanski


import cv2 as cv
import numpy as np
import os
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from openpiv.pyprocess import extended_search_area_piv
import pickle
import seaborn as sns
from scipy.stats import gstd, gmean
import statsmodels.formula.api as smf
import pandas as pd


class EntrainmentSedimentTracker:

    """
    ======================================================
    =========== INITIALIZAITION AND SETTING UP ===========
    ======================================================
    """

    def __init__(self, video_paths, max_particles_per_video = int(1e4), debug=False, min_distance=2, feature_params=None, lk_params = None, window_size=64, overlap=16, search_area_size=128):
        # video paths: array of path-like, each path is to a video to be evaluated
        # depth_samples: int, number of points to take piv samples in vertically
        # debug: bool, show debugging information
        # min_distance: int, number of pixels to discriminate different particles

        # video_paths - array of path-like
        if type(video_paths) != np.array:
            video_paths = np.array([os.path.abspath(video_path) for video_path in video_paths])

        # set variables which are used to set the shape
        n_videos = video_paths.shape[0] # for the shape of resulting arrays (tracking, piv, diameters)
        self.max_particles_per_video = max_particles_per_video
        self.min_distance = min_distance

        # set piv variables
        # convert the parameters to openPIV preferred
        self.window_size = window_size
        self.overlap = overlap
        self.search_area_size = search_area_size

        # set class variables
        self.debug = debug
        self.video_paths = video_paths

        # set the frame mask for identifying points
        self.frame_mask = None

        # set the summation variable to zero for total capture length
        total_captures_length = 0

        # validate that they are real video paths and initialize tracking arrays to memory
        for index, video_path in enumerate(self.video_paths):
            try:
                # attempt to open the video and get its length (total frames)
                capture = cv.VideoCapture(video_path)
                capture_length = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

                # add the total length to the running total
                total_captures_length += capture_length
            except Exception as e:
                # remove the video if it does not open
                self.video_paths.pop(index)

        # initialize the main resulting arrays
        self.particle_counter = 0
        self.tracking_array = np.zeros((total_captures_length, 2, n_videos*max_particles_per_video))
        self.piv_result = None
        self.diameters = np.zeros((n_videos*max_particles_per_video, 1))

        # initialize the tracking parameters
        self.feature_params = dict(maxCorners = self.max_particles_per_video, qualityLevel = 0.3, minDistance = 7, blockSize = 7) if feature_params == None else feature_params
        self.lk_params = dict(winSize  = (15, 15), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)) if lk_params == None else lk_params

        if self.debug:
            # set random colors for plotting
            self._rand_colors =  np.random.randint(0, 255, (self.max_particles_per_video, 3))

            # print debugging statment about parsed videos
            print(f"successfully parsed {self.video_paths.shape[0]} videos from {video_paths.shape[0]} ({100*self.video_paths.shape[0]/video_paths.shape[0]:.2f}%)")



    def set_focus_thresholds(self, focus_thresholds=None):
        # if the focus_thresholds are an integer, set the bedheight constant
        if type(focus_thresholds) == int:
            self.focus_thresholds = focus_thresholds * np.ones(self.video_paths.shape)
            return None
        
        # if it is an array, make sure it has consistent shape with the video paths
        elif type(focus_thresholds) == list:
            if len(focus_thresholds) != self.video_paths.shape[0]:
                raise TypeError("When passed as a list, focus_thresholds shape must be consistent with video paths")
            self.focus_thresholds = np.array(focus_thresholds)
            return None

        elif type(focus_thresholds) == np.array:
            if focus_thresholds.shape != self.video_paths:
                raise TypeError("When passed as an np-array, focus_thresholds shape must be consistent with video paths")
            self.focus_thresholds = focus_thresholds
            return None
        
        else:
            raise TypeError("Input is not int, list, or np array")



    def set_bedheights(self, bedheights=None):
        # rotate through the videos in video_paths and ask for input to 
        # TODO: write this function
        # if the bedheights are an integer, set the bedheight constant
        if type(bedheights) == int:
            self.bedheights = bedheights * np.ones(self.video_paths.shape)
            return None
        
        # if it is an array, make sure it has consistent shape with the video paths
        elif type(bedheights) == list:
            if len(bedheights) != self.video_paths.shape[0]:
                raise TypeError("When passed as a list, bedheights shape must be consistent with video paths")
            self.bedheights = np.array(bedheights)
            return None

        elif type(bedheights) == np.array:
            if bedheights.shape != self.video_paths:
                raise TypeError("When passed as an np-array, bedheights shape must be consistent with video paths")
            self.bedheights = bedheights
            return None

        # initialize the bedheights property
        self.bedheights = np.zeros(self.video_paths.shape)

        # iterate over the videos
        for index, video in enumerate(self.video_paths):
            # TODO: write the main loop in set_bedheigts
            # 1. read the first frame
            # 2. display it
            # 3. query the user input on the bedheight
            # 4. update the bedheight attribute
            continue

        return None
        


    def initialize_particle(self, position, frame):
        # don't allow the particles to go over the max count
        if self.particle_counter == self.max_particles_per_video - 1:
            return None
        
        # initialize the new particle and increment the particle counter
        particle_id = self.particle_counter
        self.particle_counter += 1

        # otsu thresholdhing after gaussian blur
        # blur = cv.GaussianBlur(frame, (5,5), 0)
        # thresh = cv.adaptiveThreshold(frame, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 315, 1)
        # focus_threshold = self.focus_thresholds[self.current_video_index]
        _, thresh = cv.threshold(frame, 200, 255, cv.THRESH_BINARY_INV)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # only select the bottommost contours
        my_cool_var = hierarchy[..., 2] == -1
        no_child_indices = np.nonzero(hierarchy[..., 2] == -1)[1]
        contours = tuple([contours[idx] for idx in no_child_indices])

        # add the contours if debugging
        if self.debug:
            self.contours = contours
        
        # parse the x, y from the position
        x, y = position.ravel()

        # validate that the point is below the bedheight
        # in the video this means that the y-position is greater than the bedheight
        if y >= self.bedheights[self.current_video_index]:
            self.particle_counter -= 1
            return None
        
        # create the image gradient to set the focus threshold
        sobel_x = cv.Sobel(frame, cv.CV_64F, 1, 0, ksize=3)
        sobel_y = cv.Sobel(frame, cv.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # find the contour that contains the position
        for contour in contours[1:]: # remove the first contour (the largest one which is the bed)
            # need at least 5 points to fit an ellipse
            if contour.shape[0] < 5:
                continue

            # get the contours coordinates as an array
            contour_points = np.vstack(contours[1:]).squeeze()
            x_values, y_values = contour_points[:, 0], contour_points[:, 1]
            gradient_magnitude_along_contour = gradient_magnitude[y_values, x_values]

            # the two conditions for initializing the point
            location_condition = cv.pointPolygonTest(contour, (x, y), False) >= 0
            focus_condition = np.max(gradient_magnitude_along_contour) >= self.focus_thresholds[self.current_video_index]
            
            # check if the position lies within or on the contour
            if location_condition and focus_condition:
                # fit an ellipse to the contour
                ellipse = cv.fitEllipse(contour)

                # extract the major and minor axes of the ellipse
                major_axis, minor_axis = ellipse[1]

                # calculate diameter (average of major and minor axes) and add it to the diameters array
                diameter = (major_axis + minor_axis) / 2

                self.diameters[particle_id] = diameter
            
                return particle_id

        # if the location and focus threshold is met for no contour, do not initialize the point
        self.particle_counter -= 1
        return None
    


    def initialize_points(self, frame):
        # initialize the mask and make the leftmost third to 255
        mask = np.zeros_like(frame)
        mask[:, 0:mask.shape[1]//3] = 255 # sets the leftmost third to 255

        # add the frame mask and clip to 255
        mask = mask + self.frame_mask
        mask.clip(0, 255).astype('uint8')

        # get good features to track
        points = cv.goodFeaturesToTrack(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), 
                                        mask=mask[..., 0], 
                                        maxCorners=self.max_points, 
                                        qualityLevel=0.3, 
                                        minDistance=self.min_distance)

        return points
    


    def set_focus_mask(self, frame):
        # sets the frame mask to be the focus threshold
        pass



    """
    ==========================================================
    =========== DATA PRE-PROCESSING AND COLLECTION ===========
    ==========================================================
    """


    def evaluate(self):
        # get the total number of videos
        total_videos = self.video_paths.size

        # iterate through every video
        for index, video_path in enumerate(self.video_paths):
            # display progress
            print("evaluating video {} of {} ({:.0f}%)".format(index + 1, total_videos, 100*(index + 1)/total_videos))

            # set the current video index
            self.current_video_index = index

            # evaluate the video
            self._evaluate_single_video(video_path,)



    def save(self, save_filepath):
        # TODO: make the saving more robust and include a load function
        fi = open(save_filepath, 'wb')
        pickle.dump(self, fi)
        fi.close()



    def _show_tracking_frame(self, frame, frame_index):
        # create a copy of the frame
        drawing_frame = frame.copy()

        # iterate through each particle
        for particle_id in range(self.tracking_array.shape[2]):
            # parse out the x and y value
            x, y = self.tracking_array[frame_index, :, particle_id].ravel()
            
            # ignore unitialized points
            if x == 0 and y == 0:
                continue

            # draw the point
            drawing_frame = cv.circle(drawing_frame, (int(x), int(y)), 5, (255, 0, 0), -1)

        # draw the bed
        bedheight = int(self.bedheights[self.current_video_index])
        cv.line(drawing_frame, (0, bedheight), (frame.shape[1] - 1, bedheight), (0, 0, 255), 2)

        # draw the contours
        cv.drawContours(drawing_frame, self.contours, -1, (0, 255, 0), 2)

        # create the image to draw, display it, and wait for input
        cv.imshow(f'frame {frame_index}', drawing_frame)
        cv.waitKey(0)
        cv.destroyAllWindows()



    def _evaluate_single_video(self, video_path, frame_average=10):
        # TODO: write this function
        # gathers and process the main data for this experiment
        # video_path - string or path-like, corresponding to the video
        # frame_average - int, number of frames over which to create a moving average

        # open the video
        capture = cv.VideoCapture(video_path)
        capture_length = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

        # read the frame and make it grayscale
        ret, old_frame = capture.read()
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

        # initialize the points and their ids
        p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **self.feature_params)
        p0_ids, p0_idx = self._initialize_array_of_particles(p0, old_gray)
        p0 = p0[p0_idx, ...]

        # define the piv result dimenisons and initialize the result
        # TODO: update this into a better way to get the shape?
        bedheight_index = int(self.bedheights[self.current_video_index])
        u_for_shape, _, _ = extended_search_area_piv(old_gray[:, :bedheight_index], old_gray[:, :bedheight_index], window_size=self.window_size, overlap=self.overlap, dt=1, search_area_size=self.search_area_size)
        depth_samples = u_for_shape.shape[0]
        self.piv_result = np.zeros((depth_samples, 2, capture_length - 2))

        # iterate over the frames:
        for frame_index in tqdm(range(capture_length - 1)):
            # read the frame and convert to grayscale
            ret, frame = capture.read()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # track the particles in the frame
            p0, p0_ids = self.track_particles_in_frame(old_gray, frame_gray, p0, p0_ids, frame_index)

            # perform piv
            self.two_frame_piv(old_gray, frame_gray, frame_index)

            # update the old gray frame
            old_gray = frame_gray

            # if debug, show all the tracking points on the frame
            if self.debug:
                self._show_tracking_frame(frame, frame_index)



    def _add_points_to_tracking_array(self, points, frame_index, point_ids):
        self.tracking_array[frame_index, :, point_ids.T] = points.reshape((points.shape[0], 2))



    def _get_new_features_to_track(self, old_gray, p0_old):
        # identify the all the trackable points
        points_identified = cv.goodFeaturesToTrack(old_gray, mask = None, **self.feature_params)

        # get the minimum x-distance of any point
        x_min = np.min(p0_old[..., 0]) if type(p0_old) != type(None) else 1e99 # check if p0 is even a vector

        # the new points are those that have a x-value less than x_min
        new_points = points_identified[points_identified[..., 0] < x_min - self.min_distance].reshape((-1, 1, 2))

        return new_points
    


    def _initialize_array_of_particles(self, particles, frame_gray):
        # initialize resulting list of particle ids
        # and particle indices for indexing
        particle_ids = []
        particle_indices = []

        # iterate through the particles in the array
        for particle_index in range(particles.shape[0]):
            # validate the intialized particle and get its id (if valid)
            particle_id = self.initialize_particle(particles[particle_index, :], frame_gray)

            # if validation is succesful, add the particle's id to the list
            if particle_id != None:
                particle_ids.append(particle_id)
                particle_indices.append(particle_index)
        
        # return a np arrays with proper shape
        particle_id_array = np.array(particle_ids, dtype=np.int32).reshape(-1, 1)
        particle_idx_array = np.array(particle_indices, dtype=np.int32)
        
        return particle_id_array, particle_idx_array 



    def track_particles_in_frame(self, old_gray, frame_gray, p0, p0_ids, frame_index):
        # TODO: write this function 
        # if p0 is empty, initialize new points, otherwise calculate optical flow from prev frame
        if type(p0) != type(None):
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **self.lk_params)
        else:
            # get the new particles
            good_new = self._get_new_features_to_track(old_gray, p0)

            # initialize the particles
            good_new_ids, good_new_idx = self._initialize_array_of_particles(good_new, frame_gray)

            # get only the validated particles
            good_new = good_new[good_new_idx, ...]

            # check if any points were validated
            if good_new.size == 0:
                return None, None

            # add the points to the tracking array
            self._add_points_to_tracking_array(good_new, frame_index, good_new_ids)

            return good_new, good_new_ids

        # select good points:
        if p1 is not None:
            # continue tracking the old particles with st = 1 and find new particles to track
            good_old = p0[st==1].reshape(-1, 1, 2)
            good_new = self._get_new_features_to_track(old_gray, good_old)

            # get the ids for the old ones and initalize the new ones
            good_old_ids = p0_ids[st==1]
            good_new_ids, good_new_idx = self._initialize_array_of_particles(good_new, frame_gray)

            # get only the validated particles
            good_new = good_new[good_new_idx, ...]

            # check if any points were validated
            if good_new.size == 0:
                return None, None

            # add the old and the new points to the tracking array
            self._add_points_to_tracking_array(good_old, frame_index, good_old_ids)
            self._add_points_to_tracking_array(good_new, frame_index, good_new_ids)

            # combine the new and old results
            combined_points = np.concatenate((good_new, good_old), axis=0)
            combined_points_ids = np.concatenate((good_old_ids, good_new_ids[:, 0]), axis=0).reshape(-1, 1)

            return combined_points, combined_points_ids
        
        else:
            return None, None
    


    def two_frame_piv(self, frame1, frame2, frame2_index):
        # TODO: write the piv step to take in the last 10 frames and calculate the 
        # velocities at the specified slice intervals

        # parse the bedheight
        bedheight_index = int(self.bedheights[self.current_video_index])

        # perform piv
        u, v, sig2noise = extended_search_area_piv(frame1[:, :bedheight_index], 
                                           frame2[:, :bedheight_index], 
                                           window_size = self.window_size,
                                           overlap = self.overlap,
                                           search_area_size = self.search_area_size,
                                           dt = 1)
        
        # average over the x-direction to get an average profile for the frame
        u_average = np.nanmean(u, axis=1)
        v_average = np.nanmean(v, axis=1)
        
        # place the results in the piv array
        self.piv_result[:, 0, frame2_index - 1] = u_average
        self.piv_result[:, 1, frame2_index - 1] = v_average

        if self.debug:
            print("mean sig2noise: {:.2f}".format(np.nanmean(sig2noise)))

    

    """
    ===========================================================
    ================== DATA POST-PROCESSING  ==================
    ===========================================================
    """

    def piv_smoothing(self, n_frames=10):
        # n_frames: int, number of frames over which to take a moving average
        # define the kernel
        kernel = np.ones((n_frames)) / n_frames

        # define variables for shape
        M = self.piv_result.shape[2]
        N = n_frames

        # initialize the smooth_piv array
        smoothed_piv = np.zeros((self.piv_result.shape[0], 2, max(M, N) - min(M, N) + 1))

        for i in range(self.piv_result.shape[0] - 1):
            # perform the moving average
            smoothed_piv[i, 0, :] = np.convolve(self.piv_result[i, 0, :], kernel, 'valid')
            smoothed_piv[i, 1, :] = np.convolve(self.piv_result[i, 1, :], kernel, 'valid')

        self.smoothed_piv = smoothed_piv
        self.smoothed_piv[self.smoothed_piv == 0] = np.nan
        self.n_frames = n_frames



    def entrained(self, factor=0.5):
        # TODO: write this function
        # gives a logical array of whether or not particles were entrained
        # tracking_array - (F, 2, nmax) where F - number of frames in video;
        #                  the x, y positions of the particles tracked
        # diameters - (nmax) array where each entry is the particle corresponding to the 3rd
        # axis of tracking_array and yields the particle diameter in pixels
        # returns:
        # entrained_array - (nmax) logical array where each entry is the particle corresponding to the
        # 3rd axis of tracking array and yields its entrained status over the video clip

        # index the vertical positions of the particles from the tracking array
        vertical_positions = self.tracking_array[:, 1, :].T
        vertical_positions[vertical_positions == 0] = np.nan

        # calculate the total maximum displacement vertically
        max_displacement = np.nanmax(vertical_positions, axis=1) - np.nanmin(vertical_positions, axis=1)
        max_displacement = max_displacement.reshape(-1, 1)

        # considered entrained when the maximum displacement is factor*diameter or greater
        entrained = np.less_equal(factor*self.diameters, max_displacement) # factor*diameter <= max_dispalcemenent

        # set as class variable
        self.entrained = entrained



    def get_u_bed(self, n):
        # TODO: write this function
        # returns the mean bed-load transport speed for a particle n in the third axis of tracking_array
        # tracking_array - (F, 2, nmax) where F - number of frames in video;
        #                  the x, y positions of the particles tracked
        # n - int; corresponding to the index of tracking_array for the particle whose velocity is being taken
        # returns:
        # u_bed - double; mean bed-load transport speed in pix/frame

        pass



    def set_scale(self, spatial, temporal):
        # spatial: array, spatial scale corresponding to each video in videos (i.e. cm/pix)
        # temporal: array, temporal scale corresponding to each video in videos (i.e. s/frame, one over framerate)

        # convert to np array if list
        if type(spatial) == list:
            spatial = np.array(spatial)

        if type(temporal) == list:
            temporal = np.array(temporal)

        if spatial.size != self.video_paths.size or temporal.size != self.video_paths.size:
            raise ValueError("the scales must be the same size as the video paths")
        
        self.spatial_conversion_factors = spatial
        self.temporal_conversion_factors = temporal



    """
    =====================================================
    =========== STATISTICS AND VISUALIZATIONS ===========
    =====================================================
    """

    def plot_entrainment_temporal_density(self):
        # set the nans to zero for easier use
        self.tracking_array[self.tracking_array == np.nan] = 0
        entrained_trajectories = self.tracking_array[:, 1, self.entrained.reshape(-1)]

        # define the array of the last seen frame
        last_seen_frames = np.zeros((self.tracking_array.shape[0]))

        # iterate through the particles
        for i in range(entrained_trajectories.shape[1] - 1):
            # get the last seen frame and put it in the array
            nonzero_indicies = np.nonzero(entrained_trajectories[:, i] > 10)[0]
            last_seen_frames[i] = nonzero_indicies[-1]

        # set the tracking array back
        self.tracking_array[self.tracking_array == 0] = np.nan

        # convert frames to seconds
        time_conversion_factor = self.temporal_conversion_factors[self.current_video_index]
        last_seen = last_seen_frames * time_conversion_factor

        # set the text parameters
        matplotlib.rcParams["text.usetex"] = True
        matplotlib.rc('font', **{'family': 'sans-serif', 'size': 18})
        plt.rcParams["figure.figsize"] = (8,6)

        # write the plot labels
        plt.xlabel("time (s)")
        plt.xlim(0, self.tracking_array.shape[0]*time_conversion_factor)
        plt.ylabel("Density")
        plt.title("Temporal distribution of entrained particles")

        # color
        entrained_color = (51/255, 85/255, 1, 0.6)

        # plot
        sns.kdeplot(last_seen, 
                    label="Entrained", 
                    color = entrained_color, 
                    fill = True)
        
        plt.show()



    def plot_bedload_shear(self, savepath=None):
        # plots the 
        # get the mean shear depth profile
        # smoothed_piv = (num_slices, 2, # of frames)
        conversion_factor = self.spatial_conversion_factors[self.current_video_index] * self.temporal_conversion_factors[self.current_video_index]
        mean_u_profile = conversion_factor * np.nanmean(self.smoothed_piv, axis = 2)[:, 0]

        # get the size of the windows for plotting
        capture = cv.VideoCapture(self.video_paths[self.current_video_index])
        height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        depth_slices = np.linspace(0, height, mean_u_profile.shape[0])        

        # get the time items
        time_scale = self.temporal_conversion_factors[self.current_video_index]
        smoothed_times = np.linspace(0, self.smoothed_piv.shape[2], self.smoothed_piv.shape[2])
        # add the original time from the boundary removal due to smoothing
        times = time_scale * self.n_frames * np.ones_like(smoothed_times) + smoothed_times
        mean_u = np.nanmean(self.smoothed_piv, axis=0)[0, :]

        # text and figure parameters
        matplotlib.rcParams["text.usetex"] = True
        matplotlib.rc('font', **{'family': 'sans-serif', 'size': 18})

        # initialize the plot
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(12, 8)

        # format the mean velocity profile
        mean_profile_df = pd.DataFrame({'depth': depth_slices, 'velocity': mean_u_profile})
        sns.lineplot(data = mean_profile_df, x = 'velocity', y = 'depth', ax = ax[0])
        ax[0].set_xlabel("Velocity, $u$ (cm s$^{-1}$)")
        ax[0].set_ylabel("Depth (cm)")
        ax[0].set_title("Mean velocity profile")

        # format the mean velocity time distribution
        velocity_t_df = pd.DataFrame({'time': times, 'velocity': mean_u})
        sns.lineplot(data = velocity_t_df, x = 'time', y = 'velocity', ax = ax[1])
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Mean velocity, $u$ (cm s$^{-1}$)")
        ax[1].set_title("Mean velocity ")

        plt.show()
        


    def plot_entrainment_velocity_by_diameter(self, savepath=None):
        # get the tracking path of only the entrained particles
        entrained_trajectories_y = self.tracking_array[:, 1, self.entrained.reshape(-1)]
        entrained_trajectories_y[entrained_trajectories_y == np.nan] = 0

        # calculate the differences to get velocities
        entrained_u_nondim = np.diff(entrained_trajectories_y, axis=0)

        # get the scaling factors and scale
        spatial_scaling_factor = self.spatial_conversion_factors[self.current_video_index]
        spatial_scaling_factor *= 1e4 # cm -> um
        temporal_scaling_factor = self.temporal_conversion_factors[self.current_video_index]
        entrained_u = spatial_scaling_factor * temporal_scaling_factor * entrained_u_nondim
        entrained_u[entrained_u == 0] = np.nan
        entrained_u_mean = np.abs(np.nanmean(entrained_u, axis=0))

        # get the entrained diameters
        entrained_diameters = spatial_scaling_factor * self.diameters[self.entrained]

        # bin the velocities by the diameter
        diameter_bins = np.linspace(0, 250, 5)
        diameter_bins_midpoint = np.linspace(25, 225, 5)
        binned_velocities = np.zeros_like(diameter_bins)
        binned_velocity_errors = np.zeros_like(diameter_bins)

        # iterate through the bin diameters
        for i in range(diameter_bins.size - 1):
            # create the bin mask and get the velocity in bin
            bin_mask = (entrained_diameters >= diameter_bins[i]) & (entrained_diameters < diameter_bins[i + 1])
            velocities_in_bin = entrained_u_mean[bin_mask]

            # perform statistics on the velocities in the bin
            mean_velocity = np.mean(velocities_in_bin)
            velocity_std = np.std(velocities_in_bin)
            binned_velocities[i] = mean_velocity
            binned_velocity_errors[i] = velocity_std

        # fit a linear regression to the diameter
        data_dict = {'diameter': diameter_bins_midpoint, 'u': binned_velocities}
        model = smf.ols('u ~ diameter', data=data_dict).fit()

        # extract the model parameters
        intercept = model.params['Intercept']
        slope = model.params['diameter']
        rsquared = model.rsquared
        interecept_se = model.bse['Intercept']
        slope_se = model.bse['diameter']

        # format text
        model_text = "\n".join(["$u = m d_0 + b$", 
                                f"$m = {slope:.2e} }} \pm {slope_se:.2e} }}$",
                                f"$b = {intercept:.2f} \pm {interecept_se:.2f}$"])
        model_text = model_text.replace("e", "\\times 10^{")
        
        # text and figure parameters
        matplotlib.rcParams["text.usetex"] = True
        matplotlib.rc('font', **{'family': 'sans-serif', 'size': 18})
        plt.rcParams["figure.figsize"] = (8,6)

        # place text
        plt.text(0, 0.75, 
                 model_text, 
                 bbox=dict(ec=(0, 0, 0, 1), boxstyle='square', fc=(1, 1, 1, 1)))

        # plot the data and model
        entrained_color_transp = (51/255, 85/255, 1, 0.6)
        entrained_color = (51/255, 85/255, 1, 1)
        model_x = np.linspace(0, 250, 10)
        model_y = model.get_prediction(exog={'diameter': model_x}).predicted
        
        model_data = pd.DataFrame({'x': model_x, 'y': model_y})
        sns.lineplot(data = model_data, x = 'x', y = 'y', color = (0, 0, 0, 1))
        
        exp_data = pd.DataFrame(data_dict)
        sns.scatterplot(data = exp_data, x = 'diameter', y = 'u', color = entrained_color_transp)

        # plot the errorbars
        plt.errorbar(diameter_bins_midpoint, 
                     binned_velocities, 
                     fmt = '.',
                     yerr = binned_velocity_errors,
                     ecolor=entrained_color_transp,
                     capsize=10)

        # add plot labels and titles
        plt.xscale('linear')
        plt.xlabel(r"Diameter, $d_0$ ($\mu$m)")
        plt.ylabel(r"Vertical velocity, $v$ ($\mu$m s$^{-1}$)")
        plt.ylim(0, 1)
        plt.title(r"Vertical velocity of mud flocculates by diameter")
        
        if savepath != None:
            plt.savefig(savepath)

        if self.debug:
            plt.show()

        

    def plot_entrainment_distributions(self, savepath=None):
        # plots the entrained and non-entrained histograms and the
        # corresponding t-statistic and p-value
        # TODO: write this plotting (histogram) function

        try:
            # get the spatial factors
            conversion_factor = self.spatial_conversion_factors[self.current_video_index]
            conversion_factor *= 1e4 # cm to um
        except Exception as e:
            conversion_factor = 1

        # parse out the diameters for the two distributions and convert to cm
        entrained_diameters = conversion_factor * self.diameters[self.entrained]
        non_entrained_diameters = conversion_factor * self.diameters[~self.entrained]
        non_entrained_diameters = np.delete(non_entrained_diameters, np.where(non_entrained_diameters >= 0.5*1e3))
        non_entrained_diameters[non_entrained_diameters == 0] = np.nan

        # get statistics on the diameters
        entrained_mean = gmean(entrained_diameters, nan_policy='omit', axis=None)
        non_entrained_mean = gmean(non_entrained_diameters, nan_policy='omit', axis=None)
        entrained_stdv = gstd(entrained_diameters[~np.isnan(entrained_diameters)])
        non_entrained_stdv = gstd(non_entrained_diameters[~np.isnan(non_entrained_diameters)])
        entrained_n = np.count_nonzero(~np.isnan(entrained_diameters))
        non_entrained_n = np.count_nonzero(~np.isnan(non_entrained_diameters))

        # define the colors (R, G, B, A)
        entrained_color = (51/255, 85/255, 1, 0.6)
        non_entrained_color = (255/255, 0, 43/255, 0.6)

        # text and figure parameters
        matplotlib.rcParams["text.usetex"] = True
        matplotlib.rc('font', **{'family': 'sans-serif', 'size': 18})
        plt.rcParams["figure.figsize"] = (8,6)

        # format and draw the text
        entrained_text = f"Entrained\n$\mu = {entrained_mean:.1f} \, \mu m$\n$\sigma = {entrained_stdv:.2f}$\n$n = {entrained_n:,}$" 
        non_entrained_text = f"Not entrained\n$\mu = {non_entrained_mean:.1f} \, \mu m$\n$\sigma = {non_entrained_stdv:.2f}$\n$n = {non_entrained_n:,}$" 
        plt.text(1.2, 0.0175, entrained_text, bbox=dict(ec=entrained_color, boxstyle='square', fc=(1, 1, 1, 1)))
        plt.text(1.2, 0.011, non_entrained_text, bbox=dict(ec=non_entrained_color, boxstyle='square', fc=(1, 1, 1, 1)))

        # plot the distributions
        sns.kdeplot(entrained_diameters, 
                    label="Entrained", 
                    color = entrained_color, 
                    fill = True)
        
        sns.kdeplot(non_entrained_diameters,  
                    label="Not entrained", 
                    color = non_entrained_color, 
                    fill = True)

        # format the plot
        plt.xlabel('Diameter ($\mu$m)' if conversion_factor != 1 else "diameter (px)")
        plt.ylabel('Density')
        plt.xticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
        plt.xlim(1, 1000)
        plt.tick_params(axis='x', which='minor', bottom=False, top=False)
        plt.title('Particle diameter distributions')
        plt.xscale('log')
        plt.legend()

        # show the plot and save it if appropriate
        if savepath != None:
            plt.savefig(savepath)



def load(load_filepath):
    # open the file and load the tracker
    fi = open(load_filepath, 'rb')
    tracker = pickle.load(fi)
    fi.close()

    return tracker



if __name__ == "__main__":
    from_scratch = True

    if from_scratch:
        filepath = os.path.abspath(r"C:\Users\Jobim\Desktop\school 23-24\research\strom\code\Entrainment\video\Day2-03-Ripup-Laser.mp4")
        
        # set up the sediment tracker
        sediment_tracker = EntrainmentSedimentTracker([filepath], debug=False)
        sediment_tracker.set_bedheights(bedheights=460)
        sediment_tracker.set_focus_thresholds(50)

        # evaluate
        sediment_tracker.evaluate()

        # save
        sediment_tracker.save("D:/tracker/tracker_piv.sav")

    else:
        sediment_tracker = load("D:/tracker/tracker2.sav")

    # set the spatial and temporal scale
    sediment_tracker.set_scale([1/1280], [1/400])

    # post-processing
    sediment_tracker.entrained()
    sediment_tracker.piv_smoothing()

    # plot the shear profile
    sediment_tracker.plot_bedload_shear()

    # plot the entrainment temporal density
    # sediment_tracker.plot_entrainment_temporal_density()

    # plot the shear profile
    # sediment_tracker.plot_entrainment_velocity_by_diameter('./diameter_vel_function.png')

    # plot the entrainment distribution
    # sediment_tracker.plot_entrainment_distributions('./entrainment_diams_dist_fig.png')