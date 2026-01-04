"""
Cardiac utils - pytorch - CMRxReconAll
Created on 2025/06/30
@author: Zi Wang
Modified from Wenjia Bai's code (https://github.com/baiwenjia/ukbb_cardiac)
Email: Zi Wang (zi.wang@imperial.ac.uk)
If you want to use this code, please cite our relevant papers in the GitHub page.
"""

import nibabel as nib
import numpy as np
import scipy.ndimage.measurements as measure
import cv2
import vtk
import pandas as pd
from vtk.util import numpy_support
import math
from scipy import interpolate
import skimage
import skimage.measure


def sa_pass_quality_control(seg_sa_name):
    """ Quality control for short-axis image segmentation """
    nim = nib.load(seg_sa_name)
    seg_sa = np.asanyarray(nim.dataobj)
    X, Y, Z = seg_sa.shape[:3]

    # Label class in the segmentation
    label = {'LV': 1, 'Myo': 2, 'RV': 3}

    # Criterion 1: every class exists and the area is above a threshold
    # Count number of pixels in 3D
    for l_name, l in label.items():
        pixel_thres = 10
        if np.sum(seg_sa == l) < pixel_thres:
            print('{0}: The segmentation for class {1} is smaller than {2} pixels. '
                  'It does not pass the quality control.'.format(seg_sa_name, l_name, pixel_thres))
            return False

    # Criterion 2: number of slices with LV segmentations is above a threshold
    # and there is no missing segmentation in between the slices
    z_pos = []
    for z in range(Z):
        seg_z = seg_sa[:, :, z]
        endo = (seg_z == label['LV']).astype(np.uint8)
        myo = (seg_z == label['Myo']).astype(np.uint8)
        pixel_thres = 10
        if (np.sum(endo) < pixel_thres) or (np.sum(myo) < pixel_thres):
            continue
        z_pos += [z]
    n_slice = len(z_pos)
    slice_thres = 6
    if n_slice < slice_thres:
        print('{0}: The segmentation has less than {1} slices. '
              'It does not pass the quality control.'.format(seg_sa_name, slice_thres))
        return False

    if n_slice != (np.max(z_pos) - np.min(z_pos) + 1):
        print('{0}: There is missing segmentation between the slices. '
              'It does not pass the quality control.'.format(seg_sa_name))
        return False

    # Criterion 3: LV and RV exists on the mid-cavity slice
    _, _, cz = [np.mean(x) for x in np.nonzero(seg_sa == label['LV'])]
    z = int(round(cz))
    seg_z = seg_sa[:, :, z]

    endo = (seg_z == label['LV']).astype(np.uint8)
    endo = get_largest_cc(endo).astype(np.uint8)
    myo = (seg_z == label['Myo']).astype(np.uint8)
    myo = remove_small_cc(myo).astype(np.uint8)
    epi = (endo | myo).astype(np.uint8)
    epi = get_largest_cc(epi).astype(np.uint8)
    rv = (seg_z == label['RV']).astype(np.uint8)
    rv = get_largest_cc(rv).astype(np.uint8)
    pixel_thres = 10
    if np.sum(epi) < pixel_thres or np.sum(rv) < pixel_thres:
        print('{0}: Can not find LV epi or RV to determine the AHA '
              'coordinate system.'.format(seg_sa_name))
        return False
    return True


def get_largest_cc(binary):
    """ Get the largest connected component in the foreground. """
    cc, n_cc = measure.label(binary)
    max_n = -1
    max_area = 0
    for n in range(1, n_cc + 1):
        area = np.sum(cc == n)
        if area > max_area:
            max_area = area
            max_n = n
    largest_cc = (cc == max_n)
    return largest_cc


def remove_small_cc(binary, thres=10):
    """ Remove small connected component in the foreground. """
    cc, n_cc = measure.label(binary)
    binary2 = np.copy(binary)
    for n in range(1, n_cc + 1):
        area = np.sum(cc == n)
        if area < thres:
            binary2[cc == n] = 0
    return binary2


def evaluate_wall_thickness(seg_name, output_name_stem, part=None):
    """ Evaluate myocardial wall thickness. """
    # Read the segmentation image
    nim = nib.load(seg_name)
    Z = nim.header['dim'][3]
    affine = nim.affine
    seg = np.asanyarray(nim.dataobj)

    # Label class in the segmentation
    label = {'BG': 0, 'LV': 1, 'Myo': 2, 'RV': 3}

    # Determine the AHA coordinate system using the mid-cavity slice
    aha_axis = determine_aha_coordinate_system(seg, affine)

    # Determine the AHA part of each slice
    part_z = {}
    if not part:
        part_z = determine_aha_part(seg, affine)
    else:
        part_z = {z: part for z in range(Z)}

    # Construct the points set to represent the endocardial contours
    endo_points = vtk.vtkPoints()
    thickness = vtk.vtkDoubleArray()
    thickness.SetName('Thickness')
    points_aha = vtk.vtkIntArray()
    points_aha.SetName('Segment ID')
    point_id = 0
    lines = vtk.vtkCellArray()

    # Save epicardial contour for debug and demonstration purposes
    # save_epi_contour = False
    save_epi_contour = True
    if save_epi_contour:
        epi_points = vtk.vtkPoints()
        points_epi_aha = vtk.vtkIntArray()
        points_epi_aha.SetName('Segment ID')
        point_epi_id = 0
        lines_epi = vtk.vtkCellArray()

    # For each slice
    for z in range(Z):
        # Check whether there is endocardial segmentation and it is not too small,
        # e.g. a single pixel, which either means the structure is missing or
        # causes problem in contour interpolation.
        seg_z = seg[:, :, z]
        endo = (seg_z == label['LV']).astype(np.uint8)
        endo = get_largest_cc(endo).astype(np.uint8)
        myo = (seg_z == label['Myo']).astype(np.uint8)
        myo = remove_small_cc(myo).astype(np.uint8)
        epi = (endo | myo).astype(np.uint8)
        epi = get_largest_cc(epi).astype(np.uint8)
        pixel_thres = 10
        if (np.sum(endo) < pixel_thres) or (np.sum(myo) < pixel_thres):
            continue

        # Calculate the centre of the LV cavity
        # Get the largest component in case we have a bad segmentation
        cx, cy = [np.mean(x) for x in np.nonzero(endo)]
        lv_centre = np.dot(affine, np.array([cx, cy, z, 1]))[:3]

        # Extract endocardial contour
        # Note: cv2 considers an input image as a Y x X array, which is different
        # from nibabel which assumes a X x Y array.
        contours, _ = cv2.findContours(cv2.inRange(endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        endo_contour = contours[0][:, 0, :]

        # Extract epicardial contour
        contours, _ = cv2.findContours(cv2.inRange(epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        epi_contour = contours[0][:, 0, :]

        # Smooth the contours
        endo_contour = approximate_contour(endo_contour, periodic=True)
        epi_contour = approximate_contour(epi_contour, periodic=True)

        # A polydata representation of the epicardial contour
        epi_points_z = vtk.vtkPoints()
        for y, x in epi_contour:
            p = np.dot(affine, np.array([x, y, z, 1]))[:3]
            epi_points_z.InsertNextPoint(p)
        epi_poly_z = vtk.vtkPolyData()
        epi_poly_z.SetPoints(epi_points_z)

        # Point locator for the epicardial contour
        locator = vtk.vtkPointLocator()
        locator.SetDataSet(epi_poly_z)
        locator.BuildLocator()

        # For each point on endocardium, find the closest point on epicardium
        N = endo_contour.shape[0]
        for i in range(N):
            y, x = endo_contour[i]

            # The world coordinate of this point
            p = np.dot(affine, np.array([x, y, z, 1]))[:3]
            endo_points.InsertNextPoint(p)

            # The closest epicardial point
            q = np.array(epi_points_z.GetPoint(locator.FindClosestPoint(p)))

            # The distance from endo to epi
            dist_pq = np.linalg.norm(q - p)

            # Add the point data
            thickness.InsertNextTuple1(dist_pq)
            seg_id = determine_aha_segment_id(p, lv_centre, aha_axis, part_z[z])
            points_aha.InsertNextTuple1(seg_id)

            # Record the first point of the current contour
            if i == 0:
                contour_start_id = point_id

            # Add the line
            if i == (N - 1):
                lines.InsertNextCell(2, [point_id, contour_start_id])
            else:
                lines.InsertNextCell(2, [point_id, point_id + 1])

            # Increment the point index
            point_id += 1

        if save_epi_contour:
            # For each point on epicardium
            N = epi_contour.shape[0]
            for i in range(N):
                y, x = epi_contour[i]

                # The world coordinate of this point
                p = np.dot(affine, np.array([x, y, z, 1]))[:3]
                epi_points.InsertNextPoint(p)
                seg_id = determine_aha_segment_id(p, lv_centre, aha_axis, part_z[z])
                points_epi_aha.InsertNextTuple1(seg_id)

                # Record the first point of the current contour
                if i == 0:
                    contour_start_id = point_epi_id

                # Add the line
                if i == (N - 1):
                    lines_epi.InsertNextCell(2, [point_epi_id, contour_start_id])
                else:
                    lines_epi.InsertNextCell(2, [point_epi_id, point_epi_id + 1])

                # Increment the point index
                point_epi_id += 1

    # Save to a vtk file
    endo_poly = vtk.vtkPolyData()
    endo_poly.SetPoints(endo_points)
    endo_poly.GetPointData().AddArray(thickness)
    endo_poly.GetPointData().AddArray(points_aha)
    endo_poly.SetLines(lines)

    writer = vtk.vtkPolyDataWriter()
    output_name = '{0}.vtk'.format(output_name_stem)
    writer.SetFileName(output_name)
    writer.SetInputData(endo_poly)
    writer.Write()

    if save_epi_contour:
        epi_poly = vtk.vtkPolyData()
        epi_poly.SetPoints(epi_points)
        epi_poly.GetPointData().AddArray(points_epi_aha)
        epi_poly.SetLines(lines_epi)

        writer = vtk.vtkPolyDataWriter()
        output_name = '{0}_epi.vtk'.format(output_name_stem)
        writer.SetFileName(output_name)
        writer.SetInputData(epi_poly)
        writer.Write()

    # Evaluate the wall thickness per AHA segment and save to a csv file
    table_thickness = np.zeros(17)
    table_thickness_max = np.zeros(17)
    np_thickness = numpy_support.vtk_to_numpy(thickness).astype(np.float32)
    np_points_aha = numpy_support.vtk_to_numpy(points_aha).astype(np.int8)

    for i in range(16):
        table_thickness[i] = np.mean(np_thickness[np_points_aha == (i + 1)])
        table_thickness_max[i] = np.max(np_thickness[np_points_aha == (i + 1)])
    table_thickness[-1] = np.mean(np_thickness)
    table_thickness_max[-1] = np.max(np_thickness)

    index = [str(x) for x in np.arange(1, 17)] + ['Global']
    df = pd.DataFrame(table_thickness, index=index, columns=['Thickness'])
    df.to_csv('{0}.csv'.format(output_name_stem))

    df = pd.DataFrame(table_thickness_max, index=index, columns=['Thickness_Max'])
    df.to_csv('{0}_max.csv'.format(output_name_stem))


def determine_aha_coordinate_system(seg_sa, affine_sa):
    """ Determine the AHA coordinate system using the mid-cavity slice
        of the short-axis image segmentation.
        """
    # Label class in the segmentation
    label = {'BG': 0, 'LV': 1, 'Myo': 2, 'RV': 3}

    # Find the mid-cavity slice
    _, _, cz = [np.mean(x) for x in np.nonzero(seg_sa == label['LV'])]
    z = int(round(cz))
    seg_z = seg_sa[:, :, z]

    endo = (seg_z == label['LV']).astype(np.uint8)
    endo = get_largest_cc(endo).astype(np.uint8)
    myo = (seg_z == label['Myo']).astype(np.uint8)
    myo = remove_small_cc(myo).astype(np.uint8)
    epi = (endo | myo).astype(np.uint8)
    epi = get_largest_cc(epi).astype(np.uint8)
    rv = (seg_z == label['RV']).astype(np.uint8)
    rv = get_largest_cc(rv).astype(np.uint8)

    # Extract epicardial contour
    contours, _ = cv2.findContours(cv2.inRange(epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    epi_contour = contours[0][:, 0, :]

    # Find the septum, which is the intersection between LV and RV
    septum = []
    dilate_iter = 1
    while len(septum) == 0:
        # Dilate the RV till it intersects with LV epicardium.
        # Normally, this is fulfilled after just one iteration.
        rv_dilate = cv2.dilate(rv, np.ones((3, 3), dtype=np.uint8), iterations=dilate_iter)
        dilate_iter += 1
        for y, x in epi_contour:
            if rv_dilate[x, y] == 1:
                septum += [[x, y]]

    # The middle of the septum
    mx, my = septum[int(round(0.5 * len(septum)))]
    point_septum = np.dot(affine_sa, np.array([mx, my, z, 1]))[:3]

    # Find the centre of the LV cavity
    cx, cy = [np.mean(x) for x in np.nonzero(endo)]
    point_cavity = np.dot(affine_sa, np.array([cx, cy, z, 1]))[:3]

    # Determine the AHA coordinate system
    axis = {}
    axis['lv_to_sep'] = point_septum - point_cavity
    axis['lv_to_sep'] /= np.linalg.norm(axis['lv_to_sep'])
    axis['apex_to_base'] = np.copy(affine_sa[:3, 2])
    axis['apex_to_base'] /= np.linalg.norm(axis['apex_to_base'])
    if axis['apex_to_base'][2] < 0:
        axis['apex_to_base'] *= -1
    axis['inf_to_ant'] = np.cross(axis['apex_to_base'], axis['lv_to_sep'])
    return axis


def determine_aha_part(seg_sa, affine_sa, three_slices=False):
    """ Determine the AHA part for each slice. """
    # Label class in the segmentation
    label = {'BG': 0, 'LV': 1, 'Myo': 2, 'RV': 3}

    # Sort the z-axis positions of the slices with both endo and epicardium
    # segmentations
    X, Y, Z = seg_sa.shape[:3]
    z_pos = []
    for z in range(Z):
        seg_z = seg_sa[:, :, z]
        endo = (seg_z == label['LV']).astype(np.uint8)
        myo = (seg_z == label['Myo']).astype(np.uint8)
        pixel_thres = 10
        if (np.sum(endo) < pixel_thres) or (np.sum(myo) < pixel_thres):
            continue
        z_pos += [(z, np.dot(affine_sa, np.array([X / 2.0, Y / 2.0, z, 1]))[2])]
    z_pos = sorted(z_pos, key=lambda x: -x[1])

    # Divide the slices into three parts: basal, mid-cavity and apical
    n_slice = len(z_pos)
    part_z = {}
    if three_slices:
        # Select three slices (basal, mid and apical) for strain analysis, inspired by:
        #
        # [1] Robin J. Taylor, et al. Myocardial strain measurement with
        # feature-tracking cardiovascular magnetic resonance: normal values.
        # European Heart Journal - Cardiovascular Imaging, (2015) 16, 871-881.
        #
        # [2] A. Schuster, et al. Cardiovascular magnetic resonance feature-
        # tracking assessment of myocardial mechanics: Intervendor agreement
        # and considerations regarding reproducibility. Clinical Radiology
        # 70 (2015), 989-998.

        # Use the slice at 25% location from base to apex.
        # Avoid using the first one or two basal slices, as the myocardium
        # will move out of plane at ES due to longitudinal motion, which will
        # be a problem for 2D in-plane motion tracking.
        z = int(round((n_slice - 1) * 0.25))
        part_z[z_pos[z][0]] = 'basal'

        # Use the central slice.
        z = int(round((n_slice - 1) * 0.5))
        part_z[z_pos[z][0]] = 'mid'

        # Use the slice at 75% location from base to apex.
        # In the most apical slices, the myocardium looks blurry and
        # may not be suitable for motion tracking.
        z = int(round((n_slice - 1) * 0.75))
        part_z[z_pos[z][0]] = 'apical'
    else:
        # Use all the slices
        i1 = int(math.ceil(n_slice / 3.0))
        i2 = int(math.ceil(2 * n_slice / 3.0))
        i3 = n_slice

        for i in range(0, i1):
            part_z[z_pos[i][0]] = 'basal'

        for i in range(i1, i2):
            part_z[z_pos[i][0]] = 'mid'

        for i in range(i2, i3):
            part_z[z_pos[i][0]] = 'apical'
    return part_z


def determine_aha_segment_id(point, lv_centre, aha_axis, part):
    """ Determine the AHA segment ID given a point,
        the LV cavity center and the coordinate system.
        """
    d = point - lv_centre
    x = np.dot(d, aha_axis['inf_to_ant'])
    y = np.dot(d, aha_axis['lv_to_sep'])
    deg = math.degrees(math.atan2(y, x))
    seg_id = 0

    if part == 'basal':
        if (deg >= -30) and (deg < 30):
            seg_id = 1
        elif (deg >= 30) and (deg < 90):
            seg_id = 2
        elif (deg >= 90) and (deg < 150):
            seg_id = 3
        elif (deg >= 150) or (deg < -150):
            seg_id = 4
        elif (deg >= -150) and (deg < -90):
            seg_id = 5
        elif (deg >= -90) and (deg < -30):
            seg_id = 6
        else:
            print('Error: wrong degree {0}!'.format(deg))
            exit(0)
    elif part == 'mid':
        if (deg >= -30) and (deg < 30):
            seg_id = 7
        elif (deg >= 30) and (deg < 90):
            seg_id = 8
        elif (deg >= 90) and (deg < 150):
            seg_id = 9
        elif (deg >= 150) or (deg < -150):
            seg_id = 10
        elif (deg >= -150) and (deg < -90):
            seg_id = 11
        elif (deg >= -90) and (deg < -30):
            seg_id = 12
        else:
            print('Error: wrong degree {0}!'.format(deg))
            exit(0)
    elif part == 'apical':
        if (deg >= -45) and (deg < 45):
            seg_id = 13
        elif (deg >= 45) and (deg < 135):
            seg_id = 14
        elif (deg >= 135) or (deg < -135):
            seg_id = 15
        elif (deg >= -135) and (deg < -45):
            seg_id = 16
        else:
            print('Error: wrong degree {0}!'.format(deg))
            exit(0)
    elif part == 'apex':
        seg_id = 17
    else:
        print('Error: unknown part {0}!'.format(part))
        exit(0)
    return seg_id


def approximate_contour(contour, factor=4, smooth=0.05, periodic=False):
    """ Approximate a contour.

        contour: input contour
        factor: upsampling factor for the contour
        smooth: smoothing factor for controling the number of spline knots.
                Number of knots will be increased until the smoothing
                condition is satisfied:
                sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) <= s
                which means the larger s is, the fewer knots will be used,
                thus the contour will be smoother but also deviating more
                from the input contour.
        periodic: set to True if this is a closed contour, otherwise False.

        return the upsampled and smoothed contour
    """
    # The input contour
    N = len(contour)
    dt = 1.0 / N
    t = np.arange(N) * dt
    x = contour[:, 0]
    y = contour[:, 1]

    # Pad the contour before approximation to avoid underestimating
    # the values at the end points
    r = int(0.5 * N)
    t_pad = np.concatenate((np.arange(-r, 0) * dt, t, 1 + np.arange(0, r) * dt))
    if periodic:
        x_pad = np.concatenate((x[-r:], x, x[:r]))
        y_pad = np.concatenate((y[-r:], y, y[:r]))
    else:
        x_pad = np.concatenate((np.repeat(x[0], repeats=r), x, np.repeat(x[-1], repeats=r)))
        y_pad = np.concatenate((np.repeat(y[0], repeats=r), y, np.repeat(y[-1], repeats=r)))

    # Fit the contour with splines with a smoothness constraint
    fx = interpolate.UnivariateSpline(t_pad, x_pad, s=smooth * len(t_pad))
    fy = interpolate.UnivariateSpline(t_pad, y_pad, s=smooth * len(t_pad))

    # Evaluate the new contour
    N2 = N * factor
    dt2 = 1.0 / N2
    t2 = np.arange(N2) * dt2
    x2, y2 = fx(t2), fy(t2)
    contour2 = np.stack((x2, y2), axis=1)
    return contour2


def atrium_pass_quality_control(label, label_dict):
    """ Quality control for atrial volume estimation """
    for l_name, l in label_dict.items():
        # Criterion 1: the atrium does not disappear at any time point so that we can
        # measure the area and length.
        T = label.shape[3]
        for t in range(T):
            label_t = label[:, :, :, t]
            area = np.sum(label_t == l)
            if area == 0:
                print('The area of {0} is 0 at time frame {1}.'.format(l_name, t))
                return False

        # Criterion 2: no fragmented segmentation
        pixel_thres = 10
        for t in range(T):
            seg_t = label[:, :, :, t]
            cc, n_cc = skimage.measure.label(seg_t == l, connectivity=2, return_num=True)
            count_cc = 0
            for i in range(1, n_cc + 1):
                binary_cc = (cc == i)
                if np.sum(binary_cc) > pixel_thres:
                    # If this connected component has more than certain pixels, count it.
                    count_cc += 1
            if count_cc >= 2:
                print('The segmentation has at least two connected components with more than {0} pixels '
                      'at time frame {1}.'.format(pixel_thres, t))
                return False

        # Criterion 3: no abrupt change of area
        A = np.sum(label == l, axis=(0, 1, 2))
        for t in range(T):
            ratio = A[t] / float(A[t - 1])
            if ratio >= 2 or ratio <= 0.5:
                print('There is abrupt change of area at time frame {0}.'.format(t))
                return False
    return True


def evaluate_atrial_area_length(label, nim, long_axis):
    """ Evaluate the atrial area and length from 2 chamber or 4 chamber view images. """
    # Area per pixel
    pixdim = nim.header['pixdim'][1:4]
    area_per_pix = pixdim[0] * pixdim[1] * 1e-2  # Unit: cm^2

    # Go through the label class
    L = []
    A = []
    landmarks = []
    labs = np.sort(list(set(np.unique(label)) - set([0])))
    for i in labs:
        # The binary label map
        label_i = (label == i)

        # Get the largest component in case we have a bad segmentation
        label_i = get_largest_cc(label_i)

        # Go through all the points in the atrium, sort them by the distance along the long-axis.
        points_label = np.nonzero(label_i)
        points = []
        for j in range(len(points_label[0])):
            x = points_label[0][j]
            y = points_label[1][j]
            points += [[x, y,
                        np.dot(np.dot(nim.affine, np.array([x, y, 0, 1]))[:3], long_axis)]]
        points = np.array(points)
        points = points[points[:, 2].argsort()]

        # The centre at the top part of the atrium (top third)
        n_points = len(points)
        top_points = points[int(2 * n_points / 3):]
        cx, cy, _ = np.mean(top_points, axis=0)

        # The centre at the bottom part of the atrium (bottom third)
        bottom_points = points[:int(n_points / 3)]
        bx, by, _ = np.mean(bottom_points, axis=0)

        # Determine the major axis by connecting the geometric centre and the bottom centre
        major_axis = np.array([cx - bx, cy - by])
        major_axis = major_axis / np.linalg.norm(major_axis)

        # Get the intersection between the major axis and the atrium contour
        px = cx + major_axis[0] * 100
        py = cy + major_axis[1] * 100
        qx = cx - major_axis[0] * 100
        qy = cy - major_axis[1] * 100

        if np.isnan(px) or np.isnan(py) or np.isnan(qx) or np.isnan(qy):
            return -1, -1, -1

        # Note the difference between nifti image index and cv2 image index
        # nifti image index: XY
        # cv2 image index: YX (height, width)
        image_line = np.zeros(label_i.shape)
        cv2.line(image_line, (int(qy), int(qx)), (int(py), int(px)), (1, 0, 0))
        image_line = label_i & (image_line > 0)

        # Sort the intersection points by the distance along long-axis
        # and calculate the length of the intersection
        points_line = np.nonzero(image_line)
        points = []
        for j in range(len(points_line[0])):
            x = points_line[0][j]
            y = points_line[1][j]
            # World coordinate
            point = np.dot(nim.affine, np.array([x, y, 0, 1]))[:3]
            # Distance along the long-axis
            points += [np.append(point, np.dot(point, long_axis))]
        points = np.array(points)
        if len(points) == 0:
            return -1, -1, -1
        points = points[points[:, 3].argsort(), :3]
        L += [np.linalg.norm(points[-1] - points[0]) * 1e-1]  # Unit: cm

        # Calculate the area
        A += [np.sum(label_i) * area_per_pix]

        # Landmarks of the intersection points
        landmarks += [points[0]]
        landmarks += [points[-1]]
    return A, L, landmarks