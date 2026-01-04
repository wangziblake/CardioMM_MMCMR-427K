"""
Data mapping, including modality_mapping, medcon_mapping, lifespan_mapping, etc.
Created on 2025/09/05
@author: Zi Wang
Email: Zi Wang (zi.wang@imperial.ac.uk)
If you want to use this code, please cite our relevant papers in the GitHub page.
"""

import os
import re


def extract_scanattrs(save_path):
    # save_path: "../Cine/TrainingSet/h5_FullSample/Center001/Siemens_30T_Vida/P001/cine_sax.mat"
    # save_path: "../Cine/TestSet/xxx/Center001/Siemens_30T_Vida/P001/cine_sax.mat"
    path_parts = save_path.split(os.sep)
    center_attrs = path_parts[-4]  # 'Center001'
    scanner_attrs = path_parts[-3]  # 'Siemens_30T_Vida'

    # Split '_' to obtain 'Siemens', '30T', 'Vida'
    vendor, field_strength, machine_type = scanner_attrs.split('_')
    if field_strength == '15T': field_strength = '1.5T'
    elif field_strength == '30T': field_strength = '3.0T'
    elif field_strength == '50T': field_strength = '5.0T'
    elif field_strength == '055T': field_strength = '0.55T'

    return center_attrs, vendor, field_strength, machine_type


def datamapping_from_filename(filepath):
    # filepath: "../Cine/TrainingSet/h5_FullSample/Center001/Siemens_30T_Vida/P001/cine_sax.mat"
    # filepath: "../Cine/TestSet/xxx/Center001/Siemens_30T_Vida/P001/cine_sax.mat"

    # Add attributes to the dataset attrs
    # TODO: get more information from multi-vendor and multi-disease dataset '_info.csv'
    attrs_center, attrs_vendor, attrs_field, attrs_scanner = extract_scanattrs(filepath)

    modality_mapping = {
        'cine': ('cine', {'sax': 'short-axis', 'lax': 'long-axis', 'ot': 'outflow tract'}),
        'T1map': ('T1 mapping', {'sax': 'short-axis', 'lax': 'long-axis'}),
        'T1mappost': ('T1 mapping post', {'sax': 'short-axis', 'lax': 'long-axis'}),
        'T1rho': ('T1 rho mapping', {'sax': 'short-axis', 'lax': 'long-axis'}),
        'T2map': ('T2 mapping', {'sax': 'short-axis', 'lax': 'long-axis'}),
        'T2smap': ('T2 star mapping', {'sax': 'short-axis', 'lax': 'long-axis'}),
        'aorta': ('cine', {'sag': 'sagittal aorta', 'tra': 'transversal aorta'}),
        'tagging': ('tagging', {'sax': 'short-axis', 'lax': 'long-axis'}),
        'flow2d': ('phase contrast', {'inplane': 'in-plane', 'throughplane': 'through-plane', 'through': 'through-plane'}),
        'perfusion': ('perfusion', {'sax': 'short-axis', 'lax': 'long-axis'}),
        'lge': ('late Gadolinium enhancement', {'sax': 'short-axis', 'lax': 'long-axis'}),
        'blackblood': ('blackblood', {'sax': 'short-axis', 'lax': 'long-axis'}),
        'T1w': ('T1 weighted', {'sax': 'short-axis', 'lax': 'long-axis'}),
        'T2w': ('T2 weighted', {'sax': 'short-axis', 'lax': 'long-axis'}),
    }

    # TODO: get more disease information from patient clinical report
    medcon_mapping = {
        'NC': 'normal control',
        'HCM': 'hypertrophic cardiomyopathy',
        'MI': 'myocardial infarction',
        'DCM': 'dilated cardiomyopathy',
        'CAD': 'coronary artery disease',
        'CHD': 'congenital heart disease',
        'ARR': 'arrhythmia',
    }

    lifespan_mapping = {
        'Adult': 'Adult',
        'Pediatric': 'Pediatric',
        'Fetal': 'Fetal',
    }

    for keyword, (modality, views) in modality_mapping.items():
        if keyword in filepath:
            attrs_modality = modality
            tempview = next((view_value for view_keyword, view_value in views.items() if view_keyword in filepath), None)
            if tempview is None:
                if keyword in ['T1map', 'T1mappost', 'T1rho', 'T2map', 'T2smap', 'tagging',
                               'perfusion', 'blackblood', 'T1w', 'T2w']:
                    tempview = 'short-axis'
                elif keyword in ['flow2d']:
                    tempview = 'through-plane'
                else:
                    tempview = 'unknown'
            attrs_view = tempview
            break
    else:
        raise ValueError('unknown data type')

    tempmedcon = next(
        (medcon_value for medcon_keyword, medcon_value in medcon_mapping.items() if medcon_keyword in filepath), None)
    if tempmedcon is None:
        if 'Center014' in filepath or 'Center015' in filepath:  # TODO: Need to check sometimes
            tempmedcon = 'normal control'
        else:
            tempmedcon = 'unknown'
    attrs_medcon = tempmedcon

    # TODO: get more lifespan information from patient clinical report
    attrs_lifespan = next((lifespan_value for lifespan_keyword, lifespan_value in lifespan_mapping.items() if lifespan_keyword in filepath), 'Adult')

    return attrs_center, attrs_vendor, attrs_field, attrs_scanner, attrs_modality, attrs_view, attrs_medcon, attrs_lifespan


def get_metadata_attribute_from_filename(filename):
    # filename: "../Cine/TestSet/xxx/Center001/Siemens_30T_Vida/P001/cine_sax.mat"
    attrs = {}

    # Add attributes to the dataset attrs
    (attrs['center'], attrs['vendor'], attrs['field'], attrs['scanner'],
     attrs['modality'], attrs['view'],
     attrs['medcon'], attrs['lifespan']) = datamapping_from_filename(filename)

    # Add undersampling scenarios to the dataset attrs
    mask_types = {
        'Uniform': 'uniform',
        'Random': 'random',
        'Gaussian': 'random',  # 'Gaussian' is also mapping to 'random'
        'Radial': 'radial',
        'ktUniform': 'uniform',
        'ktGaussian': 'random',  # 'ktGaussian' is also mapping to 'random'
        'ktRadial': 'radial'
    }

    for pattern, mask_type in mask_types.items():
        if pattern in filename:
            match = re.search(rf'{pattern}(\d+)?', filename)
            attrs['mask_type'] = mask_type
            attrs['acceleration'] = match.group(1) if match else None
            break
    else:
        raise ValueError('unknown pattern type')

    return attrs


