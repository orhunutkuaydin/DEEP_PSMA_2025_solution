import os
from os.path import join, isdir
import SimpleITK as sitk
import shutil
import pandas as pd
import numpy as np
import time

import nnunet_config_paths  # sets flag for nnunet results (resources/nnUNet_data/results)

"""
Script to run inference based on trained models from DEEP-PSMA Baseline Model. Only Fold 0 included for simple testing...
This script will return a predicted TTB label as sitk image format based on PET/CT input data and defined threshold.
The "run_inference" function is imported into the inference.py script which manages most of the filehandling at runtime.

"""
# ONLY LOCAL
# os.environ["nnUNet_results"] = "/media/orhun/storage_4tb/DEEP_PSMA_MICCAI/nnUNet_data/nnUNet_results"


# nn_predict_exe='nnUNetv2_predict' #if not available in %PATH update to appropriate location
nn_predict_exe = '/home/user/.local/bin/nnUNetv2_predict'  # if not available in %PATH update to appropriate location, nn-unet executable locations if pip installed with --user flag


# nn_predict_exe='/home/user/.local/bin/nnUNetv2_predict' #if not available in %PATH update to appropriate location, nn-unet executable locations if pip installed with --user flag
##nn_predict_exe=r"F:\TotalSegmentator_win\totseg_v24\Scripts\nnUNetv2_predict.exe"

def expand_contract_label(label,distance=5.):
    """expand or contract sitk label image by distance indicated.
    negative values will contract, positive values expand.
    returns sitk image of adjusted label"""
    distance_filter = sitk.SignedMaurerDistanceMapImageFilter() #filter to calculate distance map
    distance_filter.SetUseImageSpacing(True)  #confirm use of image spacing mm
    distance_filter.SquaredDistanceOff() #default distances computed as d^2, set to linear distance
    dmap=distance_filter.Execute(label) #execute on label, returns SITK image in same space with voxels as distance from surface
    dmap_ar=sitk.GetArrayFromImage(dmap) #array of above SITK image
    new_label_ar=(dmap_ar<=distance).astype('int16') #binary array of values less than or equal to selected d
    new_label=sitk.GetImageFromArray(new_label_ar) #create new SITK image and copy spatial information
    new_label.CopyInformation(label)
    return new_label



def refine_my_ttb_label(ttb_image,pet_image,totseg_multilabel,expansion_radius_mm=5.,
                        pet_threshold_value=3.0,totseg_non_expand_values=[5],
                        normal_image=None):
    """ refine ttb label by expanding and rethresholding the original prediction
    set expansion radius mm to control distance that inferred TTB boundary is initially grown
    set pet_threshold_value to match designated value for ground truth contouring workflow
    (eg PSMA PET SUV=3).
    Includes option to avoid growing the label in certain
    tissue types in the total segmentator label (ex PSMA avoid expanding into liver, could
    include [2,3,5,21] to also avoid kidneys and urinary bladder)
    For other organ values see "total" class map from:
    https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_to_binary.py
    Lastly, possible to include the "normal" tissue inferred label from the baseline example
    algorithm and will similarly avoid expanding into this region"""
    ttb_original_array=sitk.GetArrayFromImage(ttb_image) #original TTB array for inpainting back voxels in certain tissues
    ttb_expanded_image=expand_contract_label(ttb_image,expansion_radius_mm) #expand inferred label with function above
    ttb_expanded_array=sitk.GetArrayFromImage(ttb_expanded_image) #convert to numpy array
    pet_threshold_array=(sitk.GetArrayFromImage(pet_image)>=pet_threshold_value) #get numpy array of PET image voxels above threshold value
    ttb_rethresholded_array=np.logical_and(ttb_expanded_array,pet_threshold_array) #remove expanded TTB voxels below PET threshold

    #loop through total segmentator tissue #s and use original TTB prediction in those labels
    totseg_multilabel=sitk.Resample(totseg_multilabel,ttb_image,sitk.TranslationTransform(3),sitk.sitkNearestNeighbor,0)
    totseg_multilabel_array=sitk.GetArrayFromImage(totseg_multilabel)
    for totseg_value in totseg_non_expand_values:
        #paint the original TTB prediction into the totseg tissue regions - probably of most relevance for PSMA liver VOI
        ttb_rethresholded_array[totseg_multilabel_array==totseg_value]=ttb_original_array[totseg_multilabel_array==totseg_value]

    #check if inferred normal array is included and if so set TTB voxels to background
    if normal_image is not None:
        normal_array=sitk.GetArrayFromImage(normal_image)
        ttb_rethresholded_array[normal_array>0]=0
    ttb_rethresholded_image=sitk.GetImageFromArray(ttb_rethresholded_array.astype('int8')) #create output image & copy information
    ttb_rethresholded_image.CopyInformation(ttb_image)
    return ttb_rethresholded_image



def run_inference(pt, ct, totalseg, tracer='PSMA', suv_threshold=3.0, output_fname='deep-psma_inferred.nii.gz',
                  return_ttb_sitk=False, temp_dir='temp', fold='0',
                  expansion_radius=7.):  # fold='all'
    start_time = time.time()
    print('RUNNING DEEP PSMA BASELINE')
    if isinstance(pt, str):
        pt_suv = sitk.ReadImage(pt)
    else:
        pt_suv = pt
    if isinstance(ct, str):
        ct = sitk.ReadImage(ct)
    module_dir = os.path.dirname(__file__)

    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(pt_suv)
    rs.SetDefaultPixelValue(-1000)
    ct_rs = rs.Execute(ct)
    pt_rs = pt_suv / suv_threshold
    os.makedirs(temp_dir, exist_ok=True)  # create/empty nnU-Net temporary dir
    for f in os.listdir(temp_dir):
        if isdir(join(temp_dir, f)):
            shutil.rmtree(join(temp_dir, f))
        else:
            os.unlink(join(temp_dir, f))
    # filenames to end in _0000.nii.gz and _0001.nii.gz, 0=rescaled PET, 1=CT
    os.makedirs(join(temp_dir, 'nn_input'), exist_ok=True)
    sitk.WriteImage(pt_rs, join(temp_dir, 'nn_input', 'deep-psma_0000.nii.gz'))
    sitk.WriteImage(ct_rs, join(temp_dir, 'nn_input', 'deep-psma_0001.nii.gz'))

    call = nn_predict_exe + ' -i ' + join(temp_dir, 'nn_input') + ' -o ' + join(temp_dir, 'nn_output')

    call += ' -c 3d_fullres'

    if not fold == 'all':
        call += ' -f ' + str(fold)
    elif fold == 'all':
        call += ' -f ' 'all'

    if tracer == 'PSMA':  # select which nnunet model to use based on tracer flag
        call += ' -d ' + '800'
        # add resencl
        call += ' -p nnUNetResEncUNetMPlans_192'
        call += ' -tr nnUNetTrainerMisalign2'

        print('images loaded', round(time.time() - start_time, 1))
        print('Calling nnU-Net')
        print(call)

        os.system(call)

    elif tracer == 'FDG':
        call += ' -d ' + '802'
        # add resencl
        call += ' -p nnUNetResEncUNetMPlans'
        call += ' -tr nnUNetTrainerMisalign2'

        print('images loaded', round(time.time() - start_time, 1))
        print('Calling nnU-Net')
        print(call)

        os.system(call)

    pred_label = sitk.ReadImage(join(temp_dir, 'nn_output', 'deep-psma.nii.gz'))  # label predicted by nnUNet

    pred_ttb_ar = (sitk.GetArrayFromImage(pred_label) == 1).astype('int8')  # Predicted mask region for disease
    pred_norm_ar = (sitk.GetArrayFromImage(pred_label) == 2).astype('int8')  # predicted mask region for physiological

    pred_ttb_label = sitk.GetImageFromArray(
        pred_ttb_ar)  # convert predicted TTB label to sitk format with spacing information to run grow/expansion function
    pred_ttb_label.CopyInformation(pred_label)

    pred_norm_label = sitk.GetImageFromArray(pred_norm_ar)
    pred_norm_label.CopyInformation(pred_label)

    # Your desired blocklist
    if tracer == 'PSMA':
        no_grow_ids = [2,3,5,21]
    elif tracer == 'FDG':
        no_grow_ids = []

    output_label = refine_my_ttb_label(
        ttb_image=pred_ttb_label,
        pet_image=pt_suv,
        totseg_multilabel=totalseg,
        expansion_radius_mm=expansion_radius,
        pet_threshold_value=suv_threshold,
        totseg_non_expand_values=no_grow_ids,
        normal_image=pred_norm_label
    )

    sitk.WriteImage(output_label, join(temp_dir, 'output_label.nii.gz'))
    if output_fname:
        sitk.WriteImage(output_label, output_fname)
    print('Processing complete', round(time.time() - start_time, 1))
    if return_ttb_sitk:
        return output_label
    else:
        return

#
# # Example Usage:
# ct_fname=r"/media/orhun/storage_4tb/DEEP_PSMA_MICCAI/CHALLENGE_DATA_untouched/train_0001/FDG/CT.nii.gz"
# pt_fname=r"/media/orhun/storage_4tb/DEEP_PSMA_MICCAI/CHALLENGE_DATA_untouched/train_0001/FDG/PET.nii.gz"
# pt=sitk.ReadImage(pt_fname)
# ct=sitk.ReadImage(ct_fname)
# suv_threshold = 3.0 # 3.0
#
# ttb=run_inference(pt,ct,tracer='FDG',output_fname=r"inference_test.nii.gz",
#                   return_ttb_sitk=True,
#                   fold='all',suv_threshold=suv_threshold)
#
