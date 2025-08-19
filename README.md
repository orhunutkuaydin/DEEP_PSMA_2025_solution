# DEEP-PSMA 2025 training and inference code 
Orhun Utku Aydin

This contribution details our methodological approach to the DEEP-PSMA challenge as part of MICCAI 2025, focusing on whole body disease burden segmentation on PSMA PET/CT. Our method follows the recommendations by the AUTOPET 3 winning solution and by the DEEP-PSMA challenge organizers for preprocessing, model training and postprocessing. We base our solution on the self-configuring ResEncM variant of nnUnet. The provided training set of 100 patients was used to train both a tracer agnostic model and tracer specific model for the FDG PET. Validation was performed on the unseen validation set of 10 patients. In summary: (1) we use a single tracer-agnostic model to segment PSMA PET lesions and a tracer specific model for FDG PET lesions, (2) we increase the patch size to 192x192x192 for better contextual understanding of the tracer agnostic model, (3) use misalignment data augmentations for the CT channel (4) predict physiological tracer uptake as a second channel in a multiclass segmentation setting (5) use recommended postprocessing strategies. The implemented methods and design choices led to improvements in the challenge metrics in our preliminary testing on the validation set. 

NOTE: We use 3d fullres models and train for 1000 epochs.

## References
- Isensee, F., Jaeger, P.F., Kohl, S.A.A., Petersen, J., Maier-Hein, K.H., 2021. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nat. Methods 18, 203–211. https://doi.org/10.1038/s41592-020-01008-z 
- Kovacs, B., Netzer, N., Baumgartner, M., Schrader, A., Isensee, F., Weißer, C., Wolf, I., Görtz, M., Jaeger, P.F., Schütz, V., Floca, R., Gnirs, R., Stenzinger, A., Hohenfellner, M., Schlemmer, H.-P., Bonekamp, D., Maier-Hein, K.H., 2023. Addressing image misalignments in multi-parametric prostate MRI for enhanced computer-aided diagnosis of prostate cancer. Sci. Rep. 13, 19805. https://doi.org/10.1038/s41598-023-46747-z 
- Rokuss, M., Kovacs, B., Kirchhoff, Y., Xiao, S., Ulrich, C., Maier-Hein, K.H., Isensee, F., 2024. From FDG to PSMA: A Hitchhiker’s Guide to Multitracer, Multicenter Lesion Segmentation in PET/CT Imaging. https://doi.org/10.48550/arXiv.2409.09478 
- Sartor, O., Bono, J. de, Chi, K.N., Fizazi, K., Herrmann, K., Rahbar, K., Tagawa, S.T., Nordquist, L.T., Vaishampayan, N., El-Haddad, G., Park, C.H., Beer, T.M., Armour, A., Pérez-Contreras, W.J., DeSilvio, M., Kpamegan, E., Gericke, G., Messmann, R.A., Morris, M.J., Krause, B.J., 2021. Lutetium-177–PSMA-617 for Metastatic Castration-Resistant Prostate Cancer. N. Engl. J. Med. 385, 1091–1103. https://doi.org/10.1056/NEJMoa2107322 


