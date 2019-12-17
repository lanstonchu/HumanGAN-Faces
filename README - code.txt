The paper:

   "0 - HumanGAN for Human Faces - Lanston Hau Man Chu.pdf"

--------------------------------------------------

Before you run the code:

1. in the folder ./HumanGAN-Faces/large_files_size, you can see several zip files with names "large_files_size.zip.00x"
2. unzip these files to obtain 3 files:
	a. inv_Sigma_from_Numpy.csv
	b. inv_Sigma_theta.mat
	c. model_weights.mat
3. Put the above files in the folder ./mat_files

--------------------------------------------------

To run the codes:

- Data sampling:
1. run a1_combine_image.m to randomly sample 
2. run a1b_select_image.m to kick out unwanted images
3. run a1_combine_image.m again to combine the wanted images

- PCA (Eigenfaces):
4. run a2_PCAface.m on faces to get eigenfaces/eigenvalues
5. run a3_synthesize_faces.m to synthesize some faces to see whether the eigenfaces/eigenvalues are probably prepared

- Training Phase
6. run a4_GAN.m to construct the GAN and do training

- Human Participants Phase
7. participants run a5_human_tasks.m to record their responses in the folder ./human_response
8. participants send the responses files back to the experimenter

- AHGD (after-human gradient descent)
9. run a6a_tidy_up_D_scores.m to tidy up and combine the collected response files
10. run a6b_partial_V_partial_G.m to get partial_V_partial_G
11. run a7_grad_descent.m to do AHGD

--------------------------------------------------

Notes:

1. inv_Sigma_theta.mat is for comparison purpose only, which won't be used in the code. Instead, inv_Sigma_from_Numpy.csv would be imported by the Matlab code.