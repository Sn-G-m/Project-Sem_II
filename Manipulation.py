def manipulate(dict_paths, parent_folder):
    iterator_outside = 1
    import os
    import cv2
    for key in dict_paths:
        iterator_inside = 1
        image = cv2.imread(dict_paths[key])
        for i in range(2, 3):
            j = i / 2
            # Apply the brightness transformation
            transformed_image_b = cv2.convertScaleAbs(image, alpha=j)
            transformed_image_d = cv2.convertScaleAbs(image, alpha=1 / j)
            transformed_image_c = cv2.convertScaleAbs(image, beta=60 * j)
            file_name = os.path.basename(dict_paths[key])
            # Save the transformed images
            name_1 = "Bright" + str(i - 1) + file_name
            name_2 = "Dark" + str(i - 1) + file_name
            name_3 = "Contrast" + str(i - 1) + file_name
            print("namings")
            print(name_1)
            print(dict_paths[key])
            # Path specifications
            transformed_image_path_1 = os.path.join(parent_folder, name_1)
            transformed_image_path_2 = os.path.join(parent_folder, name_2)
            transformed_image_path_3 = os.path.join(parent_folder, name_3)
            print("paths")
            print(parent_folder)
            # Writing into the file
            cv2.imwrite(transformed_image_path_1, transformed_image_b)
            cv2.imwrite(transformed_image_path_2, transformed_image_d)
            cv2.imwrite(transformed_image_path_3, transformed_image_c)
            print("Iteration: ", iterator_outside, " ; ", iterator_inside)
            iterator_inside += 1
        iterator_outside += 1
