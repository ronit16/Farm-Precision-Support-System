import os
import tifffile

input_folder = "data"  
output_folder = "data"  
subregion_size = 512

def extract_subregions(input_path, output_folder, subregion_size=512):
    image = tifffile.imread(input_path)
    height, width = image.shape[:2]

    subregion_count = 1  

    for y in range(0, height, subregion_size):
        for x in range(0, width, subregion_size):
            top = y
            left = x
            bottom = min(y + subregion_size, height)
            right = min(x + subregion_size, width)

            # if bottom - top < subregion_size or right - left < subregion_size:
            #     continue

            subregion = image[top:bottom, left:right]

            output_filename = f"{os.path.splitext(os.path.basename(input_path))[0]}_{subregion_count}.tif"
            output_path = os.path.join(output_folder, output_filename)
            tifffile.imwrite(output_path, subregion)

            subregion_count += 1  


for filename in os.listdir(input_folder):
    if filename.endswith(".tif") or filename.endswith(".tiff"):
        input_path = os.path.join(input_folder, filename)
        extract_subregions(input_path, output_folder, subregion_size)