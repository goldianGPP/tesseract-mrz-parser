from PIL import Image
import os

# Define the input image file name and extension
input_image_file = "resized_2.jpg"

# Load the image
original_image = Image.open(input_image_file)

# Create a directory to save resized images if it doesn't exist
output_folder = "resized_images"
os.makedirs(output_folder, exist_ok=True)

# Initialize a variable to store the current image
current_image = original_image

# Resize the image 5 times by 10% and save each resized image
for i in range(1, 20):
    # Calculate the new size by reducing 10% each time
    new_width = int(current_image.width * 0.9)
    new_height = int(current_image.height * 0.9)
    
    # Resize the image
    current_image = current_image.resize((new_width, new_height))
    
    # Save the resized image with a new file name
    output_file = os.path.join(output_folder, f"resized_{i}.jpg")
    current_image.save(output_file)
    
    print(f"Saved resized image {i} as {output_file}")

# Close the original image
original_image.close()
