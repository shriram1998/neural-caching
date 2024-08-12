from PIL import Image

# Load the image
image_path = 'results/poster.jpg'  # Replace with your image path
img = Image.open(image_path)

# Resize the image
max_size = (6012, 6012)
img.thumbnail(max_size, Image.LANCZOS)

# Save the resized image
resized_image_path = 'resized_image.png'  # Replace with your desired output path
img.save(resized_image_path)

print(f"Image resized and saved as {resized_image_path}")
