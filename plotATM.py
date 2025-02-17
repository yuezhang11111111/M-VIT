import cv2
import numpy as np
import matplotlib.pyplot as plt
from MVIT import vit, utils,visualize


# Load a model
image_size = 224
classes = utils.get_imagenet_classes()
model = vit.vit_b16(
    image_size=image_size,
    activation='sigmoid',
    pretrained=True,
    include_top=True,
    pretrained_top=True
)
classes = utils.get_imagenet_classes()

# Get an image and compute the attention map
l ='/media/ubuntu/1276A91876A8FD9B/zy/WSI_path/TCGA-S4-A8RP-01.svs/TCGA-S4-A8RP-01.svs_1_1.npy'
image = utils.read(l, image_size)
attention_map = visualize.attention_map(model=model, image=image)
print('Prediction:', classes[
    model.predict(vit.preprocess_inputs(image)[np.newaxis])[0].argmax()]
)  # Prediction: Eskimo dog, husky

# Plot results
fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.axis('off')
ax2.axis('off')
ax1.set_title('Original')
ax2.set_title('Attention Map')
_ = ax1.imshow(image)
_ = ax2.imshow(attention_map)
#plt.colorbar(_ ,label='Pixel Intensity')
#plt.savefig('.jpg')
plt.show()