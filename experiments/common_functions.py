import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import PIL

# Helper function to preprocess the image so that it can be inputted in MobileNetV2
def preprocess(image):
  image = tf.cast(image, tf.float32)
  image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  image = image[None, ...]
  return image

# Helper function to extract labels from probability vector
def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]

def create_adversarial_pattern(input_image, input_label):

  return signed_grad
    
def show(img_rgb):
    return PIL.Image.fromarray(np.squeeze(np.array(img_rgb)))

def show_perturbation(signed_grad):
    p = 127.5*np.squeeze(np.array(signed_grad+1))
    return PIL.Image.fromarray(p.astype('uint8'))

decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

def png_to_rgb(png_filename):
    img_png = tf.io.read_file('imagenet_crops/guitar_01.png')
    return tf.io.decode_png(img_png)

def attack(img_rgb, signed_grad, bits):
    rgb = img_rgb.numpy().astype('int16')
    grad = signed_grad.numpy().astype('int16')
    scale = 2**(bits-1)
    print("adding +-", scale, "to image")
    attacked_img = rgb + scale*grad;
    attacked_img[attacked_img<0] = 0
    attacked_img[attacked_img>255] = 255
    rgb[:,:,:] = attacked_img
    return rgb.astype('uint8')

def load_model():
    pretrained_model = tf.keras.applications.MobileNetV2(include_top=True,weights='imagenet')
    pretrained_model.trainable = False
    return pretrained_model

def fgsm(pretrained_model, img_rgb):
    
    input_image = preprocess(img_rgb)
    input_probs = pretrained_model.predict(input_image)
    input_label = get_imagenet_label(input_probs)
    
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        loss = loss_object(input_probs, prediction)
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    
    return input_label, signed_grad