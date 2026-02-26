import tensorflow as tf
import numpy as np
import cv2

interpreter = tf.lite.Interpreter(model_path="/Users/yashkritsingh/Documents/Hero Mototcorp/edge_training/edge_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def tflite_predict(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img.astype(np.float32), axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    prob = output[0][0]
    
    if prob > 0.5:
        print(f"GOOD ({prob:.2f})")
    else:
        print(f"BAD ({prob:.2f})")

tflite_predict("/Users/yashkritsingh/Documents/Hero Mototcorp/edge_training/edge_dataset/val/good/000006.jpg")