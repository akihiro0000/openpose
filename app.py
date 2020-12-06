from flask import Flask, render_template, Response
from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
from datetime import datetime
from tf_pose.estimator import TfPoseEstimator
import json
import threading
import argparse
from threading import Thread, enumerate
from queue import Queue
import paho.mqtt.client as mqtt
import argparse
from tf_pose.networks import get_graph_path, model_wh
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--flask", action="store_true",
                    help="enable flask app")
parser.add_argument('-v', '--verbose', action="store_true",
                    required=False, default=False, help='Enable verbose output')
parser.add_argument("-i", "--ip", type=str, required=False, default=os.getenv('LISTEN_IP', '0.0.0.0'),
                    help="listen ip address")
parser.add_argument("--port", type=int, required=False, default=os.getenv('LISTEN_PORT', '8080'),
                    help="ephemeral port number of the server (1024 to 65535) default 8080")
parser.add_argument('-d', '--devno', type=int, default=os.getenv('DEVNO', '-1'),
                    help='device number for camera (typically -1=find first available, 0=internal, 1=external)')
parser.add_argument('-n', '--capture-string', type=str, default=os.getenv('CAPTURE_STRING'),
                    help='Any valid VideoCapture string(IP camera connection, RTSP connection string, etc')
parser.add_argument('-c', '--confidence', type=float,
                    default=os.getenv('CONFIDENCE', '0.3'))
parser.add_argument('-p', '--publish', action="store_true")
parser.add_argument('-s', '--sleep', type=float,
                    default=os.getenv('SLEEP', '1.0'))
parser.add_argument('--protocol', type=str,
                    default=os.getenv('PROTOCOL', 'HTTP'))
parser.add_argument('-m', '--model-name', type=str, required=False,
                    default=os.getenv('MODEL_NAME', 'ssd_mobilenet_coco'), help='Name of model')
parser.add_argument('-x', '--model-version', type=str, required=False, default=os.getenv('MODEL_VERSION', ''),
                    help='Version of model. Default is to use latest version.')
parser.add_argument('-u', '--url', type=str, required=False, default=os.getenv('TRITON_URL', 'localhost:5000'),
                    help='Inference server URL. Default is localhost:8000.')
parser.add_argument('-b', '--mqtt-broker-host', type=str, required=False, default=os.getenv('MQTT_BROKER_HOST', 'fluent-bit'),
                    help='mqtt broker host')
parser.add_argument("--mqtt-broker-port", type=int, required=False, default=os.getenv('MQTT_BROKER_PORT', '1883'),
                    help="port number of the mqtt server (1024 to 65535) default 1883")
parser.add_argument('-t', '--mqtt-topic', type=str, required=False, default=os.getenv('MQTT_TOPIC', '/demo'),
                    help='mqtt broker topic')
parser.add_argument('-ann', '--armnn', action="store_true")
parser.add_argument('-db1', '--detect-car', action="store_true")
parser.add_argument('-db2', '--detect-person', action="store_true")
parser.add_argument('-db3', '--detect-bus', action="store_true")
parser.add_argument('-db4', '--detect-bicycle', action="store_true")
parser.add_argument('-db5', '--detect-motorcycle', action="store_true")
args = parser.parse_args()

outputFrame = None
outputArray = None
lock = threading.Lock()
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
                    


def detection_loop():
    global outputFrame,outputArray,lock
    
    w, h = model_wh("432x368")
    e = TfPoseEstimator(get_graph_path("mobilenet_thin"), target_size=(w, h))
    cam = cv2.VideoCapture(0)
    cam.set(3,1280)
    cam.set(4,720)
    print("------------openpose_finished---------")
    try:
        for img in getframe(cam,e,w,h):
            if img is not None:
                with lock:
                    outputFrame = img
                    outputArray = "a"
                    mystr = '{"timestamp":"2020-11-16 03:14:43.430204","nodeid":"0","nodeid":"0","sensor":"image","car_count":"0"}'
                    mqtt_client.publish("{}/{}".format(args.mqtt_topic,'car_count'), str(mystr))
                    mqtt_client.publish("{}/{}".format(args.mqtt_topic,'car_count'), str(outputArray))
            else:
                print("--------------Thread_finished(ctl + C)-----------------")
                
    except:
        os._exit(1)
        
def getframe(cam,e,w,h):
    while True:
        ret, frame_read = cam.read()
        humans = e.inference(frame_read, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
        image = np.zeros(frame_read.shape)

        image = TfPoseEstimator.draw_humans(frame_read, humans, imgcopy=False)
        ret1, jpeg = cv2.imencode('.jpg', image)
        cv2.waitKey(1)
        jpeg = jpeg.tobytes()
        yield jpeg
    cam.release()
    yield  None,None

def generate():
    global outputFrame,outputArray,lock

    while True:
        with lock:
            if outputFrame is None:
                continue
        if outputArray!=[]:
            pass
            #print(outputArray)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + outputFrame + b'\r\n\r\n')
               
if __name__ == '__main__':
    mqtt_client = mqtt.Client()
    mqtt_client.connect(args.mqtt_broker_host, args.mqtt_broker_port, 60)
    mqtt_client.loop_start()

    t = threading.Thread(target=detection_loop)
    t.start()
    
    app.run(host='0.0.0.0',threaded=True,port=1110,debug=False)
    print("--------------finished-----------------")
    mqtt_client.disconnect()
