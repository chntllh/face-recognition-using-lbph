import numpy as np
import cv2
import os
import easygui
from tkinter import *
import tkinter.font
from PIL import Image
import csv

training_image_path = R"TrainingImage"
trained_image_path = R"TrainedImage"
hc_ff_default = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
datafile = 'data.csv'

# DetectFaces
def detectFaces(img):
    originalImage = img
    grayscaleImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    smoothGrayscaleImage = cv2.medianBlur(grayscaleImage, 7)
    faces = hc_ff_default.detectMultiScale(smoothGrayscaleImage, 1.2, 5)
    return faces


def nameExistsInData(name):
    with open(datafile) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if(row[0] == name):
                return row
        return False


def sepFaces(img, faces):
    count = int()
    for (x, y, w, h) in faces:
        name = ''
        id = ''
        seq = ''
        r = 256/w if 256/w < 256/h else 256/h
        imcv = cv2.resize(img[y:y+h, x:x+w], (int(h*r),int(w*r)))
        cv2.imshow('frame', imcv)
        name = input("Name:")
        if(name == ""):
            cv2.destroyWindow('frame')
            continue
        data = nameExistsInData(name)
        if(data == False):
            with open(datafile, mode='r') as csv_file:
                csv_reader = csv.reader(csv_file)
                count = sum(1 for row in csv_reader)
            with open(datafile, mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',')
                csv_writer.writerow([name, count, '0'])
            id = count
            seq = 0
        else:
            file = open(datafile)
            reader = csv.reader(file)
            L = []
            for row in reader:
                if row[0] == name:
                    temp = int(row[2])+1
                    seq = temp
                    id = row[1]
                    name = row[0]
                    row[2] = str(temp)
                L.append(row)
            file.close()
            file = open(datafile, 'w+', newline='')
            writer = csv.writer(file)
            writer.writerows(L)
            file.seek(0)
            reader = csv.reader(file)
            file.close()
        
        cv2.imwrite(training_image_path + os.sep + name + "." +
                    str(id) + "." + str(seq) + ".jpg", imcv)
        cv2.destroyWindow('frame')


# TrainingTest
def getImage():
    imagePaths = [os.path.join(training_image_path, f)
                  for f in os.listdir(training_image_path)]
    faces = list()
    ids = list()
    for imagePath in imagePaths:
        pilImg = Image.open(imagePath).convert('L')
        imgNp = np.array(pilImg, 'uint8')
        id = os.path.split(imagePath)[-1].split(".")[1]
        faces.append(imgNp)
        ids.append(int(id))
    return faces, ids


def trainImages():
    recognizer = cv2.face.LBPHFaceRecognizer.create()
    faces, ids = getImage()
    recognizer.train(faces, np.array(ids))
    recognizer.save(trained_image_path + os.sep + "Trainner.yml")


# Recognize
def makeDictionary():
    imagePaths = [os.path.join(training_image_path, f)
                  for f in os.listdir(training_image_path)]
    dictIdName = {}
    for imagePath in imagePaths:
        id = os.path.split(imagePath)[-1].split(".")[1]
        name = os.path.split(imagePath)[-1].split(".")[0]
        dictIdName[id] = name
    return dictIdName


def recognizeImage():
    image = cv2.imread(fileName)
    h, w = image.shape[:2]
    r = 1080/w if 1080/w < 1080/h else 1080/h
    image = cv2.resize(image, (int(w*r), int(h*r)))
    recImage = image
    faces = detectFaces(image)
    recognizer = cv2.face.LBPHFaceRecognizer.create()
    recognizer.read(trained_image_path + os.sep + "Trainner.yml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dictIdName = {}
    dictIdName = makeDictionary()
    for face in faces:
        x, y, w, h = face
        Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        cv2.rectangle(recImage, (x, y), (x+w, y+h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (0, 0, 255)
        thickness = 2
        cv2.putText(recImage, dictIdName[str(Id)], (x, y-10), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("recognized", image)
    cv2.waitKey(0)


def browse_files_button():
    global fileName
    fileName = easygui.fileopenbox()
    label_file_explorer.configure(text="File Opened: {}".format(fileName))


def detect_face_button():
    image = cv2.imread(fileName)
    faces = detectFaces(image)
    sepFaces(image, faces)


def training_button():
    trainImages()


def recognize_button():
    recognizeImage()


color_1 = '#D3CEDF'
color_2 = '#D3CEDF'
color_3 = '#F2D7D9'
color_4 = '#9CB4CC'
color_5 = '#06283D'

window = Tk()
window.title('Face Detection in Group Photo')
window.configure(bg=color_1)

font_hel = tkinter.font.Font(family="Helvetica")
font_hel_b = tkinter.font.Font(family="Helvetica", weight="bold")

label_file_explorer = Label(
    window, text="File Explorer using Tkinter", font=font_hel_b, bg=color_1, fg=color_5)
label_file_explorer.grid(row=0, column=0, columnspan=2, pady=10)

button_explore = Button(window, text="Browse Files", command=browse_files_button,
                        font=font_hel_b, bg=color_2, fg=color_5, activebackground=color_4)
button_explore.grid(row=1, column=0, columnspan=2,
                    padx=5, pady=10, sticky='news')

button_detect = Button(window, text="Detect Faces", command=detect_face_button,
                       font=font_hel_b, bg=color_2, fg=color_5, activebackground=color_4)
button_detect.grid(row=2, column=0, columnspan=2,
                   padx=5, pady=10, sticky='news')

button_training = Button(window, text="Train Faces", command=training_button,
                         font=font_hel_b, bg=color_2, fg=color_5, activebackground=color_4)
button_training.grid(row=3, column=0, columnspan=2,
                     padx=5, pady=10, sticky='news')

button_recognize = Button(window, text="Recognize Faces", command=recognize_button,
                          font=font_hel_b, bg=color_2, fg=color_5, activebackground=color_4)
button_recognize.grid(row=4, column=0, columnspan=2,
                      padx=5, pady=10, sticky='news')

# Running tkinter window
window.mainloop()
