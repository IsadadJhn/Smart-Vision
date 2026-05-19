import cv2

face_src = cv2.CascadeClassifier("face_src.xml")
camera = cv2.VideoCapture(0)


#butuh 2 function 
#function 1 untuk deteksi wajah , function 2 untuk membuat camerabox

def face_detection(frame):  #buat deteksi wajah   , parameter frame supaya bisa membaca setiap frame dari camera
    optimized_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) #diubah ke gambar black and white supaya lebih ringan memproses gambar
    faces = face_src.detectMultiScale(optimized_frame, scaleFactor=1.1,minSize=(300,300),minNeighbors=3) #10%
    return faces
def drawer_box(frame): # buat bikin drawer box
    for x, y, w, h in face_detection(frame):
        cv2.rectangle(frame, (x,y), (x + w, y + h), (255,0,0),4 )  #BGR = blue,green,red

def close_window():
    camera.release()
    cv2.destroyAllWindows()
    exit()

def main():
    while True:
        _, frame = camera.read() #permission to using camera
        drawer_box(frame)
        cv2.imshow("Camera cantik",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_window()

if __name__ == '__main__':
    main()
