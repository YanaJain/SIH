import cv2
import numpy as np


#web camera+
cap =cv2.VideoCapture("Untitled video - Made with Clipchamp.mp4")
desired_fps = 30
                                                                                                                                                                                    
min_width_rect = 80  #min_width_rectangle
min_height_rect = 80 #min_heigh_rectangle

count_line_position = 550
#initialize substructor
algo = cv2.createBackgroundSubtractorMOG2()


def center_handle(x,y,w,h):
     x1=int(w/2)
     y1=int(h/2)
     cx=x+x1
     cy=y+y1
     return cx,cy
 
detect=[]

offset = 6 #allowable errorbetween pixel
counter = 0

while True:
    ret,frame1=cap.read()
    
    gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),5)
    #applying on each frame
    
    
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE,kernel)
    dilatada = cv2.morphologyEx(dilatada , cv2.MORPH_CLOSE,kernel)
    CounterShape,h = cv2.findContours(dilatada, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
    
    
    cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(205,41,144),3)
    
    
    for(i,c) in enumerate(CounterShape):     
         (x,y,w,h) =cv2.boundingRect(c)
         validate_counter = (w>= min_width_rect) and (h>= min_height_rect)
         if not validate_counter:
            continue
        
        
         cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,0,128),3)
         #cv2.putText(frame1, "vehicle"+str(counter), (x,y-20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,244,0),2) 
    
             
         center=center_handle(x,y,w,h)
         detect.append(center)
         cv2.circle(frame1,center,4,(0,0,255),-1)    
    
         for (x,y) in detect:
             if y<(count_line_position+offset)   and y>(count_line_position-offset):
                 counter+=1
                 cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(255,127,0),3)
                 detect.remove((x,y))
                 print("CROUD DETECTING:"+str(counter))
                 
                 
                    
             cv2.putText(frame1,"CROUD DETECTING:"+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)
                 
        
        
             #cv2.imshow('detecter,dilatada') 
             cv2.imshow('detector',frame1)
     
             if cv2.waitKey(1) == 13:
    
                 break
    

cv2.destroyAllWindows()
cap.release()
#('vedio.mp4.mp4')