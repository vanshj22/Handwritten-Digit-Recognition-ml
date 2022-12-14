import pygame, sys
from pygame import Color
import numpy as np
import cv2
import joblib

pygame.init()

pixels = 640
pixels1 = 480
drawing = False
BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255,0,0)
cord_x = []
cord_y = []
boundary = 5

number_labels = {0:"Zero", 1:"One", 2:"Two", 3:"Three", 4:"Four",5:"Five", 6:"Six", 7:"Seven",8:"Eight", 9:"Nine"}

load_model = joblib.load('models/rfc_model')    # random forest classifier, support vector classification, tensorflow


screen = pygame.display.set_mode((pixels,pixels1))
#clock = pygame.time.Clock()
txt_font = pygame.font.Font('fonts/04B_19.TTF',25)    #add font 

# pygame.display.set_caption('Digit Canvas')
# Icon = pygame.image.load('logo.png') 
# pygame.display.set_icon(Icon)
predict =True
redt = True
while predict:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACED:
                screen.fill(BLACK)
        
        if event.type == pygame.MOUSEMOTION and drawing:
            x_pos,y_pos = pygame.mouse.get_pos() #event.get..
            pygame.draw.circle(screen, WHITE, (x_pos, y_pos), 4, 0)
            
            cord_x.append(x_pos)
            cord_y.append(y_pos)
        
        #if event.type == pygame.MOUSEBUTTONDOWN:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                drawing == True

        #if event.type == pygame.MOUSEBUTTONUP:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                drawing == False
                #cord_x = sorted(cord_x)
                #cord_y = sorted(cord_y)
                
                rect_min_x , rect_max_x = max(cord_x[0] - boundary,0), min(pixels, cord_x[-1] + boundary)
                rect_min_y , rect_max_y = max(cord_y[0] - boundary,0), min(cord_y[-1] + boundary,pixels)

                cord_x = []
                cord_y = []
                
                img_arr = np.array(pygame.PixelArray(screen))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

                if True:
                    img = cv2.resize(img_arr,(1,784))
                    img = np.pad(img,(10,10), 'constant',constant_values = 0)
                    img = cv2.resize(img,(28,28))/255
                    
                    label = str(number_labels[np.argmax(load_model.predict (img))])

                    textSurface = txt_font.render(label, True, RED, WHITE)
                    textRecobj = np.testing.get_rect()
                    textRecobj.left , textRecobj.bottom = rect_min_x, rect_max_y

                    screen.blit(textSurface,textRecobj) 
 
    pygame.display.update()
    #clock.tick(120)



